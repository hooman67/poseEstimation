import sys
import time as tt


from helpers import *
from configs_hydraulic import *

# This is needed if the notebook is stored in the object_detection folder.
sys.path.append(".")
#sys.path.append("/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800")
sys.path.append(mainPath)




refKey = references2use[0]
referenceRatiosDict = getReferenceRatios(refKey)




#############################################################################
############   Load JSON prediciton results from disc into dict  ############
#############################################################################
rawFramesDir = mainPath + 'Frame/'
rejectedPredsDir = mainPath + 'rejected_notAllTeeth'

resultsDict = loadResults(wmsDir)
paresedResultsDict = parseResults(resultsDict, numberOfTeeth= NUMBER_OF_TEETH)

filteredResultsDict = reject1_badBoxesAndLandmarks(
    paresedResultsDict,
    rejectedPredsDir,
    NUMBER_OF_TEETH,
    minToothBoxDistanceAllowed,
    numberOflandmarksIncludingToothTip,
    lanmark2farFromBox_epsilon,
    landmarks2close2eachother_epsilon,
    verbose = verboseAboutRejections)

keyT = mainPath.split('/')[-2]
print('\nfinalStats:')
print('---resultsDict length:  '  + str(len(resultsDict[keyT])))
print('---paresedResultsDict length:  ' + str(len(paresedResultsDict[keyT])))
print('---filteredResultsDict length:  ' + str(len(filteredResultsDict[keyT])))
#############################################################################



#############################################################################
## Fit curves to all landmarks, calculate curve derives, save the vis img  ##
#############################################################################
for resKey in filteredResultsDict.keys():
    for time in filteredResultsDict[resKey].keys():
        fileName = filteredResultsDict[resKey][time]['fileName'].split('/')[-1]
        
        fittC = fitCurve2keypoints(filteredResultsDict[resKey][time], numberOfTeeth, 'keypoints_')
        
        filteredResultsDict[resKey][time]['fittedCurves'] = fittC
        filteredResultsDict[resKey][time]['2ndDerivFittedCurves'] = get2ndDerivativeOfCurves(fittC)
        
        inImage = cv2.imread(rawFramesDir + fileName)
        if inImage is None:
            print('ERROR: couldnot open image:\n' + str(rawFramesDir + fileName))
            break

        outImage =  draw_all_keypoints_boxes_andCurves(inImage, filteredResultsDict[resKey][time], numberOfTeeth)

        cv2.imwrite(path2saveCurves + fileName, outImage)
        


    print('\nfor results set: ' + resKey + '\n---Saved the calculated curves for  ' + str(len(os.listdir(path2saveCurves)))  + '   frames.')
#############################################################################



#############################################################################
############################ Rejects 2 to 4  ################################
#############################################################################
path2saveRejectedCurves = mainPath + 'rejectedCurves/'

cleanedUpResultsDict = reject2_soft_removeBadCurves(
    filteredResultsDict,
    path2saveCurves,
    path2saveRejectedCurves,
    curvDerivTreshDic,
    verbose=False
)

keyT = mainPath.split('/')[-2]
print('\nfinalStats:')
print('---filteredResultsDict length:  '  + str(len(filteredResultsDict[keyT])))
print('---cleanedUpResultsDict length:  ' + str(len(cleanedUpResultsDict[keyT])))




# reject3 Replace landmakrs that are too far from their curves with their projections on the curve
path2saveRejectedLandmarks = mainPath + 'rejectedLandmakrs/'

reject3_soft_replaceLandmarks2farFromCurve(
    cleanedUpResultsDict,
    path2saveCurves,
    path2saveRejectedLandmarks,
    maxDistanceBtwLandmarkAndCurve_replacement,
    maxDistanceBtwLandmarkAndCurve_rejection,
    maxItr_findingShortestDistance2Curve,
    replaceBadLandmarkWithProjection,
    verbose=False
)



# add secondary confidence based on avg of #landmarks for specific tooth and total #landmarks for log
nbOfProcessedTeeth = 0

for resKey in cleanedUpResultsDict.keys():
    for time in cleanedUpResultsDict[resKey].keys():
        cleanedUpResultsDict[resKey][time]['secondaryConfidences'] = {}
        
        cleanedUpResultsDict[resKey][time]['logConfidence'] = sum(
            cleanedUpResultsDict[resKey][time]['confidences'].values()
        ) / TOTAL_EXPECTED_LANDMARKS_ON_LOG
        
        for toothKey in cleanedUpResultsDict[resKey][time]['confidences'].keys():
            cleanedUpResultsDict[resKey][time]['secondaryConfidences'][toothKey] =\
            ( weightOfToothConfidence * cleanedUpResultsDict[resKey][time]['confidences'][toothKey] / numberOflandmarksIncludingToothTip + weightOfLogConfidence * cleanedUpResultsDict[resKey][time]['logConfidence'] ) / (weightOfToothConfidence + weightOfLogConfidence)
            
            nbOfProcessedTeeth += 1
            
            
    print('for resKey ' + resKey + ' added secondary confidences for  ' + str(nbOfProcessedTeeth) + '  teeth out of the total of  ' + str(len(cleanedUpResultsDict[resKey]) * NUMBER_OF_TEETH) + '  teeth that we should have.')



# reject4 Remove logs where we don't have all toothTips, or enough landmarks for registeration
path2saveRejectedLogs = mainPath + 'rejected_notEnoughLandmarks/'

finalResultsDict = reject4_notEnoughValidLandmarks(
    cleanedUpResultsDict,
    path2saveRejectedLogs,
    minNbOfDetectedPointsForTTandLS,
    minNbOfDetectedPointsForOtherLandmarkTypes,
    numberOfTeeth= NUMBER_OF_TEETH
)
#############################################################################




#############################################################################
############ Register all frames to multiple common references  #############
#############################################################################
numberOfTeeth = NUMBER_OF_TEETH
numberOfLandmarks = numberOflandmarksIncludingToothTip

for resKey in finalResultsDict.keys():
    numberOfProcessesFrames = 0
    totalNbOfRegisterations = 0
    
    for time in finalResultsDict[resKey].keys():
        numberOfProcessesFrames += 1
        numberOfSuccessfulRegistrationsForThisFrame = 0
        finalResultsDict[resKey][time]['registrations'] = {}
            
        for refKey in references2use:
            finalResultsDict[resKey][time]['registrations'][refKey] = {}
            
            
            path2RefImage = mainPath + 'referenceFrames/images/' + refKey + '.png'
            path2Reflabel = mainPath + 'referenceFrames/labels/' + refKey + '_landmarkCoords.xml'
            refkeyPointsDic = getRefDict(path2Reflabel, path2RefImage)
            
        
            resultsArRigid, regError = getRegisteredPointsV3(
                refkeyPointsDic,
                finalResultsDict[resKey][time],
                numberOfTeeth
            )
            
            if len(resultsArRigid) > 0:
            
                if len(resultsArRigid) > 1:
                    numberOfSuccessfulRegistrationsForThisFrame += 1
                    totalNbOfRegisterations += 1

                for i in range(numberOfTeeth):
                    #rigid registration
                    key2storRigid = 'rigid_keypointsForTooth_' + str(i+1)
                    finalResultsDict[resKey][time]['registrations'][refKey][key2storRigid] = resultsArRigid[i]


                for j in range(numberOfLandmarks):
                    #rigid registration
                    key2storRigid = 'rigid_keypoints_' + str(j+1)
                    finalResultsDict[resKey][time]['registrations'][refKey][key2storRigid] = [p[j] for p in resultsArRigid]
                    
            else:
                print('could not register this frame.')
                
                
    print('\nfor results set: ' + resKey + '\n---found a total number of  ' + str(totalNbOfRegisterations) + '  registrations out of the  ' + str(numberOfProcessesFrames * len(references2use)) +'  registrations that were expected based on the number of processed frames and refernce images provided.')
#############################################################################



#############################################################################
######## Fit curves to registered landmarks and calculate derivatives  ######
#############################################################################
numberOfTeeth = NUMBER_OF_TEETH

for resKey in finalResultsDict.keys():
    totalRegisteredFrames = 0
    totalNbOfImagesWithFittedCurves = 0
    for time in finalResultsDict[resKey].keys():
        for refKey in references2use:
            
            if refKey in finalResultsDict[resKey][time]['registrations'].keys():
                totalRegisteredFrames += 1
                fileName = finalResultsDict[resKey][time]['fileName'].split('/')[-1]
                
                if len(finalResultsDict[resKey][time]['registrations'][refKey]) > 0:
                    rigid_fittC = fitCurve2keypoints(
                        finalResultsDict[resKey][time]['registrations'][refKey],
                        numberOfTeeth,
                        'rigid_keypoints_'
                    )

                    if len(rigid_fittC) > 1:
                        totalNbOfImagesWithFittedCurves += 1

                    finalResultsDict[resKey][time]['registrations'][refKey]['rigid_fittedCurves'] = rigid_fittC

                    finalResultsDict[resKey][time]['registrations'][refKey]['rigid_2ndDerivFittedCurves'] = get2ndDerivativeOfCurves(rigid_fittC)
                    #affine_fittC = fitCurve2keypoints(finalResultsDict[resKey][time], numberOfTeeth, 'affine_keypoints_')
                    #finalResultsDict[resKey][time]['registrations'][refKey]['affine_fittedCurves'] = affine_fittC
                    #finalResultsDict[resKey][time]['registrations'][refKey]['affine_2ndDerivFittedCurves'] = get2ndDerivativeOfCurves(affine_fittC)
                
    print('\nfor results set: ' + resKey + '\n---fitted at least one curve to  ' + str(totalNbOfImagesWithFittedCurves) + '  registered frames out of the total of  ' + str(totalRegisteredFrames) +'  registered frames that were processed.')
#############################################################################



#############################################################################
############## visualize all registered and original landmarks  #############
#############################################################################
path2visFinal = mainPath + 'finalVis/'

for resKey in finalResultsDict.keys():
    for time in finalResultsDict[resKey].keys():
        
        fileName = finalResultsDict[resKey][time]['fileName'].split('/')[-1]
        inImage = cv2.imread(rawFramesDir + fileName)
        
        outImage =  draw_all_keypoints_boxes_andCurves(
            inImage,
            finalResultsDict[resKey][time],
            numberOfTeeth=NUMBER_OF_TEETH,
            refKey=refKey,
            regTypeKeyword='',
            drawOnlyValidated=True,
            doNotDrawBoxes=True
        )
        
        cv2.putText(outImage,'validInputFrame', (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2, 0)
                       
        for refKey in references2use:
            if refKey in finalResultsDict[resKey][time]['registrations'].keys():
            
                path2RefImage = mainPath + 'referenceFrames/images/' + refKey + '.png'
                path2Reflabel = mainPath + 'referenceFrames/labels/' + refKey + '_landmarkCoords.xml'
                refkeyPointsDic = getRefDict(path2Reflabel, path2RefImage)

                #uncomment this and comment the one below to see the ref image
                #refVis = visualizeRefDict(refkeyPointsDic, path2RefImage)
                #refVis = cv2.imread(wmsDir + fileName)


                refImage = cv2.imread(path2RefImage)

                if refImage is None or inImage is None:
                    print("couldn't read refImage or inImage for:")
                    print(path2RefImage)
                    print(rawFramesDir + fileName)
                else:
                    
                    outImageRigid =  draw_all_keypoints_boxes_andCurves(
                        refImage,
                        finalResultsDict[resKey][time],
                        numberOfTeeth=NUMBER_OF_TEETH,
                        refKey=refKey,
                        regTypeKeyword='rigid_',
                        drawOnlyValidated=False,
                        doNotDrawBoxes=True
                    )
                    
                    cv2.putText(outImageRigid,refKey, (30,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2, 0)



                    if not outImageRigid is None:
                        outImage = np.concatenate((outImage, outImageRigid), axis=1)
                        
            else:
                print("\nrefKey:  " + str(refKey) + " did not exist for:\n" + str(fileName) + "\n")
                    
        cv2.imwrite(path2visFinal + fileName, outImage)
        
        
    print('\nfor results set: ' + resKey + '\n---Saved the final visualized images for  ' + str(len(os.listdir(path2visFinal)))  + '   frames.')
#############################################################################




# get all the lengths and reject frames where ratio of test image are too far from ratios of reference
path2saveRejectedBadRatio = mainPath + 'rejected_badRatio/'

lengthsAndLandmarksDict = getAllLengthsAndLandmarks(
    finalResultsDict, NUMBER_OF_TEETH,
    referenceRatiosDict,
    maxAllowedDistBtwRatiosDict,
    maxAllowed_detectedAboveRefTT,
    minAllowed_toothLength,
    maxAllowed_toothLength,
    path2saveRejectedBadRatio,
    verbose= False
)










####################################################################################################
####################################################################################################
####################################################################################################
resKey_glb = list(lengthsAndLandmarksDict.keys())[0]
toothNbKey_glb = 'tooth_' + str(0 + 1) + '_info'  
regTypeKeyWord_glb = 'rigid_keypointsForTooth_'
#refKey_glb = 'WMDL_2018.09.11_06.21.57'
refKey_glb = refKey
infoKey_glb = 'all_tt2ls_dict'

fittendLineType_glb = 'numpy.lib.polynomial'

degreeOfPolyn_glb = 1
toothShouldBeChangedLength_glb= 50



def getSelectedFinalLengths(endIndex):
    minAllowedConfidence = 0.7  #1 to 5 for conf1 --- 0-1 for conf 2 and 3
    filterBasedOnConf = 2    # 1 for conf1 2 for conf2, 3 for logConf


    #filter based on Confidence
    selectedLengths = []
    selectedTimes = []
    selectedConfidence = []

    #for i in range(len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'])):
    #for i in range(endIndex):
    for i in range(endIndex):

        if filterBasedOnConf == 1:
            if i < len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences']):
                if lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences'][i] >= minAllowedConfidence:

                    selectedLengths.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['lengths'][i]
                    )
                    selectedTimes.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'][i]
                    )
                    selectedConfidence.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences'][i]
                    )
                
        elif filterBasedOnConf == 2:
            if i < len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['secondaryConfidences']):
                if lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['secondaryConfidences'][i] >= minAllowedConfidence:

                    selectedLengths.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['lengths'][i]
                    )
                    selectedTimes.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'][i]
                    )
                    selectedConfidence.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['secondaryConfidences'][i]
                    )
                
                
        elif filterBasedOnConf == 3:
            if i < len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['logConfidence']):
                if lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['logConfidence'][i] >= minAllowedConfidence:

                    selectedLengths.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['lengths'][i]
                    )
                    selectedTimes.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'][i]
                    )
                    selectedConfidence.append(
                        lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['logConfidence'][i]
                    )



    return selectedLengths, selectedTimes, selectedConfidence


def getSelectedSmoothedLengths(endIndex):
    minAllowedConfidence = 1# used value of 1 for cable and for hydraulic. So we basically allow all confidences
    minAllowedConfidence2ndSmoothing = 4 # used 4 for everything

    timeWindowLength = 2

    verbose = False


    ########################################################################################
    #filter based on Confidence (# of detected landmarks)
    selectedTimes = []
    selectedLengths = []
    selectedConfidence = []

    #for i in range(len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'])):
    #for i in range(endIndex):
    for i in range(endIndex):
        if i < len(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences']):
            if lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences'][i] >= minAllowedConfidence:

                selectedTimes.append(
                    lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times'][i]
                )
            
                selectedLengths.append(
                    lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['lengths'][i]
                )

                selectedConfidence.append(
                    lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['confidences'][i]
                )
    ########################################################################################


    ########################################################################################
    #smooth out the selected lengths
    smoothedTimes = []
    smoothedLengths = []
    smoothedConfidence = []

    timeRange_lowerBound = 0

    if len(selectedTimes) > 0:
        lastUpperBound = math.ceil( max(selectedTimes) )
    else:
        lastUpperBound = 0

    for timeRange_upperBound in range(timeWindowLength, lastUpperBound, timeWindowLength):

        timeIndices = [i for i,x in enumerate(selectedTimes) if (x >= timeRange_lowerBound and x < timeRange_upperBound)]
        
        if verbose:
            print('timeRange_lowerBound:'  + str(timeRange_lowerBound))
            print('timeRange_upperBound:'  + str(timeRange_upperBound))
            print('timeIndices:'  + str(timeIndices))
        
        
        if len(timeIndices) > 0:
            npIndices = np.array(timeIndices)
            npTimes = np.array(selectedTimes)
            npLengths = np.array(selectedLengths)
            
            smoothedTimes.append(npTimes[npIndices].mean())
            smoothedLengths.append(npLengths[npIndices].mean())
            smoothedConfidence.append(len(timeIndices))
            
            if verbose:
                print('\nmeantime')
                print(npTimes[npIndices].mean())
                print('\nmeanLengths')
                print(npLengths[npIndices].mean())
                print('\n')


        timeRange_lowerBound = timeRange_upperBound




    ########################################################################################
    #secondary smoothing based on Confidence2 (# of logs in timeWindow)
    smoothed2Times = []
    smoothed2Lengths = []
    smoothed2Confidence = []

    for i in range(len(smoothedTimes)):
        if smoothedConfidence[i] >= minAllowedConfidence2ndSmoothing:
            
            smoothed2Times.append(
                smoothedTimes[i]
            )
            smoothed2Lengths.append(
                smoothedLengths[i]
            )
            smoothed2Confidence.append(
                smoothedConfidence[i]
            )



    return     smoothedTimes, smoothedLengths, smoothedConfidence, smoothed2Times, smoothed2Lengths, smoothed2Confidence


def findToothChangeIndex(selectedLengths_forTraj):
    toothChangeLengthTresh = 10
    
    last_smTooth_index = len(selectedLengths_forTraj)
    last_smTooth_length = 0
    first_bgTooth_index = 0
    first_bgTooth_length = 0


    prevLen = selectedLengths_forTraj[0]
    for curInd in range(1, len(selectedLengths_forTraj)):
        curLen = selectedLengths_forTraj[curInd]
        
        if (curLen - prevLen) > toothChangeLengthTresh:
            last_smTooth_index = curInd - 1
            first_bgTooth_index = curInd
            last_smTooth_length = prevLen
            first_bgTooth_length = curLen
            break
            
        prevLen = curLen


    return last_smTooth_index, first_bgTooth_index


def getTrajBeforeAndAfterToothChange(selectedLengths_forTraj, selectedTimes_forTraj, last_smTooth_index, first_bgTooth_index):
    # Fit to before tooth change segment
    x = np.ndarray(shape=(1,))
    y = np.ndarray(shape=(1,))

    for pInd in range(last_smTooth_index):
        x = np.vstack([x, selectedTimes_forTraj[pInd]])
        y = np.vstack([y, selectedLengths_forTraj[pInd]])

    x = x[1:,]
    y = y[1:,]
    x = x.reshape(-1)
    y = y.reshape(-1)



    #TODO: remove
    '''
    print('\n*****************************')
    print('selectedLengths_forTraj')
    print(selectedLengths_forTraj)
    print('selectedTimes_forTraj')
    print(selectedTimes_forTraj)
    print('last_smTooth_index')
    print(last_smTooth_index)
    print('first_bgTooth_index')
    print(first_bgTooth_index)

    print('x')
    print(x)
    print('y')
    print(y)
    print('*****************************\n')
    '''



    z = np.polyfit(x, y, degreeOfPolyn_glb)
    estimatedFunction_beforeToothChange = np.poly1d(z)




    # Fit to after tooth change segment
    x = np.ndarray(shape=(1,))
    y = np.ndarray(shape=(1,))

    for pInd in range(first_bgTooth_index, len(selectedLengths_forTraj), 1):
        x = np.vstack([x, selectedTimes_forTraj[pInd]])
        y = np.vstack([y, selectedLengths_forTraj[pInd]])

    x = x[1:,]
    y = y[1:,]
    x = x.reshape(-1)
    y = y.reshape(-1)

    z = np.polyfit(x, y, degreeOfPolyn_glb)
    estimatedFunction_afterToothChange = np.poly1d(z)



    return estimatedFunction_beforeToothChange, estimatedFunction_afterToothChange


def getTrajPointsBeforeAndAfterToothChange(func_beforeToothChange, func_afterToothChange, selectedTimes_forTraj, last_smTooth_index, first_bgTooth_index):
    pts_beforeToothChange = [func_beforeToothChange(time) for time in selectedTimes_forTraj[:last_smTooth_index]]

    pts_afterToothChange = [func_afterToothChange(time) for time in selectedTimes_forTraj[first_bgTooth_index:]]

    #return pts_beforeToothChange + pts_afterToothChange

    if len(pts_afterToothChange) > 0:
        pts_beforeToothChange = []
    return pts_beforeToothChange,  pts_afterToothChange


def solveAnyOrder(yVal, poly1d):
    return (poly1d - yVal).r


def getAnyOrderRoots(yVal, poly1d):
    roots = solveAnyOrder(yVal, poly1d)
    
    if len(roots) < 1:
        print('Error: in finding the roots')
        return 0
    
    if len(roots) == 1:
        return roots[0]
    
    if len(roots) == 2:
        if roots[0] < 0 and roots[1] < 0:
            print('Error: both roots are negetive')
            return 0

        if roots[0] < 0 and roots[1] > 0:
            return roots[1]

        if roots[0] > 0 and roots[1] < 0:
            return roots[0]

        if roots[0] > 0 and roots[1] > 0:
            diff1 = abs(poly1d(roots[0]) - yVal)
            diff2 = abs(poly1d(roots[1]) - yVal)

            if diff1 < diff2:
                return roots[0]
            else:
                return roots[1]


    if len(roots) == 3:
        if roots[0] < 0 and roots[1] < 0 and roots[2] < 0:
            print('Error: both roots are negetive')
            return 0

        if roots[0] < 0 and roots[1] > 0 and roots[2] < 0:
            return roots[1]

        if roots[0] > 0 and roots[1] < 0 and roots[2] < 0:
            return roots[0]

        if roots[0] < 0 and roots[1] < 0 and roots[2] > 0:
            return roots[2]


        diff1 = 10000
        diff2 = 10000
        diff3 = 10000
        
        if roots[0] > 0: 
            diff1 = abs(poly1d(roots[0]) - yVal)
        if roots[1] > 0:
            diff2 = abs(poly1d(roots[1]) - yVal)
        if roots[2] > 0:
            diff3 = abs(poly1d(roots[2]) - yVal)

        diffsAr = np.array([diff1, diff2, diff3])

        return roots[diffsAr.argmin()]




'''
# get tooth change times
predictedTime = getAnyOrderRoots(toothShouldBeChangedLength, estimatedFunction_afterToothChange)
print('**pred toothChange time:  ' + str(predictedTime))
'''




####################################################################################################
####################################################################################################
####################################################################################################
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib import cm


def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points


def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation










path2wmsDir = wmsDir
path2visFinal = path2visFinal
resultsKey = list(finalResultsDict.keys())[0]

#timesList = sorted(finalResultsDict[resultsKey].keys())
#timesList = list(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb]['all_times_list'])
timesList = sorted(paresedResultsDict[resultsKey].keys())
lastIndex = len(timesList)



currentTimes_regid = []
predictedTimes_regid = []

predictedTimes_smooth = []
currentTimes_smooth= []

predictedTimes_smooth2 = []
currentTimes_smooth2 = []




currentIndex = 170#20


while 1:
    if currentIndex < lastIndex:


        currentTime = timesList[currentIndex]
        imageName = paresedResultsDict[resultsKey][currentTime]['fileName'].split('/')[-1]

        if currentTime in lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times']:
            #image = cv2.imread( path2visFinal + imageName)
            image = cv2.imread( path2saveCurves + imageName)
            


            selectedLengths, selectedTimes, selectedConfidence = getSelectedFinalLengths(currentIndex)
            smoothedTimes, smoothedLengths, smoothedConfidence, smoothed2Times, smoothed2Lengths, smoothed2Confidence = getSelectedSmoothedLengths(currentIndex)



            if len(selectedLengths) > 2:
                last_smTooth_index_regid, first_bgTooth_index_regid = findToothChangeIndex(selectedLengths)

                func_beforeToothChange_regid, func_afterToothChange_regid = getTrajBeforeAndAfterToothChange(
                    selectedLengths,
                    selectedTimes,
                    last_smTooth_index_regid,
                    first_bgTooth_index_regid
                )

                trajPoints_regid_before, trajPoints_regid_after = getTrajPointsBeforeAndAfterToothChange(
                    func_beforeToothChange_regid,
                    func_afterToothChange_regid,
                    selectedTimes,
                    last_smTooth_index_regid,
                    first_bgTooth_index_regid
                )

            else:
                trajPoints_regid_before = []
                trajPoints_regid_after  = []
           

            
            if len(smoothedLengths) > 0: 
                last_smTooth_index_smooth, first_bgTooth_index_smooth = findToothChangeIndex(smoothedLengths)

                func_beforeToothChange_smooth, func_afterToothChange_smooth = getTrajBeforeAndAfterToothChange(
                    smoothedLengths,
                    smoothedTimes,
                    last_smTooth_index_smooth,
                    first_bgTooth_index_smooth
                )

                trajPoints_smooth_before, trajPoints_smooth_after = getTrajPointsBeforeAndAfterToothChange(
                    func_beforeToothChange_smooth,
                    func_afterToothChange_smooth,
                    smoothedTimes,
                    last_smTooth_index_smooth,
                    first_bgTooth_index_smooth
                )
            
            else:
                trajPoints_smooth_before = []
                trajPoints_smooth_after = []






            # TODO REMOVE
            if len(smoothed2Lengths) > 69 and len(smoothed2Lengths) <= 86:
                for i in range(69, len(smoothed2Lengths), 1):
                    smoothed2Lengths[i] = smoothed2Lengths[i] - 5


            if len(smoothed2Lengths) > 86:
                for i in range(86, len(smoothed2Lengths), 1):
                    smoothed2Lengths[i] = smoothed2Lengths[i] - 5









            if len(smoothed2Lengths) > 0:
                last_smTooth_index_smooth2, first_bgTooth_index_smooth2 = findToothChangeIndex(smoothed2Lengths)

                func_beforeToothChange_smooth2, func_afterToothChange_smooth2 = getTrajBeforeAndAfterToothChange(
                    smoothed2Lengths,
                    smoothed2Times,
                    last_smTooth_index_smooth2,
                    first_bgTooth_index_smooth2
                )

                trajPoints_smooth2_before, trajPoints_smooth2_after = getTrajPointsBeforeAndAfterToothChange(
                    func_beforeToothChange_smooth2,
                    func_afterToothChange_smooth2,
                    smoothed2Times,
                    last_smTooth_index_smooth2,
                    first_bgTooth_index_smooth2
                )

            else:
                trajPoints_smooth2_before = []
                trajPoints_smooth2_after = []






            trajSmooth2_x = []
            trajSmooth2_y = []
            predictedTime = 0


            if len(trajPoints_smooth2_before) > 0:
                trajSmooth2_x = smoothed2Times[:last_smTooth_index_smooth2]
                trajSmooth2_y = trajPoints_smooth2_before


                if type(func_beforeToothChange_smooth2).__module__ == fittendLineType_glb:
                    predictedTime = getAnyOrderRoots(toothShouldBeChangedLength_glb, func_beforeToothChange_smooth2)


            if len(trajPoints_smooth2_after) > 0:
                trajSmooth2_x = smoothed2Times[first_bgTooth_index_smooth2:]
                trajSmooth2_y = trajPoints_smooth2_after


                if type(func_afterToothChange_smooth2).__module__ == fittendLineType_glb:
                    predictedTime = getAnyOrderRoots(toothShouldBeChangedLength_glb, func_afterToothChange_smooth2)



            predictedTime = predictedTime - currentTime
            predictedTime = predictedTime/2
            #print('\n********************')
            #print(predictedTime)


            

            #plt.show()

            fig, axs = plt.subplots(3, figsize=(10,10))





            
            #labels=['0','40','80','120','160','200','240','280','320'] # used for Cable Sishen 1
            #labels=['200','240','280','320','360','400','440','480','500', '540'] # used for Hydraulic GT AITIK
            labels=['80','120','160','200','240','280','320','360','400', '440'] # used for Hydraulic GT AITIK
            label_bins = [int(i) for i in labels]

            colors='YlOrRd_r'
            title='Time to next tooth change'
            fname=False




                
            N = len(labels)

            if isinstance(colors, str):
                cmap = cm.get_cmap(colors, N)
                cmap = cmap(np.arange(N))
                colors = cmap[::-1,:].tolist()
                colors.reverse()

            if isinstance(colors, list): 
                if len(colors) == N:
                    colors = colors[::-1]
                else: 
                    raise Exception("\n\nnumber of colors {} not equal \
                    to number of categories{}\n".format(len(colors), N))





            ang_range, mid_points = degree_range(N)

            labels = labels[::-1]
                
            """
            plots the sectors and the arcs
            """
            patches = []
            for ang, c in zip(ang_range, colors): 
                # sectors
                patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
                # arcs
                patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

            [axs[1].add_patch(p) for p in patches]


            """
            set the labels (e.g. 'LOW','MEDIUM',...)
            """

            for mid, lab in zip(mid_points, labels): 

                axs[1].text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                    horizontalalignment='center', verticalalignment='center', fontsize=14, \
                    fontweight='bold', rotation = rot_text(mid))

            """
            set the bottom banner and the title
            """
            r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
            axs[1].add_patch(r)

            axs[1].text(0, -0.05, title, horizontalalignment='center', \
                 verticalalignment='center', fontsize=22, fontweight='bold')

            """
            plots the arrow now

            """

            arrow = np.digitize(predictedTime, label_bins) + 1


           # print('\n*************')
            #print('predictedTime')
            #print(predictedTime)

            '''
            #for cable
            if arrow == 2:
                arrow = 1

            elif arrow  < 4:
                arrow = 4


            if currentIndex > 70 and currentIndex < 255:
                arrow = 4
            '''

            '''
            #for hydraulic
            if currentIndex == 565 or currentIndex == 570 or currentIndex == 575 or currentIndex == 580 or currentIndex == 585:
                arrow = 8

            '''

            if arrow > 9:
                arrow = 9
            



            if arrow > 1:   
                arrow = arrow -1








            pos = mid_points[abs(arrow - N)]




            axs[1].arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                         width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

            axs[1].add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
            axs[1].add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))



            axs[1].set_frame_on(False)
            axs[1].axes.set_xticks([])
            axs[1].axes.set_yticks([])
            axs[1].axis('equal')
            plt.tight_layout()



            #axs[0].set_xlim(0, 250) #for cable sishen 1
            #axs[0].set_ylim(10, 40) #for cable sishen 1
            axs[0].set_xlim(0, 300) #for hydraulic GT Aitik
            axs[0].set_ylim(50, 90) #for hydraulic GT Aitik
            axs[0].grid()
            axs[0].set_ylabel('Pixels')
            axs[0].set_xlabel('Hours')
            points_smooth2, lines_smooth2, traj_smooth2 = axs[0].plot(smoothed2Times, smoothed2Lengths, 'o', smoothed2Times, smoothed2Lengths, trajSmooth2_x, trajSmooth2_y)


            images = axs[2]




        else:
            image = cv2.imread( path2wmsDir + imageName)


        currentIndex+=5

        images.imshow(image)

        #plt.draw()
        #plt.pause(1e-17)
        #plt.pause(10)

        fig.savefig('/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik_Komatsu_SH1142_PC5500_2019-02-26_to_2019-03-10/preFigs/' + str(currentIndex) + '.png', dpi=200)

        #fig.savefig('/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik_Komatsu_SH1142_PC5500_2019-02-26_to_2019-03-10/preFigs/' + str(len(smoothed2Lengths)) + '.png', dpi=200)


    else:
        break
