import sys
import time as tt

# This is needed if the notebook is stored in the object_detection folder.
sys.path.append(".")
sys.path.append("/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800")


from helpers import *
from configs import *
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
refKey_glb = 'WMDL_2018.09.11_06.21.57'
infoKey_glb = 'all_tt2ls_dict'

fittendLineType_glb = 'numpy.lib.polynomial'

toothShouldBeChangedLength_glb= 21



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
    minAllowedConfidence2ndSmoothing = 4 # used value of 7 for cable. 4 for hydraulic. This is the number of points withing the time window

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
    degreeOfPolyn = 1


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



    z = np.polyfit(x, y, degreeOfPolyn)
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

    z = np.polyfit(x, y, degreeOfPolyn)
    estimatedFunction_afterToothChange = np.poly1d(z)



    return estimatedFunction_beforeToothChange, estimatedFunction_afterToothChange


def getTrajPointsBeforeAndAfterToothChange(func_beforeToothChange, func_afterToothChange, selectedTimes_forTraj, last_smTooth_index, first_bgTooth_index):
    pts_beforeToothChange = [func_beforeToothChange(time) for time in selectedTimes_forTraj[:last_smTooth_index]]

    pts_afterToothChange = [func_afterToothChange(time) for time in selectedTimes_forTraj[first_bgTooth_index:]]

    #return pts_beforeToothChange + pts_afterToothChange

    if len(pts_afterToothChange) > 0:
        pts_beforeToothChange = []
    return pts_beforeToothChange,  pts_afterToothChange


def solve1stOrder(yVal, poly1d):
    return (yVal - poly1d[0]) / poly1d[1]







'''
# get tooth change times
predictedTime = solve1stOrder(toothShouldBeChangedLength, estimatedFunction_afterToothChange)
print('**pred toothChange time:  ' + str(predictedTime))
'''




####################################################################################################
####################################################################################################
####################################################################################################
plt.show()
fig, axs = plt.subplots(5, figsize=(30,10))


axs[0].set_xlim(0, 250)
axs[0].set_ylim(10, 40)
axs[0].grid()
axs[0].set_ylabel('Pixels')
points, lines, traj_regid = axs[0].plot([], [], 'o', [], [], [], [])


axs[1].set_xlim(0, 250)
axs[1].set_ylim(10, 40)
axs[1].grid()
axs[0].set_ylabel('Pixels')
points_smooth, lines_smooth, traj_smooth = axs[1].plot([], [], 'o', [], [], [], [])


axs[2].set_xlim(0, 250)
axs[2].set_ylim(10, 40)
axs[2].grid()
axs[0].set_ylabel('Pixels')
axs[2].set_xlabel('Hours')
points_smooth2, lines_smooth2, traj_smooth2 = axs[2].plot([], [], 'o', [], [], [], [])


axs[3].set_xlim(0, 250)
axs[3].set_ylim(0, 400)
axs[3].grid()
axs[3].set_ylabel('Hours')
timePreds_regid, timePreds_smooth, timePreds_smooth2 = axs[2].plot([], [], 'o', [], [], 'x', [], [], 'bx')


images = axs[4]





path2wmsDir = wmsDir
path2visFinal = path2visFinal
resultsKey = list(finalResultsDict.keys())[0]

#timesList = sorted(finalResultsDict[resultsKey].keys())
#timesList = list(lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb]['all_times_list'])
timesList = sorted(paresedResultsDict[resultsKey].keys())
currentIndex = 0
lastIndex = len(timesList)

while 1:
    if currentIndex < lastIndex:

        currentTime = timesList[currentIndex]
        imageName = paresedResultsDict[resultsKey][currentTime]['fileName'].split('/')[-1]

        if currentTime in lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times']:
            image = cv2.imread( path2visFinal + imageName)


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
            




            # update the plots
            lines.set_xdata(selectedTimes)
            lines.set_ydata(selectedLengths)

            points.set_xdata(selectedTimes)
            points.set_ydata(selectedLengths)

            if len(trajPoints_regid_before) > 0:
                #traj_regid.set_xdata(range(len(selectedTimes)))
                traj_regid.set_xdata(selectedTimes[:last_smTooth_index_regid])
                traj_regid.set_ydata(trajPoints_regid_before)


                if type(func_beforeToothChange_regid).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_beforeToothChange_regid)

                    print('\n****BEFORE**********')
                    print('predictedTime')
                    print(predictedTime)
                    print('*****************\n')

                    timePreds_regid.set_xdata(last_smTooth_index_regid - 1)
                    timePreds_regid.set_ydata(predictedTime)
            

            if len(trajPoints_regid_after) > 0:
                traj_regid.set_xdata(selectedTimes[first_bgTooth_index_regid:])
                traj_regid.set_ydata(trajPoints_regid_after)

                if type(func_afterToothChange_regid).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_afterToothChange_regid)

                    print('\n****After**********')
                    print('predictedTime')
                    print(predictedTime)
                    print('*****************\n')

                    timePreds_regid.set_xdata(first_bgTooth_index_regid)
                    timePreds_regid.set_ydata(predictedTime)




            lines_smooth.set_xdata(smoothedTimes)
            lines_smooth.set_ydata(smoothedLengths)

            points_smooth.set_xdata(smoothedTimes)
            points_smooth.set_ydata(smoothedLengths)

            if len(trajPoints_smooth_before) > 0:
                traj_smooth.set_xdata(smoothedTimes[:last_smTooth_index_smooth])
                traj_smooth.set_ydata(trajPoints_smooth_before)


                if type(func_beforeToothChange_smooth).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_beforeToothChange_smooth)

                    timePreds_smooth.set_xdata(last_smTooth_index_smooth - 1)
                    timePreds_smooth.set_ydata(predictedTime)


            if len(trajPoints_smooth_after) > 0:
                traj_smooth.set_xdata(smoothedTimes[first_bgTooth_index_smooth:])
                traj_smooth.set_ydata(trajPoints_smooth_after)


                if type(func_afterToothChange_smooth).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_afterToothChange_smooth)

                    timePreds_smooth.set_xdata(first_bgTooth_index_smooth)
                    timePreds_smooth.set_ydata(predictedTime)



            lines_smooth2.set_xdata(smoothed2Times)
            lines_smooth2.set_ydata(smoothed2Lengths)

            points_smooth2.set_xdata(smoothed2Times)
            points_smooth2.set_ydata(smoothed2Lengths)

            if len(trajPoints_smooth2_before) > 0:
                traj_smooth2.set_xdata(smoothed2Times[:last_smTooth_index_smooth2])
                traj_smooth2.set_ydata(trajPoints_smooth2_before)


                if type(func_beforeToothChange_smooth2).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_beforeToothChange_smooth2)

                    timePreds_smooth2.set_xdata(last_smTooth_index_smooth2 - 1)
                    timePreds_smooth2.set_ydata(predictedTime)
                

            if len(trajPoints_smooth2_after) > 0:
                traj_smooth2.set_xdata(smoothed2Times[first_bgTooth_index_smooth2:])
                traj_smooth2.set_ydata(trajPoints_smooth2_after)


                if type(func_afterToothChange_smooth2).__module__ == fittendLineType_glb:
                    predictedTime = solve1stOrder(toothShouldBeChangedLength_glb, func_afterToothChange_smooth2)

                    timePreds_smooth2.set_xdata(first_bgTooth_index_smooth2)
                    timePreds_smooth2.set_ydata(predictedTime)


        else:
            image = cv2.imread( path2wmsDir + imageName)


        currentIndex+=1

        images.imshow(image)

        plt.draw()
        plt.pause(1e-17)


    else:
        break

plt.show()











'''
class hsCanvesDrawer:
    def __init__(self, points, lines, images, points_smooth, lines_smooth, points_smooth2, lines_smooth2, path2wmsDir, path2visFinal):
        self.lines = lines
        self.points = points
        self.images = images
        self.points_smooth = points_smooth
        self.lines_smooth = lines_smooth
        self.points_smooth2 = points_smooth2
        self.lines_smooth2 = lines_smooth2

        self.cid = lines.figure.canvas.mpl_connect('button_press_event', self)

        self.path2wmsDir = path2wmsDir
        self.path2visFinal = path2visFinal
        self.resultsKey = list(finalResultsDict.keys())[0]

        self.timesList = sorted(finalResultsDict[self.resultsKey].keys())
        self.currentIndex = 0
        self.lastIndex = len(self.timesList)



    def __call__(self, event):
        #if event.inaxes!=self.lines.axes: return
        #print('click', event)
    #def update(self):

        if self.currentIndex < self.lastIndex:

            currentTime = self.timesList[self.currentIndex]
            imageName = finalResultsDict[self.resultsKey][currentTime]['fileName'].split('/')[-1]

            if currentTime in lengthsAndLandmarksDict[resKey_glb][regTypeKeyWord_glb][refKey_glb][toothNbKey_glb][infoKey_glb]['times']:
                image = cv2.imread( self.path2visFinal + imageName)


                selectedLengths, selectedTimes, selectedConfidence = getSelectedFinalLengths(self.currentIndex)
                smoothedTimes, smoothedLengths, smoothedConfidence, smoothed2Times, smoothed2Lengths, smoothed2Confidence = getSelectedSmoothedLengths(self.currentIndex)
                

                self.lines.set_data(selectedTimes, selectedLengths)
                self.points.set_data(selectedTimes, selectedLengths)

                self.lines_smooth.set_data(smoothedTimes, smoothedLengths)
                self.points_smooth.set_data(smoothedTimes, smoothedLengths)

                self.lines_smooth2.set_data(smoothed2Times, smoothed2Lengths)
                self.points_smooth2.set_data(smoothed2Times, smoothed2Lengths)


            else:
                image = cv2.imread( self.path2wmsDir + imageName)
            

            self.currentIndex+=1
            self.lines.figure.canvas.draw()
            self.points.figure.canvas.draw()
            self.lines_smooth.figure.canvas.draw()
            self.points_smooth.figure.canvas.draw()
            self.lines_smooth2.figure.canvas.draw()
            self.points_smooth2.figure.canvas.draw()

            #self.figure.canvas.flush_events()
            self.images.imshow(image)



        else:
            plt.close()




fig = plt.figure(figsize=(30,10))
plt.axis([0, 250, 10, 50]) # for cable GT

ax = plt.axes()
ax.grid()

plt.xlabel('Hours')
plt.xlabel('Pixels')

ax1 = fig.add_subplot(111)
#ax.set_title('click to build line segments')
points, lines = ax1.plot([], [], 'o', [], [])


ax2 = fig.add_subplot(211)




ax3 = fig.add_subplot(311)
points_smooth, lines_smooth = ax3.plot([], [], 'o', [], [])

ax4 = fig.add_subplot(411)
points_smooth2, lines_smooth2 = ax4.plot([], [], 'o', [], [])

linebuilder = hsCanvesDrawer(points, lines, ax2, points_smooth, lines_smooth, points_smooth2, lines_smooth2, wmsDir, path2visFinal)

plt.show()
'''
