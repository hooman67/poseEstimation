import sys

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








def getSelectedForPlotting(endIndex):
    minAllowedConfidence = 0.7  #1 to 5 for conf1 --- 0-1 for conf 2 and 3
    filterBasedOnConf = 2    # 1 for conf1 2 for conf2, 3 for logConf


    path2savePlots = mainPath + 'finalPlots/'
    resKey = list(lengthsAndLandmarksDict.keys())[0]
    toothNbKey = 'tooth_' + str(0 + 1) + '_info'  
    regTypeKeyWord = 'rigid_keypointsForTooth_'
    refKey = 'WMDL_2018.09.11_06.21.57'
    infoKey = 'all_tt2ls_dict'

        

    #filter based on Confidence
    selectedLengths = []
    selectedTimes = []
    selectedConfidence = []

    #for i in range(len(lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['times'])):
    for i in range(0, endIndex, 1):

        if filterBasedOnConf == 1:
            if lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['confidences'][i] >= minAllowedConfidence:

                selectedLengths.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['lengths'][i]
                )
                selectedTimes.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['times'][i]
                )
                selectedConfidence.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['confidences'][i]
                )
                
        elif filterBasedOnConf == 2:
            if lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['secondaryConfidences'][i] >= minAllowedConfidence:

                selectedLengths.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['lengths'][i]
                )
                selectedTimes.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['times'][i]
                )
                selectedConfidence.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['secondaryConfidences'][i]
                )
                
                
        elif filterBasedOnConf == 3:
            if lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['logConfidence'][i] >= minAllowedConfidence:

                selectedLengths.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['lengths'][i]
                )
                selectedTimes.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['times'][i]
                )
                selectedConfidence.append(
                    lengthsAndLandmarksDict[resKey][regTypeKeyWord][refKey][toothNbKey][infoKey]['logConfidence'][i]
                )



    return selectedLengths, selectedTimes, selectedConfidence











class hsCanvesDrawer:
    def __init__(self, points, lines, path2wmsDir, path2visFinal):
        self.lines = lines
        self.points = points

        self.cid = lines.figure.canvas.mpl_connect('button_press_event', self)

        self.path2wmsDir = path2wmsDir
        self.path2visFinal = path2visFinal
        self.resultsKey = list(finalResultsDict.keys())[0]

        self.timesList = sorted(finalResultsDict[self.resultsKey].keys())
        self.currentIndex = 0
        self.lastIndex = len(self.timesList)



    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.lines.axes: return


        if self.currentIndex < self.lastIndex:
            imageName = finalResultsDict[self.resultsKey][self.timesList[self.currentIndex]]['fileName'].split('/')[-1]
            self.currentIndex+=1

            if os.path.exists(self.path2visFinal + imageName):
                image = cv2.imread( self.path2visFinal + imageName)

                selectedLengths, selectedTimes, selectedConfidence = getSelectedForPlotting(self.currentIndex)
                
                self.lines.set_data(selectedTimes, selectedLengths)
                self.points.set_data(selectedTimes, selectedLengths)


            else:
                image = cv2.imread( self.path2wmsDir + imageName)


            #plt.imshow(image)
            self.lines.figure.canvas.draw()
            self.points.figure.canvas.draw()



        else:
            plt.close()




fig = plt.figure(figsize=(30,10))
plt.axis([0, 250, 10, 50]) # for cable GT

ax = plt.axes()
ax.grid()

plt.xlabel('Hours')
plt.xlabel('Pixels')

#ax = fig.add_subplot(111)
#ax.set_title('click to build line segments')
points, lines = plt.plot([], [], 'o', [], [])


linebuilder = hsCanvesDrawer(points, lines, wmsDir, path2visFinal)

plt.show()










'''
            
    plotTitle = str(resKey + '__' + str(toothNbKey) + '__' + regTypeKeyWord + '__' + infoKey + '__'+ refKey)
    plt.title(plotTitle)
    plt.plot(selectedTimes, selectedLengths, label='nbOfPoints: ' + str(len(selectedLengths)))
    plt.plot(selectedTimes, selectedLengths,'o')
    for i in range(len(selectedTimes)):
        ax.annotate( round(selectedConfidence[i], 1), (selectedTimes[i], selectedLengths[i]) )
        
        
    ax.legend()
    plt.savefig(path2savePlots + plotTitle + '.png')
class hsCanvesDrawer:
    def __init__(self, line, path2wmsDir, path2visFinal):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.path2wmsDir = path2wmsDir
        self.path2visFinal = path2visFinal
        self.resultsKey = list(finalResultsDict.keys())[0]
        self.timesList = sorted(finalResultsDict[self.resultsKey].keys())
        self.currentIndex = 0
        self.lastIndex = len(self.timesList)
    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        #self.line.set_data(self.xs, self.ys)
        if self.currentIndex < self.lastIndex:
            imageName = finalResultsDict[self.resultsKey][self.timesList[self.currentIndex]]['fileName'].split('/')[-1]
            self.currentIndex+=1
            if os.path.exists(self.path2visFinal + imageName):
                image = cv2.imread( self.path2visFinal + imageName)
            else:
                image = cv2.imread( self.path2wmsDir + imageName)
            plt.imshow(image)
            self.line.figure.canvas.draw()
        else:
            plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('click to build line segments')
line, = ax.plot([0], [0])  # empty line
linebuilder = hsCanvesDrawer(line, wmsDir, path2visFinal)
plt.show()
'''