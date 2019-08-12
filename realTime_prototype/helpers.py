#Imports
import os, shutil, sys, tarfile, zipfile, cv2, json, xmltodict, math
import numpy as np
import openpyxl as oxl
import six.moves.urllib as urllib
from copy import deepcopy
from datetime import datetime
from collections import deque
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image
import xml.etree.ElementTree as ET

#for finding the shortest distance btw curve and point
from scipy.optimize import fmin_cobyla

sys.path.append(".")
from configs import *


def draw_patch_keypoints(image, patch_keypoints, validKeypoints = []):
    colors = [[255, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [153, 255, 255]]
    circle_radius=5

    for i, joint_keypoint in enumerate(patch_keypoints):
        x, y = joint_keypoint
        
        if x == 0 or y == 0:
            continue
            
        if len(validKeypoints) > 1 and joint_keypoint not in validKeypoints:
            continue

        cv2.circle(image, (x, y), circle_radius, colors[i], -1)

    return image


def draw_fittedCurves_for1setOfKeypoints(image, setof_keypoints, esimatorFunction, colorNb=0):
    #colors = [[204, 0, 204], [255, 255, 51], [255, 51, 255], [153, 51, 255], [255, 102, 255], [178, 102, 255]]
    colors = [[255, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [153, 255, 255]]
    
    #pts = [[p[0], int(esimatorFunction(p[0]))] for p in setof_keypoints if(p[0] > 0 and p[1]>0)]
    pts = [[indx, int(esimatorFunction(indx))] for indx in range(10, 600, 50)]
    pts_np = np.array(pts, np.int32)
    pts_np = pts_np.reshape((-1,1,2))
    cv2.polylines(image,[pts_np],False,colors[colorNb])

    return image


def draw_box(image, box):
    colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),(0, 255, 255)]
    labels=["Tooth", "Toothline", "BucketBB", "MatInside", "WearArea"]
    image_h, image_w, _ = image.shape
    
    if len(box) > 3:
        xmin = int(box[0] * image_w)
        xmax = int(box[1] * image_w)
        ymin = int(box[2] * image_h)
        ymax = int(box[3] * image_h)
        label, score = box[4:6]
        #print(str(xmin) + '  ' + str(xmax) + '  '  + str(ymin) + '  ' + str(ymax) + '  ' + label)
        
        #add tooth length (from box) as a field
        box.append(ymax-ymin)

        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0

        color = colors[labels.index(label)]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

        font_increase = 1.
        cv2.putText(
            image,
            str(score),
            (xmin, ymin - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            8e-4 * image_h * font_increase,
            color, 2)

    return image


def draw_all_boxes(image, boxesDict):
    for key in boxesDict.keys():
        if 'keypoints' not in key and key not in ['nbOfDetectedTeeth','fileName', 'fittedCurves', '2ndDerivFittedCurves']:
            image = draw_box(image, boxesDict[key])
            
    return image


def draw_all_keypoints_boxes_andCurves(inImage, resultsDict, numberOfTeeth, refKey='', regTypeKeyword='', drawOnlyValidated=False, doNotDrawBoxes=True, numberOflandmarksIncludingToothTip = 5):

    image = inImage.copy()
    
    if regTypeKeyword == '':
        esimatorFunctions = resultsDict['fittedCurves']
        
        for toothNb in range(numberOfTeeth):
            toothKeyPointKey = regTypeKeyword + 'keypointsForTooth_' + str(toothNb + 1)

            if toothKeyPointKey in resultsDict:

                if drawOnlyValidated:
                    image = draw_patch_keypoints(image, resultsDict[toothKeyPointKey], resultsDict[regTypeKeyword +'validatedKeypoints'])
                else:
                    image = draw_patch_keypoints(image, resultsDict[toothKeyPointKey], [])


        for keypointNb in range(numberOflandmarksIncludingToothTip):
            toothKeyPointKey = 'keypoints_' + str(keypointNb + 1)

            if toothKeyPointKey in esimatorFunctions:
                image = draw_fittedCurves_for1setOfKeypoints(image, resultsDict[toothKeyPointKey], esimatorFunctions[toothKeyPointKey], colorNb=keypointNb)


        if not doNotDrawBoxes:
            draw_all_boxes(image, resultsDict)

        return image


    elif (regTypeKeyword + 'fittedCurves') in resultsDict['registrations'][refKey].keys():
        esimatorFunctions = resultsDict['registrations'][refKey][regTypeKeyword + 'fittedCurves']
    
        for toothNb in range(numberOfTeeth):
            toothKeyPointKey = regTypeKeyword + 'keypointsForTooth_' + str(toothNb + 1)

            if toothKeyPointKey in resultsDict['registrations'][refKey]:

                if drawOnlyValidated:
                    image = draw_patch_keypoints(image, resultsDict['registrations'][refKey][toothKeyPointKey], resultsDict['validatedKeypoints'])
                else:
                    image = draw_patch_keypoints(image, resultsDict['registrations'][refKey][toothKeyPointKey], [])


        for keypointNb in range(numberOflandmarksIncludingToothTip):
            toothKeyPointKey = 'keypoints_' + str(keypointNb + 1)

            if toothKeyPointKey in esimatorFunctions:
                image = draw_fittedCurves_for1setOfKeypoints(image, resultsDict[toothKeyPointKey], esimatorFunctions[toothKeyPointKey], colorNb=keypointNb)


        if not doNotDrawBoxes:
            draw_all_boxes(image, resultsDict)

        return image


def draw_just_keypoints_fromGroundTruth(inImage, resultsDict, numberOfTeeth, refKey='', numberOflandmarksIncludingToothTip = 5):

    image = inImage.copy()
        
    for toothNb in range(numberOfTeeth):
        toothKeyPointKey ='keypointsForTooth_' + str(toothNb + 1)

        if toothKeyPointKey in resultsDict:
                image = draw_patch_keypoints(image, resultsDict[toothKeyPointKey], [])

    return image


def dimg(framesDir, fileName, teethBoxes, keypoints):
    inImage = cv2.imread(framesDir + fileName.replace('.json','.png'))
    outImage = draw_all_keypoints(inImage, data[0], circle_radius=3)
    outImage = draw_boxes(outImage, data[1]) 
    plt.imshow(outImage)
    plt.show()


def getFrameTime(resultsDict, resultsKey, fileName):
    for timeKey in resultsDict[resultsKey].keys():
        if fileName in resultsDict[resultsKey][timeKey]['fileName']:
            print(timeKey)
            print(resultsDict[resultsKey][timeKey]['fileName'])
            print('\n')
            return timeKey
    
    print('could not find the requested file')
    return None


def loadResults(resultsDir, rawFramesDir=None):
    #Load JSON prediciton results from disc into dict
    resultsDirectories = []


    
    #for resultsDir in resultsDirectories:
    resultsDir = wmsDir
    resultKey = resultsDir.split('/')[-3]

    
    


    resultsDic = {}

    datetimemask = "%Y.%m.%d %H.%M.%S"
    
    resultsDic[resultKey] = {}

    zeroTimeRef = None

    for fileName in sorted(os.listdir(resultsDir)):
        if fileName and '.json' in fileName:
   
            fileNameAr = fileName[:len(fileName)-5].split('_')
            time = fileNameAr[1] + ' ' + fileNameAr[2]
            dateTime = datetime.strptime(time, datetimemask)

            curHourSince = 0

            if zeroTimeRef == None:
                zeroTimeRef = dateTime
            else:
                timeDif = dateTime - zeroTimeRef
                totalSeconds = timeDif.seconds
                totalDays = timeDif.days
                curHourSince = totalSeconds/3600 + totalDays*24



            with open(resultsDir + fileName, 'r') as fjson:
                data = tuple(json.load(fjson))
                #print(fileName)

                resultsDic[resultKey][curHourSince] = {}      
                resultsDic[resultKey][curHourSince]['time'] = time
                resultsDic[resultKey][curHourSince]['keypoints'] = data[0]
                resultsDic[resultKey][curHourSince]['teeth'] = data[1]
                resultsDic[resultKey][curHourSince]['buckets'] = data[2]
                resultsDic[resultKey][curHourSince]['nbOfDetectedTeeth'] = len(data[1])
                resultsDic[resultKey][curHourSince]['fileName'] = resultsDir + fileName.replace('.json', '.png')
                tempImage = cv2.imread(resultsDic[resultKey][curHourSince]['fileName'])
                image_h, image_w, _ = tempImage.shape
                resultsDic[resultKey][curHourSince]['image_h'] = image_h
                resultsDic[resultKey][curHourSince]['image_w'] = image_w
                


            #print(resultsDic)
            #dimg(rawFramesDir, fileName, data)
            #break



    print("loaded the results for  " + str(len(resultsDic[resultKey])) + "   frames. For key: " + resultKey)
    return resultsDic


def parseResults(resultsDic, numberOfTeeth):
    numberOflandmarksIncludingToothTip = 5
    parsedResultsDic = {}
    
    for resKey in resultsDic.keys():
        parsedResultsDic[resKey] = {}
        
        for time in resultsDic[resKey].keys():
            parsedResultsDic[resKey][time] = {}
            
            parsedResultsDic[resKey][time]['nbOfDetectedTeeth'] = resultsDic[resKey][time]['nbOfDetectedTeeth'] 
            parsedResultsDic[resKey][time]['fileName'] = resultsDic[resKey][time]['fileName']
            parsedResultsDic[resKey][time]['image_h'] = resultsDic[resKey][time]['image_h']
            parsedResultsDic[resKey][time]['image_w'] = resultsDic[resKey][time]['image_w']
            
            #parse the bucket objects
            for obj in resultsDic[resKey][time]['buckets']:
                parsedResultsDic[resKey][time][obj[4]] = obj
                
            #parse the teeth 
            toothNb = 1
            
            #TODO: delete this
            #print(parsedResultsDic[resKey][time]['fileName'])
            
            for obj in sorted(resultsDic[resKey][time]['teeth'], key=lambda rv: rv[0]):
                parsedResultsDic[resKey][time]['Tooth_' + str(toothNb)] = obj
                toothNb += 1
                
            toothNb = 1
            for keypointsSet in sorted(resultsDic[resKey][time]['keypoints'], key=lambda kv: kv[0]):           
                parsedResultsDic[resKey][time]['keypointsForTooth_' + str(toothNb)] = keypointsSet
                toothNb += 1
                
                
            for keypointsNb in range(numberOflandmarksIncludingToothTip):       
                parsedResultsDic[resKey][time]['keypoints_' + str(keypointsNb + 1)] = [keypointsSet[keypointsNb] for keypointsSet in sorted(resultsDic[resKey][time]['keypoints'], key=lambda kv: kv[0])]
            
                
    return parsedResultsDic


def getMinDistanceBtwPointAndToothBox(toothBox, landmarkPointsList, image_h):
    outDist = 10000
    
    yminBox = int(toothBox[2] * image_h)
    ymaxBox = int(toothBox[3] * image_h)
    
    x0, y0 = landmarkPointsList[0]
    _, y1 = landmarkPointsList[0]
        
    if not (x0 == 0 or y0 == 0):
        outDist = min( min( abs(yminBox - y0), abs(ymaxBox - y0) ), min( abs(yminBox - y1), abs(ymaxBox - y1) ) )

    return outDist


def getDistanceBtwEdges(leftBox, rightBox):
    xmaxLeft = float(leftBox[1])
    xminRight = float(rightBox[0])

    return xminRight - xmaxLeft


def reject1_badBoxesAndLandmarks(
    parsedResultsDic,
    rejectedResultsDir,
    numberOfTeeth,
    minToothBoxDistanceAllowed,
    numberOflandmarksIncludingToothTip,
    lanmark2farFromBox_epsilon,
    landmarks2close2eachother_epsilon,
    verbose=False):

    filteredResults = deepcopy(parsedResultsDic)
    
    
    for resKey in parsedResultsDic.keys():
        deletedCount = 0
        deletedDuplicateKeypointCount = 0
        
        for time in parsedResultsDic[resKey].keys():
            filePath = parsedResultsDic[resKey][time]['fileName']
            fileName = filePath.split('/')[-1]
            
            
            #***********************************************************************************************#
            #reject1 get rid of adjacent tooth boxes that are too close to eachother
            for toothNb in range(numberOfTeeth - 1):
                toothKeyLeft = 'Tooth_' + str(toothNb + 1)
                toothKeyRight = 'Tooth_' + str(toothNb + 2)
                
                if toothKeyLeft in parsedResultsDic[resKey][time] and toothKeyRight in parsedResultsDic[resKey][time]:
                    distanceBtwAdjTeeth = getDistanceBtwEdges(
                        parsedResultsDic[resKey][time][toothKeyLeft],
                        parsedResultsDic[resKey][time][toothKeyRight]
                    )

                    if distanceBtwAdjTeeth <= minToothBoxDistanceAllowed:
                        del filteredResults[resKey][time]['Tooth_' + str(toothNb + 2)]
                        parsedResultsDic[resKey][time]['nbOfDetectedTeeth'] = 0 #this garantees reject
                        filteredResults[resKey][time]['nbOfDetectedTeeth'] = 0 #this garantees reject
                        
                        if verbose:
                            print('\nrejected:\n' + str(filePath) + '\nbecause adjacent teeth were too close.')
            #***********************************************************************************************#
                        
            
            #***********************************************************************************************#
            #reject2  get rid of toothtips and lipshrouds that are not inside their tooth box
            for toothNb in range(numberOfTeeth - 1):
                toothKey = 'Tooth_' + str(toothNb + 1)  
                keypointKey = 'keypointsForTooth_' + str(toothNb + 1)
                
                if toothKey in filteredResults[resKey][time]:
                    miDis = getMinDistanceBtwPointAndToothBox(
                        filteredResults[resKey][time][toothKey],
                        filteredResults[resKey][time][keypointKey],
                        filteredResults[resKey][time]['image_h'])

                    if( miDis > lanmark2farFromBox_epsilon ):
                        filteredResults[resKey][time][keypointKey][0] = [0, 0]
                        filteredResults[resKey][time][keypointKey][1] = [0, 0]
                        filteredResults[resKey][time]['keypoints_' + str(1)][toothNb] = [0, 0]
                        filteredResults[resKey][time]['keypoints_' + str(2)][toothNb] = [0, 0]
                        parsedResultsDic[resKey][time]['nbOfDetectedTeeth'] = 0 #this garantees reject
                        filteredResults[resKey][time]['nbOfDetectedTeeth'] = 0 #this garantees reject
                        if verbose:
                            print('\nrejected:\n' + str(filePath) + '\nbecause landmarks were too far from box.')
            #***********************************************************************************************#
                    
                            

            #Not a reject yet get rid of landmarks with Xcords that are too close to eachother on adjacent teeth
            for lanmarkNb in range(numberOflandmarksIncludingToothTip):
                for toothNb in range(numberOfTeeth - 1):
                    if 'keypointsForTooth_' + str(toothNb+1) in filteredResults[resKey][time] and 'keypointsForTooth_' + str(toothNb+2) in filteredResults[resKey][time]:
                        
                        val2del = filteredResults[resKey][time]['keypointsForTooth_' + str(toothNb+2)][lanmarkNb]
                        
                        diff = abs(filteredResults[resKey][time]['keypointsForTooth_' + str(toothNb+1)][lanmarkNb][0] - val2del[0])

                        if diff < landmarks2close2eachother_epsilon:
                            filteredResults[resKey][time]['keypointsForTooth_' + str(toothNb+2)][lanmarkNb] = [0, 0]
                            filteredResults[resKey][time]['keypoints_' + str(lanmarkNb+1)][toothNb+1] = [0, 0]
                                                                                                               
                            deletedDuplicateKeypointCount +=1
                            
                            
                            
            
            #***********************************************************************************************#
            #reject3 not enough teeth boxes detected. (frames with missing TTips are rejected later)
            if parsedResultsDic[resKey][time]['nbOfDetectedTeeth']  < numberOfTeeth:
                shutil.copy(filePath, rejectedResultsDir)
                
                if verbose:
                    print('\nrejected:\n' + str(filePath) + '\nbecause not enough teeth were detected.')
                
                del filteredResults[resKey][time]
                deletedCount += 1
            #***********************************************************************************************#

                
            '''
            #reject4 no wearArea detected
            elif 'WearArea' not in parsedResultsDic[resKey][time].keys():
                shutil.copy(filePath, rejectedResultsDir)
                print('\nrejected:\n' + str(filePath) + '\nbecause wearArea was not detected.')
                del filteredResults[resKey][time]
                deletedCount += 1
            '''
            
    
        print('\nfor results set: ' + resKey + '  rejected ' + str(deletedCount) + ' logs from the parsedResultsDic which were not copied into filteredResults. And removed ' +str(deletedDuplicateKeypointCount) + '  landmarks that had Xcords too close to each other.' )
        
        return filteredResults


def fitCurve2keypoints(keypoints, numberOfTeeth, keypointTypeString):
    numberOflandmarksIncludingToothTip = 5
    degreeOfPolyn = 2
    minNumberOflandmarksNeededToFitCurve = 3
    
    estimatedFunctions = {}
    
    for landmarkNb in range(numberOflandmarksIncludingToothTip):
        landmarkKey = keypointTypeString + str(landmarkNb + 1)
        key2Stor = 'keypoints_' + str(landmarkNb + 1)
        
        x = np.ndarray(shape=(1,))
        y = np.ndarray(shape=(1,))
        
        for point in keypoints[landmarkKey]:
            if(point[0] > 0 and point[1] > 0):
                x = np.vstack([x, point[0]])
                y = np.vstack([y, point[1]])
            
        x = x[1:,]
        y = y[1:,]
        x = x.reshape(-1)
        y = y.reshape(-1)
        
        if len(x) >= minNumberOflandmarksNeededToFitCurve and len(y) >= minNumberOflandmarksNeededToFitCurve:
            z = np.polyfit(x, y, degreeOfPolyn)

            estimatedFunctions[key2Stor] = np.poly1d(z)
        
    
    return estimatedFunctions


def get2ndDerivativeOfCurves(fittC):
    secondDervs = {}
    for key in fittC.keys():
        secondDervs[key] = fittC[key].deriv().deriv().c[0]
        
    return secondDervs


def getShortestDistance2Curve(testPoint, curveFunc, maxItr, verbose=False):
    
    def minimizationObjective(X):
        x,y = X
        return np.sqrt( (x - testPoint[0])**2 + (y - testPoint[1])**2 )

    def minimizationCritaria(X):
        #fmin_cobyla will make sure this is always >= 0. So I'm making sure this is > 0 only when the point is on the curve.
        x,y = X
        return abs(curveFunc(x) - y)*-1

    minDistanceSoFar = sys.maxsize
    initialGuess = testPoint
    
    for itr in range(maxItr):
        projectedPoint = fmin_cobyla(minimizationObjective, x0=initialGuess, cons=[minimizationCritaria])
        curDistance = minimizationObjective(projectedPoint)
        
        
        if verbose:
            print('itr:  ' + str(itr))
            print('minDistanceSoFar:   ' + str(minDistanceSoFar))
            print('projectedPoint is:   ' + str(projectedPoint))
            print('shortestDistance is:  ' + str(curDistance))

            x = np.linspace(-100, 1000, 100)
            plt.plot(x, curveFunc(x), 'r-', label='f(x)')
            plt.plot(testPoint[0], testPoint[1], 'bo', label='testPoint')
            plt.plot(projectedPoint[0], projectedPoint[1], 'bx', label='projectedPoint')
            plt.plot([testPoint[0], projectedPoint[0]], [testPoint[1], projectedPoint[1]], 'g-', label='shortest distance')
            plt.axis('equal')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(loc='best')
            plt.show()
        
        
        if curDistance < minDistanceSoFar:
            minDistanceSoFar = curDistance
        else:
            break
            
    projectedPointAsInt = [int(projectedPoint[0]), int(projectedPoint[1])]
    return curDistance, projectedPointAsInt


def reject2_soft_removeBadCurves(filteredResultsDict, path2saveCurves, rejectedPredsDir, curvDerivTreshDic, verbose=False):
    
    cleanedUpResultsDict = deepcopy(filteredResultsDict)
    
    for resKey in filteredResultsDict.keys():
        deletedKeypointsCount = 0
        totalNbOfKeyPoints = 0
        deletedCurvesCount = 0
        totalNbOfCurves = 0
        problematicFramesCount = 0
        
        for time in filteredResultsDict[resKey].keys():
            frameHadIssues = False
            fileName = filteredResultsDict[resKey][time]['fileName'].split('/')[-1]
            filePath = path2saveCurves + fileName
            
            
            for keypointsKey in curvDerivTreshDic.keys():
                totalNbOfKeyPoints += 1
                
                #***********************************************************************************************#
                #reject4 no curve calculated for this keypoint type (because there was less than 2 of them)
                if keypointsKey not in filteredResultsDict[resKey][time]['fittedCurves']:
                    del(cleanedUpResultsDict[resKey][time][keypointsKey])
                    #shutil.copy(filePath, rejectedPredsDir)
                    
                    if verbose:
                        print('\nrejected:\n' + str(filePath) + '\nbecause no curve was fitted for this keypoint type. For keypointkey: ' + keypointsKey)
                    
                    deletedKeypointsCount += 1
                    frameHadIssues = True
                #***********************************************************************************************#
                    
            
            #***********************************************************************************************#
            #reject5 2nd derivative of fitted curve for this keypoint type doesnt satisfy thresholds        
            for keypointsKey in filteredResultsDict[resKey][time]['2ndDerivFittedCurves'].keys():
                totalNbOfCurves += 1
                derivative = filteredResultsDict[resKey][time]['2ndDerivFittedCurves'][keypointsKey]
                
                if derivative < curvDerivTreshDic[keypointsKey][0]\
                or derivative > curvDerivTreshDic[keypointsKey][1]:
                    
                    if keypointsKey in cleanedUpResultsDict[resKey][time]:
                        del(cleanedUpResultsDict[resKey][time][keypointsKey])
                        del(cleanedUpResultsDict[resKey][time]['2ndDerivFittedCurves'][keypointsKey])
                        del(cleanedUpResultsDict[resKey][time]['fittedCurves'][keypointsKey])

                        shutil.copy(filePath, rejectedPredsDir)
                        
                        if verbose:
                            print('\nreject2_soft rejected:\n' + str(filePath) + '\nbecause derivative of fitted curve didnt fit the criteria. For keypointkey: ' + keypointsKey)
                            
                        deletedCurvesCount += 1
                        frameHadIssues = True
                #***********************************************************************************************#
                

            if frameHadIssues:
                problematicFramesCount += 1

        print('\nfor results set: ' + resKey + ':\n---Total of  ' + str(problematicFramesCount) + '  frames had issues, so not all of their info were copied from filteredResults into cleanedUpResultsDict.\n'+ '---rejected  ' + str(deletedCurvesCount) + '  curves out of the  ' + str(totalNbOfCurves) +'  in total because the derivative of fitted curve didnt fit the criteria.' + '\n---rejected  ' + str(deletedKeypointsCount) + '  keypoints out of  '+ str(totalNbOfKeyPoints) +'  in total because we couldnt fit a curve to them.\n')
        
        return cleanedUpResultsDict


def reject3_soft_replaceLandmarks2farFromCurve(cleanedUpResultsDict, path2SavedCurves, path2saveRejectedLandmarks, maxDistanceBtwLandmarkAndCurve_replacement, maxDistanceBtwLandmarkAndCurve_rejection, maxItr_findingShortestDistance2Curve, replaceBadLandmarkWithProjection, verbose=False):
    #reject 6. Get rid of zeros and landmarks too far from their curves.
    #keypoints not in validated will not be used for image2image registration
    
    for resKey in cleanedUpResultsDict.keys():
        landmarksNotCopiedToValid = 0
        replacedLandmarks2farFromCurve = 0
        totalNbOfLandmakrs = 0
        
        for time in cleanedUpResultsDict[resKey].keys():
            cleanedUpResultsDict[resKey][time]['validatedKeypoints'] = []
            
            fileName = cleanedUpResultsDict[resKey][time]['fileName'].split('/')[-1]
            filePath = path2SavedCurves + fileName
            
            
            # setup the confidences array to store number of good landmarks
            confidences = {}
            for tnb in range(NUMBER_OF_TEETH):
                confidences['tooth_' + str(tnb+1)] = 0
            ###############################################################
            
            
            for ln in range(numberOflandmarksIncludingToothTip):
                keypointType = 'keypoints_' + str(ln + 1)
                
                if keypointType in cleanedUpResultsDict[resKey][time]['2ndDerivFittedCurves'].keys():
                    
                    for tn in range(NUMBER_OF_TEETH):
                        curKeyPoint = cleanedUpResultsDict[resKey][time][keypointType][tn]

                        if not (curKeyPoint[0] == 0 and curKeyPoint[1] == 0):
                            totalNbOfLandmakrs +=1

                            dist2Curv, projectedPoint = getShortestDistance2Curve(curKeyPoint, cleanedUpResultsDict[resKey][time]['fittedCurves'][keypointType], maxItr_findingShortestDistance2Curve)

                            
                            if dist2Curv > maxDistanceBtwLandmarkAndCurve_rejection:
                                shutil.copy(filePath, path2saveRejectedLandmarks)
                                landmarksNotCopiedToValid += 1

                                if verbose:
                                    print('\nrejected a kypoint in:\n' + str(filePath) + '\nbecause the keypoint was too far from the curve. For keypointkey: ' + keypointType + '   keyPont:  ' + str(curKeyPoint) + '\ndistance to curve was:  ' + str(dist2Curv) + '.')
                                    
                            elif dist2Curv > maxDistanceBtwLandmarkAndCurve_replacement and\
                            replaceBadLandmarkWithProjection == True:
                                
                                    cleanedUpResultsDict[resKey][time]['validatedKeypoints'].append(projectedPoint)

                                    cleanedUpResultsDict[resKey][time][keypointType][tn] = projectedPoint
                                    cleanedUpResultsDict[resKey][time]['keypointsForTooth_' + str(tn + 1)][ln] = projectedPoint
                                    replacedLandmarks2farFromCurve += 1

                                    if verbose:
                                        print('\nreplaced a kypoint in:\n' + str(filePath) + '\nbecause it was too far from the curve. For keypointkey: ' + keypointType + '   keyPont:  ' + str(curKeyPoint) + '\ndistance to curve was:  ' + str(dist2Curv) + '\nWe replaced this keypoint with its projection to the curve instead which was found to be:  ' + str(projectedPoint) )
                                    
                            else :
                                cleanedUpResultsDict[resKey][time]['validatedKeypoints'].append(curKeyPoint)
                                confidences['tooth_' + str(tn + 1)] += 1
                                

                                 

                                    
            cleanedUpResultsDict[resKey][time]['confidences'] = confidences
            
            
    print('\nfor results set: ' + resKey + ':\n---replaced  ' + str(replacedLandmarks2farFromCurve) + '  landmarks out of  ' + str(totalNbOfLandmakrs)+ '  in total with their projections on the curve. Total of  '+ str(landmarksNotCopiedToValid) + '  landmarks were NOT added to validated and frames were copies in path2saveRejectedLandmarks.\n')


def countValidLandmarks(resDictForFrame, minNbOfDetectedPointsForOtherLandmarkTypes):
    detectedLandmakTypesCount = 0
    detectedLipShroudCount = 0
    detectedToothTipCount = 0
    
    
    if 'keypoints_1' in resDictForFrame:
        detectedToothTipCount = len([x for x in resDictForFrame['keypoints_1'] if x in resDictForFrame['validatedKeypoints']])
        
    
    if 'keypoints_2' in resDictForFrame:
        detectedLipShroudCount = len([x for x in resDictForFrame['keypoints_2'] if x in resDictForFrame['validatedKeypoints']])

    
    if 'keypoints_3' in resDictForFrame and\
    len([x for x in resDictForFrame['keypoints_3'] if x in resDictForFrame['validatedKeypoints']]) >= minNbOfDetectedPointsForOtherLandmarkTypes:
        detectedLandmakTypesCount += 1
        
        
    if 'keypoints_4' in resDictForFrame and\
    len([x for x in resDictForFrame['keypoints_4'] if x in resDictForFrame['validatedKeypoints']]) >= minNbOfDetectedPointsForOtherLandmarkTypes:
        detectedLandmakTypesCount += 1
        
        
    if 'keypoints_5' in resDictForFrame and\
    len([x for x in resDictForFrame['keypoints_5'] if x in resDictForFrame['validatedKeypoints']]) >= minNbOfDetectedPointsForOtherLandmarkTypes:
        detectedLandmakTypesCount += 1

        
    return detectedLandmakTypesCount, detectedLipShroudCount, detectedToothTipCount


def reject4_notEnoughValidLandmarks(cleanedUpResultsDict, rejectedResultsDir, minNbOfDetectedPointsForTTandLS, minNbOfDetectedPointsForOtherLandmarkTypes, numberOfTeeth, verbose=False):

    finalResultsDict = deepcopy(cleanedUpResultsDict)
    
    for resKey in cleanedUpResultsDict.keys():
        deletedCount = 0
        totalFramesCount = 0
        
        for time in cleanedUpResultsDict[resKey].keys():
            totalFramesCount += 1
            alreadyRejected = False
            filePath = cleanedUpResultsDict[resKey][time]['fileName']
            fileName = filePath.split('/')[-1]
            path2FrameWithCurve = path2saveCurves + fileName
            
            
            detectedLandmakTypesCount, detectedLipShroudCount, detectedToothTipCount = countValidLandmarks(
                cleanedUpResultsDict[resKey][time],
                minNbOfDetectedPointsForOtherLandmarkTypes
            )
            
            #reject1 not enough teeth boxes + tips detected.
            if detectedToothTipCount < minNbOfDetectedPointsForTTandLS:
                shutil.copy(path2FrameWithCurve, rejectedResultsDir)
                
                if verbose:
                    print('\nrejected:\n' + str(filePath) + '\nbecause not enough toothTips were detected. We counted   ' + str(detectedToothTipCount) + '  toothTips for this log.')
                    
                del finalResultsDict[resKey][time]
                
                if not alreadyRejected:
                    deletedCount += 1
                    alreadyRejected = True
                
            
            #reject2 not enough lipShroud landmarks are detected
            elif detectedLipShroudCount < minNbOfDetectedPointsForTTandLS:
                shutil.copy(path2FrameWithCurve, rejectedResultsDir)
                
                if verbose:
                    print('\nrejected:\n' + str(filePath) + '\nbecause not enough lipshrouds were detected. We counted   ' + str(detectedLipShroudCount) + '  lipShrouds for this log.')
                
                del finalResultsDict[resKey][time]
                
                if not alreadyRejected:
                    deletedCount += 1
                    alreadyRejected = True
                
                
            #reject3 not enough registeration landmarks detected
            elif detectedLandmakTypesCount < 1:
                shutil.copy(path2FrameWithCurve, rejectedResultsDir)
                
                if verbose:
                    print('\nrejected:\n' + str(filePath) + '\nbecause no other landmarks types aside from TT and LS were detected.')
                del finalResultsDict[resKey][time]
                if not alreadyRejected:
                    deletedCount += 1
                    alreadyRejected = True
            
        
                
        print('\nfor results set: ' + resKey + '\n---rejected  ' + str(deletedCount) + '  logs out of the total of  ' + str(totalFramesCount) +'  which were not copied from cleanedUpResultsDict into finalResultsDict.')
        
        return finalResultsDict


def getReferenceRatios(refKey):
    
    path2RefImage = mainPath + 'referenceFrames/images/' + refKey + '.png'
    path2Reflabel = mainPath + 'referenceFrames/labels/' + refKey + '_landmarkCoords.xml'
    refkeyPointsDic = getRefDict(path2Reflabel, path2RefImage)

    def getRatio(lm1, lm2, lm3, lm4):
        return (lm2[1]-lm1[1])/(lm4[1]-lm3[1])
    
    if NUMBER_OF_TEETH > 6:
        minAllowedRatiosDict = {
            'tooth_1':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1'], 
                     refkeyPointsDic['castLip_1']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['castLip_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['castLip_1']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ), 
            },
            'tooth_2':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2'], 
                     refkeyPointsDic['castLip_2']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['castLip_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['castLip_2']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ), 
            },
            'tooth_3':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3'], 
                     refkeyPointsDic['castLip_3']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['castLip_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['castLip_3']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ), 
            },
            'tooth_4':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4'], 
                     refkeyPointsDic['castLip_4']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['castLip_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['castLip_4']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ), 
            },
            'tooth_5':{
                 'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5'], 
                     refkeyPointsDic['castLip_5']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['castLip_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['castLip_5']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
            },
            'tooth_6':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6'], 
                     refkeyPointsDic['castLip_6']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['castLip_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['castLip_6']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ), 
            },
            'tooth_7':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['liftingEye_7'], 
                     refkeyPointsDic['bucketLandmark_7']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['liftingEye_7'], 
                     refkeyPointsDic['castLip_7']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['castLip_7'], 
                     refkeyPointsDic['bucketLandmark_7']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['liftingEye_7']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['castLip_7']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_7'],
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['lipShroud_7'], 
                     refkeyPointsDic['bucketLandmark_7']
                 ), 
            },
            'tooth_8':{
                 'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['liftingEye_8'], 
                     refkeyPointsDic['bucketLandmark_8']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['liftingEye_8'], 
                     refkeyPointsDic['castLip_8']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['castLip_8'], 
                     refkeyPointsDic['bucketLandmark_8']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['liftingEye_8']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['castLip_8']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_8'],
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['lipShroud_8'], 
                     refkeyPointsDic['bucketLandmark_8']
                 ),
            }
        }
        
    else:
        minAllowedRatiosDict = {
            'tooth_1':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1'], 
                     refkeyPointsDic['castLip_1']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['castLip_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['liftingEye_1']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['castLip_1']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_1'],
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['lipShroud_1'], 
                     refkeyPointsDic['bucketLandmark_1']
                 ), 
            },
            'tooth_2':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2'], 
                     refkeyPointsDic['castLip_2']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['castLip_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['liftingEye_2']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['castLip_2']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_2'],
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['lipShroud_2'], 
                     refkeyPointsDic['bucketLandmark_2']
                 ), 
            },
            'tooth_3':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3'], 
                     refkeyPointsDic['castLip_3']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['castLip_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['liftingEye_3']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['castLip_3']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_3'],
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['lipShroud_3'], 
                     refkeyPointsDic['bucketLandmark_3']
                 ), 
            },
            'tooth_4':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4'], 
                     refkeyPointsDic['castLip_4']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['castLip_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['liftingEye_4']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['castLip_4']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_4'],
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['lipShroud_4'], 
                     refkeyPointsDic['bucketLandmark_4']
                 ), 
            },
            'tooth_5':{
                 'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5'], 
                     refkeyPointsDic['castLip_5']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['castLip_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['liftingEye_5']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['castLip_5']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_5'],
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['lipShroud_5'], 
                     refkeyPointsDic['bucketLandmark_5']
                 ),
            },
            'tooth_6':{
                'tt2ls_over_le2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ),
                'tt2ls_over_le2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6'], 
                     refkeyPointsDic['castLip_6']
                 ),
                'tt2ls_over_cl2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['castLip_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ),
                'tt2ls_over_ls2le': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['liftingEye_6']
                 ),
                'tt2ls_over_ls2cl': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['castLip_6']
                 ),
                'tt2ls_over_ls2bk': 
                 getRatio(
                     refkeyPointsDic['toothTip_6'],
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['lipShroud_6'], 
                     refkeyPointsDic['bucketLandmark_6']
                 ), 
            },
        }
        
    
    return minAllowedRatiosDict


def getRefDict(path2labelXml, path2UnlabeledImage):
    refkeyPointsDic = {}
    
    img = cv2.imread(path2UnlabeledImage)
    if img is None:
        print('faild to find ref image. You probably forgot to set its name in config')
    height, width, dim=img.shape

    with open(path2labelXml) as fd:
        tempDic = xmltodict.parse(fd.read())
        
        for typeK in tempDic['hs_frame_wear_landmarks'].keys():
            if not typeK == 'img_name':
                for nbK in tempDic['hs_frame_wear_landmarks'][typeK]:
                    xcor = int(float(tempDic['hs_frame_wear_landmarks'][typeK][nbK]['@x']) * width)
                    ycor = int(float(tempDic['hs_frame_wear_landmarks'][typeK][nbK]['@y']) * height)
                    refkeyPointsDic[nbK] = [xcor, ycor]
                
    return refkeyPointsDic


def visualizeRefDict(refDict, path2refImage, numberOfTeeth):
    numberOfLandmarks = 5

    
    trgAr = []

    landmakrsList = ['toothTip_', 'lipShroud_', 'liftingEye_', 'castLip_', 'bucketLandmark_']

    for i in range(numberOfLandmarks):
        keypointsKey = 'keypoints_' + str(i + 1 )

        for j in range(numberOfTeeth):
            trgAr.append(refkeyPointsDic[landmakrsList[i] + str(j+1)])

    trgPoints = getRegPointSetFromArray(trgAr)


    refImage = cv2.imread(path2RefImage)
    
    displayPointsSetOnImage(refImage, trgPoints)
    
    
    #plt.imshow(refImage)
    #plt.title('visualizing refDict')
    #plt.show()
    
    return refImage


def getRegPointSetFromArray(validPoints):
    outPoints = np.zeros((1, len(validPoints), 2), np.int32)
    counter = 0
    
    for pp in validPoints:
        outPoints[0, counter, 0] = int(pp[0])
        outPoints[0, counter, 1] = int(pp[1])
        counter+=1
  
    return outPoints


def getDistanceBtwPoints(point1, point2):
    #return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1]) 
    #return math.sqrt(math.pow((point2[0] - point1[0]), 2) + math.pow((point2[1] - point1[1]), 2) )
    return math.sqrt(math.pow((point2[1] - point1[1]), 2) )


def getRegError(registeredPoints, targetPointSet):
    totalError = 0
    counter = 0
    
    for i in range(len(targetPointSet[0])):
        totalError += getDistanceBtwPoints(registeredPoints[0][i], targetPointSet[0][i])
        counter+=1
        
    return totalError/counter


def getRegisteredPointsV3(refkeyPointsDic, resDicForFrame, numberOfTeeth, verbose = False, numberOfLandmarks = 5):
    errorWithAll = 10000
    errorJustLipShrouds = 10000
    resultsWithAll = []
    resultsWithJustLipShroud = []
    
    
    
    ###############################################################################################################
    #################################### USE ALL LANDMARKS ########################################
    ###############################################################################################################
    srcAr = []
    trgAr = []
    indexToKeyMap = []
    resultsArRigid = [[[0, 0] for j in range(numberOfLandmarks)] for i in range(numberOfTeeth)]


    landmakrsList = ['toothTip_', 'lipShroud_', 'liftingEye_', 'castLip_', 'bucketLandmark_']

    for i in range(1, numberOfLandmarks, 1): #do not use toothtip (i=0) landmarks in calculating transfromation
        keypointsKey = 'keypoints_' + str(i + 1 )

        if keypointsKey in resDicForFrame:
            keyPoints = resDicForFrame[keypointsKey]

            for j in range(numberOfTeeth):
                if keyPoints[j] in resDicForFrame['validatedKeypoints']:
                    srcAr.append(keyPoints[j])
                    trgAr.append(refkeyPointsDic[landmakrsList[i] + str(j+1)])


    srcPoints = getRegPointSetFromArray(srcAr)
    trgPoints = getRegPointSetFromArray(trgAr)

    transformationRigid = cv2.estimateRigidTransform(srcPoints, trgPoints, False)

 
    points2moveAr = []
    for i in range(numberOfLandmarks):
        keypointsKey = 'keypoints_' + str(i + 1 )

        if keypointsKey in resDicForFrame:
            keyPoints = resDicForFrame[keypointsKey]

            for j in range(numberOfTeeth):
                if keyPoints[j] in resDicForFrame['validatedKeypoints']:
                    indexToKeyMap.append({'landmarkNb': i, 'toothNb':j})
                    points2moveAr.append(keyPoints[j])


    points2move = getRegPointSetFromArray(points2moveAr)

    
    if not transformationRigid is None:
        if transformationRigid.any():
            transformedPointsRigid = cv2.transform(points2move, transformationRigid)

            for i in range(transformedPointsRigid.shape[1]):
                pointRigid =  transformedPointsRigid[0, i, :].tolist()
                resultsArRigid[indexToKeyMap[i]['toothNb']][indexToKeyMap[i]['landmarkNb']] = pointRigid

            curError = getRegError(transformedPointsRigid, srcPoints)
            
            if verbose:
                print('RegidTransformation error using all landmarks was:  ' + str(curError) + '\n')
                
                
            errorWithAll = curError
            resultsWithAll = resultsArRigid

    else:
        print('failed to find Rigid transformation using all landmarks for:')
        print(resDicForFrame['fileName'])
        
        
        
        
        
        
        
        
             
    srcAr = []
    trgAr = []
    indexToKeyMap = []
    resultsArRigid = [[[0, 0] for j in range(numberOfLandmarks)] for i in range(numberOfTeeth)]


    landmakrsList = ['toothTip_', 'lipShroud_', 'liftingEye_', 'castLip_', 'bucketLandmark_']

    i = 1 # use only lipShroud landmarks for registration
    keypointsKey = 'keypoints_' + str(i + 1 )

    if keypointsKey in resDicForFrame:
        keyPoints = resDicForFrame[keypointsKey]

        for j in range(numberOfTeeth):
            if keyPoints[j] in resDicForFrame['validatedKeypoints']:
                srcAr.append(keyPoints[j])
                trgAr.append(refkeyPointsDic[landmakrsList[i] + str(j+1)])


    srcPoints = getRegPointSetFromArray(srcAr)
    trgPoints = getRegPointSetFromArray(trgAr)

    transformationRigid = cv2.estimateRigidTransform(srcPoints, trgPoints, False)

 
    points2moveAr = []
    for i in range(numberOfLandmarks):
        keypointsKey = 'keypoints_' + str(i + 1 )

        if keypointsKey in resDicForFrame:
            keyPoints = resDicForFrame[keypointsKey]

            for j in range(numberOfTeeth):
                if keyPoints[j] in resDicForFrame['validatedKeypoints']:
                    indexToKeyMap.append({'landmarkNb': i, 'toothNb':j})
                    points2moveAr.append(keyPoints[j])


    points2move = getRegPointSetFromArray(points2moveAr)
    
    if not transformationRigid is None:
        if transformationRigid.any():
            transformedPointsRigid = cv2.transform(points2move, transformationRigid)

            for i in range(transformedPointsRigid.shape[1]):
                pointRigid =  transformedPointsRigid[0, i, :].tolist()
                resultsArRigid[indexToKeyMap[i]['toothNb']][indexToKeyMap[i]['landmarkNb']] = pointRigid

            curError = getRegError(transformedPointsRigid, srcPoints)
            
            if verbose:
                print('RegidTransformation error using just LipShrouds was:  ' + str(curError) + '\n')
                
                
            errorJustLipShrouds = curError
            resultsWithJustLipShroud = resultsArRigid
            
            if errorJustLipShrouds*regEr_WithLs_lessThan_withAll_multiple < errorWithAll:
                print('RegidTransformation using just lipShroud was better than using all landmarks for log:\n' + str(resDicForFrame['fileName']) +  '\nusing all landmarks the error was  ' + str(errorWithAll) + '  using just lipshrouds it was  ' + str(errorJustLipShrouds) + '\n')

                return [], 10000
                

    else:
        print('failed to find Rigid transformation using just LipShrouds for:')
        print(resDicForFrame['fileName'])
        

        
    return resultsWithAll, errorWithAll


def getLandmarkOrApproxY(keypointsForThisTooth, fittedCurves):
        ttY = keypointsForThisTooth[0][1]
        ttX = keypointsForThisTooth[0][0]
        lsY = keypointsForThisTooth[1][1]
        leY = keypointsForThisTooth[2][1]
        clY = keypointsForThisTooth[3][1]
        bkY = keypointsForThisTooth[4][1]
        
        if not ttY or ttY <= 0:
            print("error: missing tooth tip point landmark. Returning None. This should NOT happen")
            return None, None, None, None, None
        
        if not lsY or lsY <= 0:
            if 'keypoints_2' in fittedCurves:
                lsY = fittedCurves['keypoints_2'](ttX)
            else:
                lsY = None
            
        if not leY or leY <= 0:
            if 'keypoints_3' in fittedCurves:
                leY = fittedCurves['keypoints_3'](ttX)
            else:
                leY = None
                
        if not clY or clY <= 0:
            if 'keypoints_4' in fittedCurves:
                clY = fittedCurves['keypoints_4'](ttX)
            else:
                clY = None
                
        if not bkY or bkY <= 0:
            if 'keypoints_5' in fittedCurves:
                bkY = fittedCurves['keypoints_5'](ttX)
            else:
                bkY = None
                
        return ttY, lsY, leY, clY, bkY


def areRatiosOk(lengths, referenceRatiosDict, maxAllowedDistBtwRatiosDict, toothKey):
    refsAreOk = True
    if\
    'le2bk' in lengths and abs( (lengths['tt2ls'] / lengths['le2bk']) - referenceRatiosDict[toothKey]['tt2ls_over_le2bk'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_le2bk'] or\
    'cl2bk' in lengths and abs( (lengths['tt2ls'] / lengths['cl2bk']) - referenceRatiosDict[toothKey]['tt2ls_over_cl2bk'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_cl2bk'] or\
    'le2cl' in lengths and abs( (lengths['tt2ls'] / lengths['le2cl']) - referenceRatiosDict[toothKey]['tt2ls_over_le2cl'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_le2cl'] or\
    'ls2bk' in lengths and abs( (lengths['tt2ls'] / lengths['ls2bk']) - referenceRatiosDict[toothKey]['tt2ls_over_ls2bk'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_ls2bk'] or\
    'ls2cl' in lengths and abs( (lengths['tt2ls'] / lengths['ls2cl']) - referenceRatiosDict[toothKey]['tt2ls_over_ls2cl'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_ls2cl'] or\
    'ls2le' in lengths and abs( (lengths['tt2ls'] / lengths['ls2le']) - referenceRatiosDict[toothKey]['tt2ls_over_ls2le'] ) > maxAllowedDistBtwRatiosDict[toothKey]['tt2ls_over_ls2le']:
        return False
    else:
        return True


def getAllLengthsForOneTooth(resDicForFrame, refKeyPointsDic, toothNumber, keypointsTypeString, referenceRatiosDict, maxAllowedDistBtwRatiosDict, maxAllowed_detectedAboveRefTT, minAllowed_toothLength, maxAllowed_toothLength, path2saveRejected, filePath, verbose = False):
    #toothNumber starts from 1
    lengths = {}
    landmarks = {}

    if keypointsTypeString == 'keypointsForTooth_':
        fittedCurves = resDicForFrame['fittedCurves']
    elif keypointsTypeString == 'rigid_keypointsForTooth_':
        if 'rigid_fittedCurves' in resDicForFrame.keys():
            fittedCurves = resDicForFrame['rigid_fittedCurves']
        else:
            return lengths, landmarks
    elif keypointsTypeString == 'affine_keypointsForTooth_':
        fittedCurves = resDicForFrame['affine_fittedCurves']
    else:
        print('ERROR: getAllLengthsForOneTooth  got a keypointsTypeString that was not recognized')
        
        
    ttY, lsY, leY, clY, bkY = getLandmarkOrApproxY(
        resDicForFrame[keypointsTypeString + str(toothNumber)],
        fittedCurves
    )
    
    if (ttY, lsY, leY, clY, bkY) == (None, None, None, None, None):
        print(resDicForFrame['fileName'])
    
    
    toothKey = 'Tooth_' + str(toothNumber)

    #lengths['box'] = resDicForFrame[toothKey][6]
    
    if refKeyPointsDic is None:
        print('ERROR in getAllLengthsForOneTooth did not find: ' + 'lipShroud_' + str(toothNumber) + '  in refkeyPointsDic')
    else:
        refLsY = int(refKeyPointsDic['lipShroud_' + str(toothNumber)][1])
        refTtY = int(refKeyPointsDic['toothTip_' + str(toothNumber)][1])
        
        
    '''
    # ********** EARLY RETURN1***************
    if not keypointsTypeString == 'keypointsForTooth_' and (refTtY - ttY) > maxAllowed_detectedAboveRefTT:
        #shutil.copy(filePath, path2saveRejected)
        try:
            shutil.move(path2visFinal + filePath.split('/')[-1], path2saveRejected)
        except:
            pass
        
        print('getAllLengthsForOneTooth did not receive info for log\n\n' + filePath + '\n\nbecause the registered toothTip was too far above the reference toothTip. This happened for:\n---tooth  ' + str(toothKey) + '\n--registration type  ' + str(keypointsTypeString) + '\n\n\n')
        
        return {}, {}
    # **************************************
    '''


    if lsY:
        lengths['tt2ls'] = lsY - ttY
        lengths['tt2_ref_ls'] = refLsY - ttY
        landmarks['ttY'] = ttY
        landmarks['lsY'] = lsY
        
        
        # ********** EARLY RETURN1***************
        if keypointsTypeString == 'keypointsForTooth_' and (lengths['tt2ls'] > maxAllowed_toothLength or lengths['tt2ls'] < minAllowed_toothLength):

            #shutil.copy(filePath, path2saveRejected)
            try:
                shutil.move(path2visFinal + filePath.split('/')[-1], path2saveRejected)
            except:
                pass
            
            if verbose:
                print('getAllLengthsForOneTooth did not receive info for log\n\n' + filePath + '\n\nbecause the detected toothLengths was not withing the acceptable range. This happened for:\n---tooth  ' + str(toothKey) + '\n--registration type  ' + str(keypointsTypeString) + '\npredicted toothLength:  ' + str(lengths['tt2ls']) + '\n\n\n')

            return {}, {}
        # **************************************

    else:
        if verbose:
            print("\n\nError:getAllLengthsForOneTooth could not find the lipShroud point for this image for key:\n" + str(keypointsTypeString + str(toothNumber)) + '\nandResults:\n' +str(resDicForFrame[keypointsTypeString + str(toothNumber)]) + '\n' + toothKey + '\n this should not have happened')

        
        
    if leY:
        landmarks['leY'] = leY
        if(leY - lsY) > 0:
            lengths['ls2le'] = leY - lsY
            
        if clY:
            landmarks['clY'] = clY
            if(clY - leY ) > 0:
                lengths['le2cl'] = clY - leY 
            
        if bkY:
            landmarks['bkY'] = bkY
            if(bkY - leY) > 0:
                lengths['le2bk'] = bkY - leY

   
    if verbose:
        if 'le2bk' in lengths:
            print('le2bk:  ' + str(lengths['le2bk']))
            print()

        if 'cl2bk' in lengths:
            print('cl2bk:  ' + str(lengths['cl2bk']))
            print()

        if 'le2cl' in lengths:
            print('le2cl:  ' + str(lengths['le2cl']))
            print()

        if 'ls2bk' in lengths:
            print('ls2bk:  ' + str(lengths['ls2bk']))
            print()

        if 'ls2cl' in lengths:
            print('ls2cl:  ' + str(lengths['ls2cl']))
            print()

        if 'ls2le' in lengths:
            print('ls2le:  ' + str(lengths['ls2le']))
            print()
        
        
        
     # ********** #reject from validation plots ***************
    if areRatiosOk(lengths, referenceRatiosDict, maxAllowedDistBtwRatiosDict, 'tooth_' + str(toothNumber)):
        lengths['valid_tt2ls'] = lengths['tt2ls']
    else:
        pass
    # **************************************
        
        
        
        
    return lengths, landmarks


def getAllLengthsForOneRegisterationType(resDicForThisResultsSet, refkeyPointsDic, toothNumber,refKey, regTypeKeyword, referenceRatiosDict, maxAllowedDistBtwRatiosDict, maxAllowed_detectedAboveRefTT,minAllowed_toothLength, maxAllowed_toothLength, path2saveRejected, verbose = False):
    
    #put everything in a list for easier plotting
    all_times_list  = set()
    all_tt2ls_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_ls2le_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_le2cl_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_le2bk_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_ls2cl_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_cl2bk_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    all_ls2bk_dict  = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    valid_tt2ls_dict= {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    tt2_ref_ls      = {'times':[], 'lengths':[], 'confidences':[], 'secondaryConfidences':[], 'logConfidence':[]}
    forSmoothing = {}

    nbOfSuccessfulCalcs = 0
    for time in sorted( resDicForThisResultsSet.keys() ):
        
        confidence = resDicForThisResultsSet[time]['confidences']['tooth_' + str(toothNumber)]
        secondaryConfidence = resDicForThisResultsSet[time]['secondaryConfidences']['tooth_' + str(toothNumber)]
        logConfidence = resDicForThisResultsSet[time]['logConfidence']
        
        if regTypeKeyword == 'keypointsForTooth_':
            lengths, landmarks = getAllLengthsForOneTooth(
                resDicForThisResultsSet[time],
                refkeyPointsDic,
                toothNumber,
                regTypeKeyword,
                referenceRatiosDict,
                maxAllowedDistBtwRatiosDict,
                maxAllowed_detectedAboveRefTT,
                minAllowed_toothLength,
                maxAllowed_toothLength,
                path2saveRejected,
                resDicForThisResultsSet[time]['fileName'],
                verbose = verbose
            )
        
        else:
            lengths, landmarks = getAllLengthsForOneTooth(
                resDicForThisResultsSet[time]['registrations'][refKey],
                refkeyPointsDic,
                toothNumber,
                regTypeKeyword,
                referenceRatiosDict,
                maxAllowedDistBtwRatiosDict,
                maxAllowed_detectedAboveRefTT,
                minAllowed_toothLength, 
                maxAllowed_toothLength,
                path2saveRejected,
                resDicForThisResultsSet[time]['fileName'],
                verbose = verbose
            )
            
            
        if len(lengths.keys()) > 0 and len(landmarks.keys()) > 0:
            nbOfSuccessfulCalcs += 1
        else:
            if verbose:
                print('getAllLengthsForOneRegisterationType did not receive info for log\n\n' + resDicForThisResultsSet[time]['fileName'] + '\n\nbecause either the registered toothTip was too far above the reference toothTip or detected toothLengths was not withing the acceptable range. This happened for:\n---reference frame  ' + str(refKey) + '\n--registration type  ' + str(regTypeKeyword) +  '\n---time step   ' + str(time) + '\n\n\n')

        
        if 'tt2ls' in lengths:
            all_tt2ls_dict['lengths'].append(lengths['tt2ls'])
            all_tt2ls_dict['times'].append(time)
            all_tt2ls_dict['confidences'].append(confidence)
            all_tt2ls_dict['secondaryConfidences'].append(secondaryConfidence)
            all_tt2ls_dict['logConfidence'].append(logConfidence)
            
            all_times_list.add(time)
            
            forSmoothing[time] = lengths['tt2ls']
            
            
            
        if 'ls2le' in lengths:
            all_ls2le_dict['lengths'].append(lengths['ls2le'])
            all_ls2le_dict['times'].append(time)
            all_ls2le_dict['confidences'].append(confidence)
            all_ls2le_dict['secondaryConfidences'].append(secondaryConfidence)
            all_ls2le_dict['logConfidence'].append(logConfidence)
        if 'le2cl' in lengths:
            all_le2cl_dict['lengths'].append(lengths['le2cl'])
            all_le2cl_dict['times'].append(time)
            all_le2cl_dict['confidences'].append(confidence)
            all_le2cl_dict['secondaryConfidences'].append(secondaryConfidence)
            all_le2cl_dict['logConfidence'].append(logConfidence)
        if 'le2bk' in lengths:
            all_le2bk_dict['lengths'].append(lengths['le2bk'])
            all_le2bk_dict['times'].append(time)
            all_le2bk_dict['confidences'].append(confidence)
            all_le2bk_dict['secondaryConfidences'].append(secondaryConfidence)
            all_le2bk_dict['logConfidence'].append(logConfidence)
        if 'cl2bk' in lengths:
            all_cl2bk_dict['lengths'].append(lengths['cl2bk'])
            all_cl2bk_dict['times'].append(time)
            all_cl2bk_dict['confidences'].append(confidence)
            all_cl2bk_dict['secondaryConfidences'].append(secondaryConfidence)
            all_cl2bk_dict['logConfidence'].append(logConfidence)
        if 'ls2cl' in lengths:
            all_ls2cl_dict['lengths'].append(lengths['ls2cl'])
            all_ls2cl_dict['times'].append(time)
            all_ls2cl_dict['confidences'].append(confidence)
            all_ls2cl_dict['secondaryConfidences'].append(secondaryConfidence)
            all_ls2cl_dict['logConfidence'].append(logConfidence)
        if 'ls2bk' in lengths:
            all_ls2bk_dict['lengths'].append(lengths['ls2bk'])
            all_ls2bk_dict['times'].append(time)
            all_ls2bk_dict['confidences'].append(confidence)
            all_ls2bk_dict['secondaryConfidences'].append(secondaryConfidence)
            all_ls2bk_dict['logConfidence'].append(logConfidence)
        if 'valid_tt2ls' in lengths:
            valid_tt2ls_dict['lengths'].append(lengths['valid_tt2ls'])
            valid_tt2ls_dict['times'].append(time)
            valid_tt2ls_dict['confidences'].append(confidence)
            valid_tt2ls_dict['secondaryConfidences'].append(secondaryConfidence)
            valid_tt2ls_dict['logConfidence'].append(logConfidence)
        if 'tt2_ref_ls' in lengths:
            tt2_ref_ls['lengths'].append(lengths['tt2_ref_ls'])
            tt2_ref_ls['times'].append(time)
            tt2_ref_ls['confidences'].append(confidence)
            tt2_ref_ls['secondaryConfidences'].append(secondaryConfidence)
            tt2_ref_ls['logConfidence'].append(logConfidence)



    outDict = {
        'landmarks'       :  landmarks,
        'all_times_list'  : all_times_list,
        'all_tt2ls_dict'  : all_tt2ls_dict,
        'all_ls2le_dict'  : all_ls2le_dict,
        'all_le2cl_dict'  : all_le2cl_dict,
        'all_le2bk_dict'  : all_le2bk_dict,
        'all_ls2cl_dict'  : all_ls2cl_dict,
        'all_cl2bk_dict'  : all_cl2bk_dict,
        'all_ls2bk_dict'  : all_ls2bk_dict,
        'valid_tt2ls_dict': valid_tt2ls_dict,
        'tt2_ref_ls'      : tt2_ref_ls,
    }
    
    return outDict, forSmoothing


def getAllLengthsAndLandmarks(finalResultsDict, numberOfTeeth, referenceRatiosDict, maxAllowedDistBtwRatiosDict, maxAllowed_detectedAboveRefTT, minAllowed_toothLength, maxAllowed_toothLength, path2saveRejected, verbose = False):
    
    listOfKeyWords = ['keypointsForTooth_', 'rigid_keypointsForTooth_']
    
    toothLengthsDict = {}
    
    for resKey in finalResultsDict.keys():
        toothLengthsDict[resKey] = {}
        totalNbOfExpectedTimeSteps = len(finalResultsDict[resKey].keys()) 
        totalNbOfSuccessfullCalcs = 0
    
        toothLengthsDict[resKey]['forSmoothing'] = getEmptySmoothedDict(
            numberOfTeeth,
            ['rigid_keypointsForTooth_'],
            references2use
        )
        
        for regTypeKeyWord in listOfKeyWords:
            
            toothLengthsDict[resKey][regTypeKeyWord] = {}

            for refKey in references2use:
                
                if refKey in finalResultsDict[resKey][0]['registrations'].keys():
                    path2RefImage = mainPath + 'referenceFrames/images/' + refKey + '.png'
                    path2Reflabel = mainPath + 'referenceFrames/labels/' + refKey + '_landmarkCoords.xml'
                    refkeyPointsDic = getRefDict(path2Reflabel, path2RefImage)
                    
                    toothLengthsDict[resKey][regTypeKeyWord][refKey] = {}
                    
                    for i in range(numberOfTeeth):
                        toothNumber = i + 1

                        lengthsForRegType, dicForSmoothing = getAllLengthsForOneRegisterationType(
                            finalResultsDict[resKey],
                            refkeyPointsDic,
                            toothNumber,
                            refKey,
                            regTypeKeyWord,
                            referenceRatiosDict,
                            maxAllowedDistBtwRatiosDict,
                            maxAllowed_detectedAboveRefTT,
                            minAllowed_toothLength,
                            maxAllowed_toothLength,
                            path2saveRejected,
                            verbose = verbose
                        )
                        
                        toothNbKey = 'tooth_'+str(toothNumber)+'_info'
                        
                        toothLengthsDict[resKey][regTypeKeyWord][refKey][toothNbKey] = lengthsForRegType

                        if len(lengthsForRegType['all_tt2ls_dict']['lengths']) == totalNbOfExpectedTimeSteps:
                            totalNbOfSuccessfullCalcs += 1
                        else:
                            if verbose:
                                print('\n\n\n*****************************************************')
                                print('getAllLengthsAndLandmarks did not find the values for all of the  ' +\
                                      str(totalNbOfExpectedTimeSteps) + '  it expected to find values for.\nrefKey: '\
                                      + str(refKey) + '\nregType:  ' + str(regTypeKeyWord) + '\n')
                                print('*************************************************************\n\n\n')
                        
                        if regTypeKeyWord == 'rigid_keypointsForTooth_':
                            toothLengthsDict[resKey]['forSmoothing'][toothNbKey]['all_times'].update(lengthsForRegType['all_times_list']) 

                            toothLengthsDict[resKey]['forSmoothing'][toothNbKey][regTypeKeyWord][refKey] = dicForSmoothing

                
    totalNbOfSuccessfullCalcs = str(totalNbOfSuccessfullCalcs/numberOfTeeth)
    nbOfExpectedCalcs = str(len(listOfKeyWords) * len(references2use))
    print('\n\n****for results set: ' + resKey + '\ngetAllLengthsAndLandmarks successfully calculated all of the  '+ str(totalNbOfExpectedTimeSteps)+ ' expected lengths in   ' + totalNbOfSuccessfullCalcs + '   out of the   ' + nbOfExpectedCalcs + '   total number of referenceFrame-registerationType combinations that we processed. You can turn on verbose to get more info.\n\n')                       
    return toothLengthsDict


def getEmptySmoothedDict(numberOfTeeth, listOfRegTypeKeyWords, listOfReferences2use):
    
    outDict = {}
    
    for i in range(numberOfTeeth):
        toothNbKey = 'tooth_'+str(i+1)+'_info'
        
        outDict[toothNbKey] = {}
        outDict[toothNbKey]['all_times'] = set()
        
        
        for regTypeKeyWord in listOfRegTypeKeyWords:
            outDict[toothNbKey][regTypeKeyWord] = {}
            
            
            for refKey in listOfReferences2use: 
                
                outDict[toothNbKey][regTypeKeyWord][refKey] = {}
                
                
    return outDict