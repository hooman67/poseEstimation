# CONFIGS
mainPath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800/'

# in case you wanna use groundTruth
wmsDir = mainPath +  'groundTruthLabels/' # 'groundTruthLabels/'  'yolo_preds/'

path2saveCurves = mainPath + 'curves/'



##########THESE ARE GLOBAL##############
NUMBER_OF_TEETH = 8  #for cable 8  for hydraulic 6
numberOfTeeth = NUMBER_OF_TEETH

numberOflandmarksIncludingToothTip = 5
verboseAboutRejections = False
TOTAL_EXPECTED_LANDMARKS_ON_LOG = NUMBER_OF_TEETH * numberOflandmarksIncludingToothTip
########################################

#for reject1
minToothBoxDistanceAllowed =  0.0 #0.0  -1000000  if distanceBtwAdjTeeth <= minToothBoxDistanceAllowed: REJECT
lanmark2farFromBox_epsilon =  1000000  #10   1000000   if( miDis > lanmark2farFromBox_epsilon ): REJECT
landmarks2close2eachother_epsilon = 20 #20 -1000000   if diff < landmarks2close2eachother_epsilon: REJECT


#for reject2
'''
# used this for hydraulic. With libre groundTruth too
curvDerivTreshDic = {
    'keypoints_1' : [0.0001, 0.002],
    'keypoints_2' : [0.0001, 0.002],
    'keypoints_3' : [0.0001, 0.002],
    'keypoints_4' : [0.0001, 0.002],
    'keypoints_5' : [-0.0001, 0.002],
}

# used this for hydraulic too. Just to be stricter
curvDerivTreshDic = {
    'keypoints_1' : [0.0001, 0.002],
    'keypoints_2' : [0.0009, 0.002],
    'keypoints_3' : [0.0001, 0.002],
    'keypoints_4' : [0.0001, 0.002],
    'keypoints_5' : [-0.0001, 0.002],
}

# used this for cable
curvDerivTreshDic = {
    'keypoints_1' : [0.0001, 0.0042],
    'keypoints_2' : [0.0001, 0.0042],
    'keypoints_3' : [0.0009, 0.0042],
    'keypoints_4' : [0.0009, 0.0042],
    'keypoints_5' : [0.0009, 0.0042],
}
'''
# used this for cable
curvDerivTreshDic = {
    'keypoints_1' : [0.0001, 0.0042],
    'keypoints_2' : [0.0001, 0.0042],
    'keypoints_3' : [0.0009, 0.0042],
    'keypoints_4' : [0.0009, 0.0042],
    'keypoints_5' : [0.0009, 0.0042],
}


#for reject3
maxDistanceBtwLandmarkAndCurve_replacement = 3 # used 3 for all cable and hydraulic
maxDistanceBtwLandmarkAndCurve_rejection = 100
maxItr_findingShortestDistance2Curve = 10
replaceBadLandmarkWithProjection = True

#for reject4
minNbOfDetectedPointsForTTandLS = NUMBER_OF_TEETH
minNbOfDetectedPointsForOtherLandmarkTypes = 3


#for registration
'''
#for hydraulic test used referenceFrames/images/WMDL_2018.09.11_10.50.26.png'
# first try cable was with referenceFrames/images/WMDL_2018.09.11_06.43.19.png' images/WMDL_2018.09.11_06.21.57.png' was better tho, WMDL_2018.09.11_06.30.39 is middle


# used this for  PH01_2800
references2use = ['WMDL_2018.09.11_06.21.57', 'WMDL_2018.09.11_06.30.39', 'WMDL_2018.09.11_06.43.19']

# used this for  PH02_2800
references2use = ['WMDL_2018.11.05_10.13.29', 'WMDL_2018.11.06_11.26.35', 'WMDL_2018.11.04_12.37.28', 'WMDL_2018.11.06_11.38.56', 'WMDL_2018.11.05_11.10.03']

#used this for PH03-2800
references2use = ['WMDL_2019.02.05_08.16.35', 'WMDL_2019.02.05_08.41.56']

#used this for PH03-4100:
couldn't find any good images so used the same ones as in above (PH03-2800)

# used this for wmdlLogs_Pinto (hydraulic):
references2use = ['1_20161116-074000_0001n0_9767', '1_20161116-152500_0001n0_783']

#used this for wmdlLogs_aitik (hydraulic)
references2use = ['WMDL_2019.02.27_10.03.11','WMDL_2017.11.27_23.58.12', 'WMDL_2017.11.28_00.33.32', 'WMDL_2017.11.27_23.52.22']
'''
# used this for  PH01_2800
references2use = ['WMDL_2018.09.11_06.21.57']
#1st ref is used below for cleaning up final lengths (valid) 


#For cleaning up final lengths (valid) 
#for cable PH01_2800 using ref WMDL_2018.09.11_06.21.57_labeled

#moved to main
#refKey = references2use[0]
#referenceRatiosDict = getReferenceRatios(refKey)


'''
#Best for PH01_2800 cable
refKey = 'WMDL_2018.09.11_06.21.57'
maxAllowedDistBtwRatiosDict = {
        'tooth_1':{
            'tt2ls_over_le2bk': 0.5,
            'tt2ls_over_le2cl': 0.5,
            'tt2ls_over_cl2bk': 0.5,
            'tt2ls_over_ls2le': 0.5, 
            'tt2ls_over_ls2cl': 0.5, 
            'tt2ls_over_ls2bk': 0.5,
        },
        'tooth_2':{ # at 1, you see no effects.
            'tt2ls_over_le2bk': 0.5, 
            'tt2ls_over_le2cl': 0.5, 
            'tt2ls_over_cl2bk': 0.5,
            'tt2ls_over_ls2le': 0.5, 
            'tt2ls_over_ls2cl': 0.5,
            'tt2ls_over_ls2bk': 0.5,
        },
        'tooth_3':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1, # at 0.5 with all rest being 1, only this one has an effect
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_4':{
            'tt2ls_over_le2bk': 1.1,
            'tt2ls_over_le2cl': 1.1,
            'tt2ls_over_cl2bk': 1.1,
            'tt2ls_over_ls2le': 1.1,
            'tt2ls_over_ls2cl': 1.1,
            'tt2ls_over_ls2bk': 1.1,
        },
        'tooth_5':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_6':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_7':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_8':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        }
    }


#for hydraulic. But it can be better. This just allows everything through
maxAllowedDistBtwRatiosDict = {
        'tooth_1':{
            'tt2ls_over_le2bk': 10,
            'tt2ls_over_le2cl': 10,
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10, 
            'tt2ls_over_ls2cl': 10, 
            'tt2ls_over_ls2bk': 10,
        },
        'tooth_2':{ # at 1, you see no effects.
            'tt2ls_over_le2bk': 10, 
            'tt2ls_over_le2cl': 10, 
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10, 
            'tt2ls_over_ls2cl': 10,
            'tt2ls_over_ls2bk': 10,
        },
        'tooth_3':{
            'tt2ls_over_le2bk': 10,
            'tt2ls_over_le2cl': 10,
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10, # at 0.5 with all rest being 1, only this one has an effect
            'tt2ls_over_ls2cl': 10,
            'tt2ls_over_ls2bk': 10,
        },
        'tooth_4':{
            'tt2ls_over_le2bk': 10,
            'tt2ls_over_le2cl': 10,
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10,
            'tt2ls_over_ls2cl': 10,
            'tt2ls_over_ls2bk': 10,
        },
        'tooth_5':{
            'tt2ls_over_le2bk': 10,
            'tt2ls_over_le2cl': 10,
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10,
            'tt2ls_over_ls2cl': 10,
            'tt2ls_over_ls2bk': 10,
        },
        'tooth_6':{
            'tt2ls_over_le2bk': 10,
            'tt2ls_over_le2cl': 10,
            'tt2ls_over_cl2bk': 10,
            'tt2ls_over_ls2le': 10,
            'tt2ls_over_ls2cl': 10,
            'tt2ls_over_ls2bk': 10,
        },
    }
'''
#Best for PH01_2800 cable
refKey = 'WMDL_2018.09.11_06.21.57'
maxAllowedDistBtwRatiosDict = {
        'tooth_1':{
            'tt2ls_over_le2bk': 0.5,
            'tt2ls_over_le2cl': 0.5,
            'tt2ls_over_cl2bk': 0.5,
            'tt2ls_over_ls2le': 0.5, 
            'tt2ls_over_ls2cl': 0.5, 
            'tt2ls_over_ls2bk': 0.5,
        },
        'tooth_2':{ # at 1, you see no effects.
            'tt2ls_over_le2bk': 0.5, 
            'tt2ls_over_le2cl': 0.5, 
            'tt2ls_over_cl2bk': 0.5,
            'tt2ls_over_ls2le': 0.5, 
            'tt2ls_over_ls2cl': 0.5,
            'tt2ls_over_ls2bk': 0.5,
        },
        'tooth_3':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1, # at 0.5 with all rest being 1, only this one has an effect
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_4':{
            'tt2ls_over_le2bk': 1.1,
            'tt2ls_over_le2cl': 1.1,
            'tt2ls_over_cl2bk': 1.1,
            'tt2ls_over_ls2le': 1.1,
            'tt2ls_over_ls2cl': 1.1,
            'tt2ls_over_ls2bk': 1.1,
        },
        'tooth_5':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_6':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_7':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        },
        'tooth_8':{
            'tt2ls_over_le2bk': 1,
            'tt2ls_over_le2cl': 1,
            'tt2ls_over_cl2bk': 1,
            'tt2ls_over_ls2le': 1,
            'tt2ls_over_ls2cl': 1,
            'tt2ls_over_ls2bk': 1,
        }
    }


maxAllowed_detectedAboveRefTT = 1


#not used
minAllowed_toothLength = 0#10
maxAllowed_toothLength = 300#30

#for calculating secondary confidence.
weightOfToothConfidence = 1
weightOfLogConfidence = 1.2 #we decided on 1.1 but I used 1.2 for hydraulic and traj investigations for some reason


#**** only used by getRegisteredPointsV3.rejects logs where regError using LS is smaller than using all landmarks.
regEr_WithLs_lessThan_withAll_multiple = 1000 # used 1 for hydraulic. For cable 1000 to disable this.