import os, cv2, json
import numpy as np
import matplotlib.pyplot as plt

'''
shovelType = 'cable'

inputImagePath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800/Frame/'

outputPredPath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800/yolo_preds/'

keypointsWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/cable_pose_resnet_18.pb'

useSSD_insteadOfYolo = False

yoloWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/cable_full_yolo_bb_cable_final.h5'

ssdPredsDir = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Liebherr/ssd_preds/'
'''


shovelType = 'hydraulic'

inputImagePath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik_Komatsu_SH1142_PC5500_2019-02-26_to_2019-03-10/usingNewPoseNet/Frame/'

outputPredPath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik_Komatsu_SH1142_PC5500_2019-02-26_to_2019-03-10/usingNewPoseNet/yolo_preds/'

#keypointsWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/hydraulic_pose_resnet_18.pb'
keypointsWeights = '/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/poseNet/hydraulic/try4(byAnuar)-OndataTry2_withoutAug_sigmaForHeatmaps1InsteadOf2/final_state.pb'



useSSD_insteadOfYolo = False

yoloWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/hydraulic_full_yolo_bb_final.h5'

ssdPredsDir = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Liebherr/ssd_preds/'



if shovelType == 'hydraulic':
    from yoloUtil_hydraulics import WmNetRunner
elif shovelType == 'cable':
    from yoloUtil_cables import WmNetRunner
else:
    print('No shovel type specified')



wmNetRunner = WmNetRunner(keypointsWeights, yoloWeights, ssdPredsDir)

for inImgName in os.listdir(inputImagePath):
    inImgPath = inputImagePath + inImgName
    print(("Processing ", inImgPath))

    inImage = cv2.imread(inImgPath)


    if useSSD_insteadOfYolo:
        boxes, outImageBackbone = wmNetRunner.getBoxesFromSsdPreds(inImgName.replace('.png','.json'), inImage)
    else:
        boxes, outImageBackbone = wmNetRunner.yoloPredict(inImage)


    #plt.imshow(outImageBackbone)
    #plt.show()
    #break


    mappedKeypoints, outImageKeypoints = wmNetRunner.keypointsPredict(inImage, boxes)

    teethAr = []
    bucketAr = [] 
    for bx in boxes:
        jsonedBox = bx.get_json()
        if jsonedBox[4] == "Tooth":
            teethAr.append(jsonedBox)
        else:
            bucketAr.append(jsonedBox)

    json2save = [mappedKeypoints, teethAr, bucketAr]


    '''
    print('\n\nsavedJson')
    print(json2save)
    print('\n')
    plt.imshow(outImageKeypoints)
    plt.show()
    break
    '''

    cv2.imwrite(outputPredPath + inImgName, np.uint8(outImageKeypoints))

    with open(outputPredPath + inImgName.replace('.png', '.json'), 'w') as fjson:
        json.dump(json2save, fjson) 