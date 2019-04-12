import os, cv2, json
import numpy as np
import matplotlib.pyplot as plt
from yoloUtil import WmNetRunner


inputImagePath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik/frame3Chan/'

outputPredPath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_aitik/yolo_preds/'

yoloWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/full_yolo_bb_final.h5'

keypointsWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/poseEstimation/hydraulic_pose_resnet_18.pb'


wmNetRunner = WmNetRunner(yoloWeights, keypointsWeights)

for inImgName in os.listdir(inputImagePath):
    inImgPath = inputImagePath + inImgName
    print(("Processing ", inImgPath))

    inImage = cv2.imread(inImgPath)

    boxes, outImageYolo = wmNetRunner.yoloPredict(inImage)

    mappedKeypoints, outImageKeypoints = wmNetRunner.keypointsPredict(inImage, boxes)

    boxesAr = []
    for bx in boxes:
        boxesAr.append(bx.get_json())

    json2save = [mappedKeypoints, boxesAr]
    

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