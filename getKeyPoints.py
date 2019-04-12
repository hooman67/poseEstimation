import os, cv2
import numpy as np
from yoloUtil import WmNetRunner


inputImagePath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Liebherr/frame3Chan/'

outputPredPath = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Liebherr/yolo_preds/'

yoloWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/full_yolo_bb_final.h5'

keypointsWeights = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/hydraulic_pose_resnet_18.pb'


wmNetRunner = WmNetRunner(yoloWeights, keypointsWeights)

for inImgName in os.listdir(inputImagePath):
    inImgPath = inputImagePath + inImgName
    print(("Processing ", inImgPath))

    inImage = cv2.imread(inImgPath)
    #plt.imshow(image)
    #plt.show()

    boxes, outImageYolo = wmNetRunner.yoloPredict(inImage)

    mappedKeypoints, outImageKeypoints = wmNetRunner.keypointsPredict(inImage, boxes)

    cv2.imwrite(outputPredPath + inImgName, np.uint8(outImageKeypoints))         