import sys, time, os, cv2

# This is needed if the notebook is stored in the object_detection folder.
sys.path.append(".")
sys.path.append("/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800")


from helpers import *
from configs import *


import random
 




 

#plt.axis([0, 250, 10, 50]) # for cable GT

#ax = plt.axes()
#ax.grid()

#plt.xlabel('Hours')
#plt.xlabel('Pixels')



#fig = plt.figure(figsize=(30,10))


#ax1 = fig.add_subplot(111)


xdata = []
ydata = []

plt.show()
fig, axs = plt.subplots(3, sharex=True)

axs[0].set_xlim(0, 250)
axs[0].set_ylim(-50, +50)

axs[0].set_ylabel('Pixels')
lines, = axs[0].plot(xdata, xdata)

axs[1].set_xlim(0, 250)
axs[1].set_ylim(-50, +50)
axs[1].set_xlabel('Hours')

points, = axs[1].plot(xdata, xdata)



images = axs[2]


'''
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
lines, = axes.plot([], [])
'''

path = '/media/hooman/961293e3-04a5-40c5-afc0-2b205d0a7067/WM_PROJECT/algorithmDev/wmAlgo_usingWearLandmarsk_optical_hydraulics/try1/wmdlLogs_Sishen_cable/PH01_2800/finalVis/'
imList = os.listdir(path)

ysample = range(-50, 50, 1)
 
for i in range(100):
    xdata.append(i)
    ydata.append(ysample[i])
    lines.set_xdata(xdata)
    lines.set_ydata(ydata)

    points.set_xdata(xdata)
    points.set_ydata(ydata)

    image = cv2.imread(path + imList[i])
    images.imshow(image)

    plt.draw()
    plt.pause(1e-17)
