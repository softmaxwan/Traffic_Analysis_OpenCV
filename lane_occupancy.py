# coding: utf-8
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# In[ ]:


def calc_occupancy(frame, area_pts):
    base_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    area_mask = cv.fillPoly(np.zeros(base_frame.shape, base_frame.dtype), 
                            [area_pts], 
                            (255, 255, 255)
                           )[:, :, 0]

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # this used for noise reduction at night time
    frameGray = cv.cvtColor(base_frame, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(frameGray)

    # getting edges with Canny filter
    edges = cv.Canny(frameGray, 50, 70)
    # invert them to get white background
    edges = ~edges

    # blur with additional use of bilateralFilter to remove color noise
    blur = cv.bilateralFilter(cv.blur(edges,(21,21), 100),9,200,200)

    # threshold with ROI overlapping
    _, threshold = cv.threshold(blur,230, 255,cv.THRESH_BINARY)
    t = cv.bitwise_and(threshold, threshold, mask = area_mask)

    # counting capacity area
    free = np.count_nonzero(t)
    capacity = 1 - float(free)/np.count_nonzero(area_mask)

    # creating plot for debugging and visualization
    img = np.zeros(base_frame.shape, base_frame.dtype)
    img[:, :] = (0, 50, 0)
    mask = cv.bitwise_and(img, img, mask=area_mask)
    cv.addWeighted(mask, 1, base_frame, 1, 0, base_frame)

    return (base_frame, edges, blur, t, capacity)


# In[ ]:

def detect_occupancy(vcap, areas, end=None, output_video=True):
    font = cv.FONT_HERSHEY_SIMPLEX
    out_frames = []
    frame_occupancies = []
    
    idx = 0
    while True:
        if end != None and idx >= end:
            break
            
        ret, frame = vcap.read()
        if not ret:
            break
            
        frame, edges, blur, t, capacity = calc_occupancy(frame, areas)
        
        if output_video:
            frame = cv.putText(frame,
                               "CAPACITY: {:.2%}".format(capacity),
                               (10, 50), 
                               font, 2, 
                               (255, 100, 200), 3, 
                               cv.LINE_AA)
            out_frames.append(frame)
        
        frame_occupancies.append(capacity)
        idx += 1
    
    if output_video:
        ret = out_frames, frame_occupancies
    else:
        ret = frame_occupancies
        
    return ret


def plot_test_occupancy(frame, area_pts):
    base_frame, edges, blur, t, capacity = calc_occupancy(frame, area_pts)

    fig = plt.figure(figsize=(16,9))
    fig.suptitle("Capacity: {}%".format(capacity*100), fontsize=16)
    plt.subplot(221),plt.imshow(base_frame),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(edges),plt.title('Cany edges')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223),plt.imshow(blur),plt.title('Blur')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224),plt.imshow(t),plt.title('Threshold with ROI mask')
    plt.xticks([]), plt.yticks([])

    plt.show()

