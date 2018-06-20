import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def train_bg_substractor(frames):
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=len(frames), detectShadows=True)

    for frame in frames:
        bg_subtractor.apply(frame, None, 0.001)

    return bg_subtractor


def refine_fgmask(fg_mask):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    closing = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv.dilate(opening, kernel, iterations=2)

    # thresholding
    dilation[dilation < 240] = 0

    fg_mask = dilation
    
    return fg_mask


def count_traffic(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame, None, 0.001)
    fg_mask = refine_fgmask(fg_mask)

    # finding external contours
    im, contours, hierarchy = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    matches = countour_filter(contours)
    return matches

    
def run_traffic_count(vcap, bg_subtractor, end=None, output_video=True):
    font = cv.FONT_HERSHEY_SIMPLEX
    out_frames = []
    frame_matches = []
    
    idx = 0
    while True:
        if end != None and idx >= end:
            break
            
        ret, frame = vcap.read()
        if not ret:
            break
            
        matches = count_traffic(frame, bg_subtractor)
        if output_video:
            for match in matches:
                pts = contour_to_pts(match)
                cv.polylines(frame, [pts], True, (0,255,0), thickness = 3)
                frame = cv.putText(frame,
                                   "COUNT: {}".format(len(matches)),
                                   (10, 50),
                                   font, 2,
                                   (100, 100, 200), 3,
                                   cv.LINE_AA)
                out_frames.append(frame)
        
        frame_matches.append(matches)
        idx += 1
    
    if output_video:
        ret = out_frames, frame_matches
    else:
        ret = frame_matches
        
    return ret


def countour_filter(contours):
    min_contour_width = 30
    min_contour_height = 30
    
    matches = []
    # filtering by width, height
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (
            h >= min_contour_height)
        if not contour_valid:
            continue
        matches.append((x, y, w, h))
    return matches


def contour_to_pts(contour):
    x, y, w, h = contour
    pt1 = [x, y]
    pt2 = [x + w, y]
    pt3 = [x + w, y + h]
    pt4 = [x, y + h]
    return np.array([pt1, pt2, pt3, pt4])


def plot_test_count(frame, bg_subtractor):
    fg_mask = bg_subtractor.apply(frame, None, 0.001)
    fg_mask = refine_fgmask(fg_mask)
    im, contours, hierarchy = cv.findContours(fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    matches = countour_filter(contours)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    img = frame.copy()
    for match in matches:
        pts = contour_to_pts(match)
        cv.polylines(img, [pts], True, (0,255,0), thickness = 3)
        img = cv.putText(img,
                         "COUNT: {}".format(len(matches)),
                         (10, 50),
                         font, 2,
                         (100, 100, 200), 3,
                         cv.LINE_AA)
        
    fig = plt.figure(figsize=(20,12))
    plt.subplot(221),plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)),plt.title('Original')
    plt.subplot(222),plt.imshow(fg_mask),plt.title('Foreground Mask')
    plt.subplot(223),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Contours')
    plt.show()
    
    
    