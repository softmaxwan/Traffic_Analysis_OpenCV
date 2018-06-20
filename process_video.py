import cv2 as cv


def skip_frames(video, end=0):
    idx = 0
    while True: 
        if idx >= end:
            break
        ret, frame = video.read()
        if not ret:
            break
        idx +=1
        

def write_frames(frames, file, fps, img_w, img_h, color_mapping=cv.COLOR_RGB2BGR):
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter()

    opened = out.open(file, fourcc, fps, (img_w, img_h))

    for frame in frames:
        frame = cv.cvtColor(frame, color_mapping)
        out.write(frame)

    out.release()
    
    
def read_frames(video, end=0):
    frames = []
    idx = 0
    while True:
        if idx >= end:
            break
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
    return frames