import cv2
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

cap1 = cv2.VideoCapture('final_video.mp4')
cap2 = cv2.VideoCapture('final_video_2.mp4')
cap3 = cv2.VideoCapture('metrics_video.mp4')

video = cv2.VideoWriter('demo.mp4', 0x00000021, 1, (711, 422))

while cap1.isOpened() or cap2.isOpened():

    okay1, frame1 = cap1.read()
    okay2, frame2 = cap2.read()
    okay3, frame3 = cap3.read()

    both = okay1 and okay2

    if okay1:
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        #if both == False:
            #cv2.imshow('fake', hsv1)

    if okay2:
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        #if both == False:
            #cv2.imshow('real', hsv2)

    if both and okay3:
        scale_percent = 33  # percent of original size
        width1 = int(frame2.shape[1] * scale_percent / 100)
        height1 = int(frame2.shape[0] * scale_percent / 100)
        dim1 = (width1, height1)

        # resize image
        resized1 = cv2.resize(frame1, dim1, interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(frame2, dim1, interpolation=cv2.INTER_AREA)
        resized3 = cv2.resize(frame3, dim1, interpolation=cv2.INTER_AREA)
        cv2.imshow('double', np.hstack((resized1, resized2, resized3)))
        video.write(np.hstack((resized1, resized2, resized3)))

    if not okay1 or not okay2:
        print('Cant read the video , Exit!')
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)

cap1.release()
cap2.release()
video.release()

image_clip = VideoFileClip('demo.mp4')
image_clip.write_videofile('demo.mp4')

cv2.destroyAllWindows()