import os
import cv2
import numpy as np

# https://www.geeksforgeeks.org/converting-image-ascii-image-python/
'''
for f in $(ls | sort -n -t _ -k 1); do
  cat $f
  sleep 0.04
done
'''

# 67 levels of gray
gscale67 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~i!lI;:,\"^`. "
len_gscale67 = len(gscale67) - 1
# 10 levels of gray
gscale10 = "@%#*+=-:. "
len_gscale10 = len(gscale10) - 1

WIDTH = 320
HEIGHT = 96
SCALE = -1
NEW_WIDTH, NEW_HEIGTH = 0, 0
FRAME_NUMBER = 0

# TODO
# get scale, width, height, framerate, grayscale palette from args
# implement drop frames to produce less ascii text from video

path_to_save = os.path.join(os.getcwd(), 'result-ascii-360x96-edges')
path_to_video = os.path.join(os.getcwd(), 'test_720P_1500K_217812511.mp4')
video = cv2.VideoCapture(path_to_video)

if video.isOpened():
    ret, frame = video.read()
    h, w, _ = frame.shape
    SCALE = WIDTH / w
    NEW_WIDTH = int(w * SCALE)
    NEW_HEIGTH = int(h * SCALE / 2)
    print('num of frames:', video.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.destroyAllWindows()


while video.isOpened():
    ret, frame = video.read()
    # Remove noise by blurring with a Gaussian filter
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    gray = cv2.resize(gray, (NEW_WIDTH, NEW_HEIGTH))
    edges = ~cv2.Canny(gray, 75, 120)
    # laplacian = cv2.Laplacian(gray, ddepth=cv2.CV_16S, ksize=3)
    cv2.imshow('edges', edges)
    ascii_frame = ""
    ascii_list = []
    ascii_list.append('\n')
    for row in range(NEW_HEIGTH):
        for col in range(NEW_WIDTH):
            gsval = gscale67[int((edges[row, col]*len_gscale67)/255)]
            ascii_list.append(gsval)
        ascii_list.append('\n')
    ascii_frame = ''.join(ascii_list)
    with open(path_to_save + '/' + 'frame_' + (8 - len(str(FRAME_NUMBER)))*'0' + str(FRAME_NUMBER), 'wt') as f:
        f.write(ascii_frame)
    FRAME_NUMBER += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()