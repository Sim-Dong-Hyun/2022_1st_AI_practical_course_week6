# ! conda install -c conda-forge opencv
import cv2
import time
import datetime

fourcc = cv2.VideoWriter_fourcc(*'DIVX')

cap = cv2.VideoCapture(1)
width = int(cap.get(3)) ; height = int(cap.get(4))
fps = 30.0
now = datetime.datetime.now()
vid_file_name = "{}.mp4".format(now.strftime("%m%d_%H%M%S"))
vid_file = cv2.VideoWriter(vid_file_name, fourcc, fps, (width, height))

strt_time = time.time()

print('recording')
while(time.time() - strt_time < 15):
    ret, frame = cap.read()
    cv2.imshow('video',frame)
    vid_file.write(frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
vid_file.release()
