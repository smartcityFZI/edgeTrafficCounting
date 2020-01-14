# Function to save videos
import cv2

cap = cv2.VideoCapture('fz1.mp4')
if (cap.isOpened() == False):
  print("Unable to read camera feed")
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0,(int(cap.get(3)), int(cap.get(4))))

count = 0
while (True):
    ret, frame = cap.read()

    if ret == True:
        # Write the frame into the file 'output.avi'
        out.write(frame)
        print(count)
        count = count+1
    else:
        break


cap.release()
out.release()
