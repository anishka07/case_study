import cv2
import numpy as np 
cap = cv2.VideoCapture('video.mp4')
algo = cv2.createBackgroundSubtractorKNN()
count_line_position = 550
def center_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx, cy
detect=[]
offset = 6
counter = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatdata = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatdata = cv2.morphologyEx(dilatdata, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatdata, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    line = cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)
    for (i, c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>=80) and (h>=80)
        if not validate_counter:
            continue
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, 'Vehicle counter :'+ str(counter), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,67,45), 5)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        for (x, y) in detect:
            if y<(count_line_position + offset) and y>(count_line_position - offset):
                counter += 1
            cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 255), 3)
            detect.remove((x,y))
            print('Vehicle counter:',str(counter))
    cv2.putText(frame, 'Vehicle counter :'+ str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
    cv2.imshow('Detector', dilatdata)
    cv2.imshow('Video Original', frame)
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()
cap.release()