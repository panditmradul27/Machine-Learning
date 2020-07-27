import cv2
import numpy as np

Red_cupss = cv2.CascadeClassifier('Red_cups.xml')

carss = cv2.CascadeClassifier('cars.xml')

Bus_fronts = cv2.CascadeClassifier('Bus_front.xml')

WallClocks = cv2.CascadeClassifier('WallClock.xml')

#noses = cv2.CascadeClassifier('nose.xml')

bananas = cv2.CascadeClassifier('banana.xml')

Football_cascades = cv2.CascadeClassifier('Football_cascade.xml')

haarcascade_fullbodys = cv2.CascadeClassifier('haarcascade_fullbody.xml')

eyes = cv2.CascadeClassifier('eye.xml')

stop_signals = cv2.CascadeClassifier('stop_signal.xml')






if Red_cupss.empty():
    raise IOError('Unable to Red_cups.xml file')

if carss.empty():
    raise IOError('Unable to load cars.xml file')

if Bus_fronts.empty():
    raise IOError('Unable to load Bus_front.xml file')

if WallClocks.empty():
    raise IOError('Unable to load WallClock.xml file')

#if noses.empty():
   # raise IOError('Unable to load nose.xml file')

if bananas.empty():
    raise IOError('Unable to load banana.xml file')

if Football_cascades.empty():
    raise IOError('Unable to load Football_cascade.xml file')

if haarcascade_fullbodys.empty():
    raise IOError('Unable to load haarcascade_fullbody.xml file')

if eyes.empty():
    raise IOError('Unable to load eye.xml file')

if stop_signals.empty():
    raise IOError('Unable to load stop_signal.xml file')



capture = cv2.VideoCapture(0)

while True:
    ret, capturing = capture.read()
    x,y,w,h=0,0,0,0

    resize_frame = cv2.resize(capturing, None, fx=1, fy=1,interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)





    Red_cups = Red_cupss.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in Red_cups:
        #REd
        cv2.rectangle(resize_frame, (x,y), (x+w,y+h), (0,0,255), 10)
        text = "Cups"
        cv2.putText(resize_frame, text, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    cars = carss.detectMultiScale(gray, 1.3, 5)

    for (cars_x, cars_y, cars_w, cars_h) in cars:
        #Blue
        cv2.rectangle(resize_frame,(cars_x,cars_y),(cars_x + cars_w, cars_y + cars_h),(255,0,0),5)
        text = "Car"
        cv2.putText(resize_frame, text, (cars_x, cars_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)




    Bus_front = Bus_fronts.detectMultiScale(gray, 1.3, 5)

    for (Bus_front_x, Bus_front_y, Bus_front_w, Bus_front_h) in Bus_front:
        cv2.rectangle(resize_frame, (Bus_front_x, Bus_front_y), (Bus_front_x + Bus_front_w, Bus_front_y + Bus_front_h), (0,255,0), 5)
        text = "Bus"
        cv2.putText(resize_frame, text, (Bus_front_x, Bus_front_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)





    WallClock = WallClocks.detectMultiScale(gray, 1.3, 5)

    for (WallClock_x, WallClock_y, WallClock_w, WallClock_h) in WallClock:
        # orange
        cv2.rectangle(resize_frame, (WallClock_x, WallClock_y), (WallClock_x + WallClock_w, WallClock_y + WallClock_h), (0,165,255), 5)
        text = "Clock"
        cv2.putText(resize_frame, text, (WallClock_x, WallClock_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



   # nose =noses.detectMultiScale(gray, 1.3, 5)

    #for (nose_x, nose_y, nose_w, nose_h) in nose:
        #Yellow
     #   cv2.rectangle(resize_frame, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0,255,255), 5)
      #  text = "nose"

       # cv2.putText(resize_frame, text, (nose_x, nose_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

        


    banana = bananas.detectMultiScale(gray, 1.3, 5)

    for (banana_x, banana_y, banana_w, banana_h) in banana:
        #pink
        cv2.rectangle(resize_frame, (banana_x, banana_y), (banana_x + banana_w, banana_y + banana_h), (147,20,255), 5)
        text = "banana"
        cv2.putText(resize_frame, text, (banana_x, banana_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    Football_cascade = Football_cascades.detectMultiScale(gray, 1.3, 5)

    for (Football_cascade_x, Football_cascade_y, Football_cascade_w, Football_cascade_h) in Football_cascade:
        #gray
        cv2.rectangle(resize_frame, (Football_cascade_x, Football_cascade_y), (Football_cascade_x + Football_cascade_w, Football_cascade_y + Football_cascade_h), (128,128,128), 5)
        text = "Clock"
        cv2.putText(resize_frame, text, (Football_cascade_x, Football_cascade_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    haarcascade_fullbody = haarcascade_fullbodys.detectMultiScale(gray, 1.3, 5)

    for (haarcascade_fullbody_x, haarcascade_fullbody_y, haarcascade_fullbody_w, haarcascade_fullbody_h) in haarcascade_fullbody:
        #brown
        cv2.rectangle(resize_frame, (haarcascade_fullbody_x, haarcascade_fullbody_y), (haarcascade_fullbody_x + haarcascade_fullbody_w, haarcascade_fullbody_y + haarcascade_fullbody_h), (19,69,139), 5)
        text = "Body"
        cv2.putText(resize_frame, text, (haarcascade_fullbody_x, haarcascade_fullbody_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)



    eye = eyes.detectMultiScale(gray, 1.3, 5)

    for (eye_x,eye_y, eye_w, eye_h) in eye:
        #marun
        cv2.rectangle(resize_frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0,0,128), 5)
        text = "eye"
        cv2.putText(resize_frame, text, (eye_x, eye_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)


    stop_signal = stop_signals.detectMultiScale(gray, 1.3, 5)

    for (stop_signal_x, stop_signal_y, stop_signal_w, stop_signal_h) in stop_signal:
        #Cyan
        cv2.rectangle(resize_frame, (stop_signal_x, stop_signal_y), (stop_signal_x + stop_signal_w,stop_signal_y + stop_signal_h), (125,125,0), 5)
        text = "stop-signal"
        cv2.putText(resize_frame, text, (stop_signal_x, stop_signal_h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)




    cv2.imshow("Real-time Detection", resize_frame)







    c = cv2.waitKey(1)
    if c == 27:
        break
capture.release()
cv2.destroyAllWindows()
