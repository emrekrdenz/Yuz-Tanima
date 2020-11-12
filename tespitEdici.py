import cv2
from cv2 import cv2

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml')
cascadePath='face.xml'
faceCascade=cv2.CascadeClassifier(cascadePath)
path="yuzverileri"
cam=cv2.VideoCapture(0)

while True:
    _,resim=cam.read()
    griton=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(griton,scaleFactor=1.2,minNeighbors=5)
    for(x,y,w,h) in faces:
        tahminEdilenKisi,conf=recognizer.predict(griton[y:y+h,x:x+w]) ## kıyasla
        cv2.rectangle(resim,(x,y),(x+w,y+h),(225,0,0),2)
       
        if(tahminEdilenKisi == 1):
             tahminEdilenKisi= 'Emre Karadeniz'
        elif (tahminEdilenKisi == 2):
            tahminEdilenKisi = 'Sena Gedik'
        elif (tahminEdilenKisi == 3):
            tahminEdilenKisi = 'Cihan Demir'  
        elif (tahminEdilenKisi == 5):
            tahminEdilenKisi = 'Ahmet Yuce' 
        elif (tahminEdilenKisi == 4):
            tahminEdilenKisi = 'Ahmet Ugur' 
        else:
            tahminEdilenKisi= "Sen Kimsin ya"

        fontFace=cv2.FONT_HERSHEY_SIMPLEX
        fontThickness=2
        fontScale=1
        fontColor=(0,0,255)
        cv2.putText(resim,str(tahminEdilenKisi),(x,y+h),fontFace,fontScale,fontColor,fontThickness)
        cv2.imshow("Canim Arkadaslarim",resim)
    if cv2.waitKey(10) &0xFF==ord('q'):
        break