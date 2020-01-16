import cv2
#from modelSave import load_model
from keras.models import load_model
import numpy as np


record = cv2.VideoCapture(0)

#model = load_model(r'c:/users/gigabyte/desktop/3DMAD-ftweights18.h5',96,96)
model = load_model('./liveness.model')
detkaka = cv2.CascadeClassifier('./face.xml')

#img = cv2.imread(r'C:/users/gigabyte/desktop/15.png')

while True:
    
    _, frame = record.read()
    
    gry = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    
    detections = detkaka.detectMultiScale(gry, 1.2, 5)
    
    
    
    
    
    
    
    
    #print(int(model.predict_classes(rgb)))
    if len(detections) > 0 : 
        
        x,y,w,h = detections[0]
        rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    
        rgb = cv2.resize(rgb , (64,64))
    
        rgb = np.reshape(rgb,(1,64,64,3))
        rgb = rgb/float(255)
        
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, str(int(model.predict_classes(rgb))), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        
    cv2.imshow("See", frame)
    
    
    if cv2.waitKey(1) & 0xff == 27:
        break
    
cv2.destroyAllWindows()
record.release()