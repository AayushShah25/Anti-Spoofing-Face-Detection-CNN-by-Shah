import cv2
from keras.models import load_model
import numpy as np

model = load_model('./liveness.model')


def isFake(roi):
    
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
        rgb = cv2.resize(rgb , (64,64))
    
        rgb = np.reshape(rgb,(1,64,64,3))
        rgb = rgb/float(255)
        
        
    
        fakeornot = model.predict(rgb)
        print("\n")
        print("Fake : " , fakeornot[0][0] , "  Real : " , fakeornot[0][1])
        if fakeornot[0][0] >  0.65 :
            return 0
        else:
            return 1
        
