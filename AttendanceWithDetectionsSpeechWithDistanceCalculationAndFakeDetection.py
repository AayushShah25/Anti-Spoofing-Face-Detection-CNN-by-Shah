import cv2
import numpy as np
from numpy import array,float32,float64
import face_recognition
import os
from datetime import date
from datetime import datetime
from DetectMotion import Movement

#import matplotlib.pyplot as plt
from keras.models import load_model
from FaceEmmbedings2 import FaceNet
import mysql.connector as connector

from FakeDetection import isFake
import pyttsx3 
  
engine = pyttsx3.init() 
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
  

#### DATE CHAEKING

mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
cursor = mydb.cursor()

q  = "select datekaka from checkdate limit 1"

cursor.execute(q)
datecheck = cursor.fetchone()

datecheck = datecheck[0]

var = datetime.now()
todaydate = var.strftime("%d%m%y")
print(datecheck)
print(todaydate)


if datecheck == todaydate:
    pass

else:

    #### TRUNCATE
    mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
    cursor = mydb.cursor()

    q = "TRUNCATE attendance";

    cursor.execute(q)
    mydb.commit()

    #### ENTER DATA

    mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
    cursor = mydb.cursor()

    q = "insert into attendance(id,name,hold) select id,name,hold from userse where isin = 1"

    cursor.execute(q)
    mydb.commit()

    #### Update Date

    mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
    cursor = mydb.cursor()

    q = "update checkdate set datekaka = '"+todaydate+"'"
    print(q)
    cursor.execute(q)
    mydb.commit()









#### GET DETAILS

mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
cursor = mydb.cursor()

q = "SELECT Id,name,encodings FROM userse where isin = 1"
cursor.execute(q)
result = cursor.fetchall()

today = date.today()
today = today.strftime("%d-%m-%y")

try:
    os.mkdir('./Unknowns/'+today)
except FileExistsError:
    pass


# =============================================================================
# A Function to check the SAME ELEMENTS in the list
# =============================================================================
def AllSame(l):
    
    if len(l) == 0:
        return False
    
    return l.count(l[0]) == len(l)


# =============================================================================
# Function Ended
# =============================================================================
    
people = {}



for ID,name,emb in result:

    people[str(ID) + " - " + str(name)] = eval(emb)

print(people)

model=load_model('face_recognizer.MODEL')
cascade = cv2.CascadeClassifier('./data/Face.xml')

label='None'
Vectorizer= FaceNet()



guessFaces = []


record = cv2.VideoCapture(0)



_, prev = record.read()
_, frame = record.read()
canvas = np.zeros((480,640,3))

while True: # Just for Check ( Continuous check )
    
    
    Code_Positive = False # A Variable to get entry Directly after matching 25 Faces
    if  not Movement(prev, frame):

        canvas = np.zeros((480,640,3))
        prev = frame



    else:
        
#        plt.subplot(121)
#        plt.imshow(prev)
#        plt.subplot(122)
#        plt.imshow(frame)
#        plt.show()
        
        
        Code_Positive = False
        cv2.destroyWindow('NO DETECTION')
        countSec = 100
        while countSec != 0:
            # Detection ( An Original while True loop )

            faces = []
            coordinates = []
            fakeCount = []

            _, frameA = record.read()
            frameF = cv2.flip(frameA,1)
            
            
            gry_frame = cv2.cvtColor(frameF, cv2.COLOR_BGR2GRAY)

            detect = cascade.detectMultiScale(gry_frame, 1.2, 10)

            for x,y,w,h in detect:

                faces.append( frameF[y:y+h, x:x+w] )
                coordinates.append((x,y,w,h))
                fakeCount.append(isFake(frameF[y:y+h, x:x+w]))
                
                
            if len(detect) != 0:
                
                countSec = 100






            if(faces is not None):
                for i in range(len(faces)):
                    face1 = faces[i]

                    coordinate =coordinates[i]
                    spoofStatus = 'Fake' if fakeCount[i] == 0 else 'Real'
                    
                    FaceIsolated = face1

                    face1 =cv2.resize(face1,(160,160))

                    face1=face1.astype('float')/255.0
                    face1=np.expand_dims(face1,axis=0)

                    feed=Vectorizer.calculate(face1)
                    #feed=np.expand_dims(feed,axis=0)
### NOW THE PART CHANGES...
                    
                    whois = {}
                    for person,encodings in people.items():
                        
                        whois[person] = min(face_recognition.face_distance(encodings,feed))
                        
                    
                    prediction = min(whois.keys(), key=(lambda k: whois[k]))
                    accuracy = whois[prediction]
                    
                    if accuracy < 10:
                        label=prediction
                        
                        guessFaces.append(prediction)
                    
                    else:
                        
                        #print('Found an Unknown Face')
                        now= datetime.now()
                        now= now.strftime("%H%M%S")
                        try:
        
                            cv2.imwrite("./Unknowns/"+today+"/"+str(now)+".png", FaceIsolated)
        
                        except Exception:
                            print('Error in Saving the Unknown face... Please Restart the System.')
                            
                            
        
                        
                        
                        
                        label="Unknown"
                        guessFaces.append("Unknown")
                        
                        
                        
                        
                        
                    if len(guessFaces) == 25:
                        
                        if AllSame(guessFaces):
                            
                            print("Went in 25 : ")
                            if guessFaces[0] != 'Unknown':
                                Code_Positive = True
                            else:
                                Code_Positive = False
                            
                            
                    if len(guessFaces) >= 50 or Code_Positive :
                        
                        print("Inside 50 ")



                        countTimes = {}

                        for I in guessFaces:

                            countTimes[I] = guessFaces.count(I)


                        countTimes = dict( (v,k) for k,v in countTimes.items() )

                        print(countTimes)
                        print(countTimes[max(countTimes)])
                        
                        if countTimes[max(countTimes)] == 'Unknown': 
                            
                            print('Max Unknown FOUND -- Resetting the Variables' )
                            
                            guessFaces = [] 
                            continue
                    
                        if max(countTimes) >= 45 or Code_Positive :
                            ##Provide a Label and Do Stuff

                            print ("Inside More than 45 : Getting Attend !!")

                            #resultRev = dict((v,k) for k,v in people.items())
                            Full_of_Emp = countTimes[max(countTimes)]
                            
                            
                            a_person_ID = Full_of_Emp[0] # Aayush is the biggest, then return the Id of him from the People. 

                            pos = a_person_ID
                            mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
                            cursor = mydb.cursor()
                            check = "select attend from attendance where ID ="+str(pos)

                            cursor.execute(check)
                            resultYN = cursor.fetchall()
                            print("Result came : ", resultYN)

                            if resultYN[0][0] == 1:

                                label_speech=Full_of_Emp.split('-')[1].strip()
                                
                                engine.setProperty('rate',150)
                                engine.say("Welcome"+label_speech+" + you are present today ")
                                engine.runAndWait() 
                                guessFaces = []
                                Code_Positive = False

                            elif (resultYN[0][0] == None or resultYN[0][0] == 0):
                                mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
                                cursor = mydb.cursor()
                                checkforHold = "select hold from attendance where id="+str(pos)

                                cursor.execute(checkforHold)
                                resYes = cursor.fetchall()

                                if resYes[0][0] == 0:
                                    mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")

                                    q = "update attendance set ATTEND = true where id ="+str(pos)

                                    cursor = mydb.cursor()
                                    cursor.execute(q)
                                    mydb.commit()
                                    guessFaces = []
                                    Code_Positive = False

                                else:

                                    label = "Unknown"
                                    now= datetime.now()
                                    now= now.strftime("%H%M%S")
                                    try:

                                        cv2.imwrite("./Unknowns/"+today+"/"+str(now)+".png", FaceIsolated)

                                    except Exception:
                                        print('Error in Saving the Unknown face... Please Restart the System.')

                                    guessFaces = []
                                    Code_Positive = False




                        else:

                            guessFaces = []
                            Code_Positive = False









        
            



                    cv2.putText(frameF,label+" : "+str(len(guessFaces)) +"  "+spoofStatus,(coordinate[0],coordinate[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,200),2)
            
                    cv2.rectangle(frameF,(coordinate[0],coordinate[1]),(coordinate[0] + coordinate[2] , coordinate[1] + coordinate[3]),(123,234,1),3)
        
                
            cv2.imshow('Recognizition',frameF)
        
            countSec -= 1
            prev = frameA
            if cv2.waitKey(1) & 0xff == ord('q') :
                break
    
    
    
        cv2.destroyWindow('Recognizition')
                    
                    

                    

                        
                            










# =============================================================================
# The Ending Part
# =============================================================================

    cv2.imshow("NO DETECTION", canvas)

    
    _, frame = record.read()

    if cv2.waitKey(100) & 0xff == 27:
        break


cv2.destroyAllWindows()
record.release()



# =============================================================================
#  Excel Saving Part
# =============================================================================


month = date.today()
month = month.strftime("%B-%Y")

f_list = os.listdir(r'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/')


if month not in f_list:
    
    os.mkdir(r'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/'+month)


mydb= connector.connect(host = "localhost", user= "root", passwd = "aayush123", database="testdb")
cursor = mydb.cursor()

q = "SELECT ID,name,attend FROM attendance INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/"+month+"/"+ str(today) +".csv' FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\n';"

try:
    
    print('Normal')
    cursor.execute(q)
    print('Normal DONE !')
    
except Exception:
    print('Exeption')    
    os.chmod('C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/'+str(month)+'/',0o777)
    try:
        
        print('Second')
        os.remove(r'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/'+str(month)+'/'+ str(today) +'.csv')
        cursor.execute(q)
        print('Second Done')
    except FileNotFoundError:
        
        cursor.execute(q)
