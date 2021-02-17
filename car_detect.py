####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

import cv2

import os 
import requests
import json
import re
import time

import numpy as np

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for carplate recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")
# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Carplate
cascadePath = "car_plate_cascade.xml"

# Create classifier from prebuilt model
PlatesCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
# https://webnautes.tistory.com/1390
cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Use OCR solution through WEB-API or Embeded(offline) OCR as alternatives for the better performanace
def ocr_space_file(filename, overlay=False, api_key='helloworld', language='kor'):

    payload = {'isOverlayRequired': overlay,
               'apikey': 'c7635b08db88957',
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()

#def ocr_space_url(url, overlay=False, api_key='helloworld', language='kor'):
#
#    payload = {'url': url,
#               'isOverlayRequired': overlay,
#               'apikey': api_key,
#               'language': language,
#               }
#    r = requests.post('https://api.ocr.space/parse/image',
#                      data=payload,
#                      )
#    return r.content.decode()

while True:

    try :
        # Read the video frame    
        ret, im = cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #gray = cv2.medianBlur(gray, 10)

        # Get all plate from the video frame
        Plates = PlatesCascade.detectMultiScale(gray, 1.2,5)
        
        # For each plate in plates
        for(x,y,w,h) in Plates:

            # Create rectangle around the plate
            cv2.rectangle(im, (x-150,y-20), (x+w+150,y+h+20), (0,255,0), 4)

            cv2.imwrite("dataset/picture_of_plate.jpg", gray[y:y+h,x-120:x+w+120])

            # Recognize the face belongs to which ID
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if(os.path.exists("dataset/picture_of_plate.jpg")):
                test_file = ocr_space_file(filename="dataset/picture_of_plate.jpg", language='kor')
                #test_url = ocr_space_url(url='http://aaa.bbb.ccc/ddd.jpg')
                parseResult = json.loads(test_file)
                for item in parseResult['ParsedResults']:
                    full_plate_no_origin = item['ParsedText']
                    full_plate_no = item['ParsedText']
                    #정규화 체크 로직 추가 ex) [334 가 2440] or [23 가 2345]
                    old_version = re.search('[0-9][0-9]\s*\w\s*[0-9][0-9][0-9][0-9]', full_plate_no)
                    new_version = re.search('[0-9][0-9][0-9]\s*\w\s*[0-9][0-9][0-9][0-9]', full_plate_no)
                    if(old_version != None):
                        full_plate_no = old_version
                    
                    elif(new_version != None):
                        full_plate_no = new_version

                #os.remove('dataset/picture_of_plate.jpg')
            
            # Use examples:
            #Id = "{0:.2f}%".format(round(100 - confidence, 2))
            # Put text describe who is in the picture
            cv2.rectangle(im, (x-22,y-90), (x+w+20, y-20), (0,255,0), -1)
            #cv2.putText(im, rows[0][1] + Id, (x,y-40), font, 1, (255,255,255), 3)
            #cv2.putText(im, rows[0][1], (x,y-40), font, 1, (255,255,255), 3)
            #if(full_plate_no is not None):
            cv2.putText(im, full_plate_no_origin, (x,y-40), font, 1, (255,255,255), 3)
            print(full_plate_no_origin)
        
        # Display the video frame with the bounded rectangle
        cv2.imshow('im',im) 

        # If 'q' is pressed, close program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    except Exception as ex:
        print(ex)

# Camera Release
cam.release()

# Close all windows
cv2.destroyAllWindows()
