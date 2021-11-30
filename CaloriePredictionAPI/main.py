from typing import Optional
# import pandas as pd
# from matplotlib import pyplot as plt
# from plotly import graph_objs as go
import cv2
import mediapipe as mp
import time
import requests
import json
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile


# class Item(BaseModel):
#     veg: Optional[str] = 'veg'
#     cuisine: Optional[str] = 'Healthy Food'
#     relevance: Optional[int] = 0
#     Dish: Optional[str] = 'brown rice'


app = FastAPI()


def get_pixel_cm_relation(img):

    #cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    #imgPath = './Slices-apples-table.jpg'
    #img = cv2.imread(imgPath)
    #img = cv2.resize(img,(512,512))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks is not None:

        # print(results.multi_hand_landmarks)

        l4 = None
        l2 = None
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    if id in (4, 3, 2):
                        # print(id, lm)

                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        #print(id, cx, cy)
                        if id == 2 and l2 is None:
                            l2 = cy
                        if id == 4 and l4 is None:
                            l4 = cy
                        # if id == 4:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        #print("Length: ", abs(l4-l2),"Pixels")
        relation = 2.5/(abs(l4-l2))
        #print("relation = ",relation," cm/pixel i.e cms in 1 pixel difference")

        return relation
        # cv2.imshow("Image", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    else:
        print("No hand detected")
        return None


calories = {'egg': 155, 'aloo gobhi': 208, 'aloo tikki': 89, 'apple': 52, 'banana': 89, 'beetroot': 43, 'boiled-egg': 78, 'bread': 53,
            'brown-rice': 219, 'chapati': 270, 'chicken': 239, 'cucumber': 10, 'daal': 75, 'dosa': 165, 'rice': 219, 'white-rice': 220, 'smoothie': 37, 'dhokla': 160}


def get_calories(relation):
    # api-endpoint
    URL = "http://20.115.37.217:5003/files"

    # location given here
    imgPath = "./input.jpg"
    #img = cv2.imread(imgPath)
    files = {'file': open(imgPath, 'rb')}
    r = requests.post(url=URL, files=files)

    # extracting data in json format
    # print(r.text)
    if len(json.loads(r.text)['boxes']) > 0:
        box = json.loads(r.text)['boxes'][0]
        classes = json.loads(r.text)['classes']
        total_calories = 0
        for c in classes:
            total_calories += calories[c]
        #print("Total calories : ",total_calories)
        area = None
        if len(box) > 0 and relation is not None:
            area = box[2]*relation*box[3]*relation
            #print("length: ", box[3]*relation , "cm ,width: ",box[2]*relation," cm")
            #print(area , "cm**2")
            return {'area': area, 'length': box[3]*relation, 'width': box[2]*relation, 'total_calories': total_calories}
        else:
            return {'area': area, 'length': None, 'width': None, 'total_calories': total_calories}
    else:
        return {'area': None, 'length': None, 'width': None, 'total_calories': 'No food located'}


@app.post("/files/")
async def create_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('input.jpg', img)
    relation = get_pixel_cm_relation(img)
    # if relation is not None:
    return get_calories(relation)
    # else:
    return {"calories": '[INFO] No calories detected!'}
