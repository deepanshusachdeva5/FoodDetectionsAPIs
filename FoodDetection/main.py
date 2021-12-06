from typing import Optional
import os
import glob
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import random
import cv2
import numpy as npa
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from fastapi import FastAPI
from fastapi.responses import FileResponse
import json
setup_logger()
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from detectron2.utils.visualizer import ColorMode
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import warnings
import glob


# class Item(BaseModel):
#     veg: Optional[str] = 'veg'
#     cuisine: Optional[str] = 'Healthy Food'
#     relevance: Optional[int] = 0
#     Dish: Optional[str] = 'brown rice'

#set configuration and load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.MODEL.DEVICE ='cpu'
cfg.OUTPUT_DIR = './Model2/'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
predictor = DefaultPredictor(cfg)


app = FastAPI()

def get_ratings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    ratings = pd.read_csv("FoodAndRatings/Ratings.csv")
    #ratings.head()

    movies = pd.read_csv("FoodAndRatings/indian_food.csv")
    #movies.head()

    n_ratings = len(ratings)
    n_movies = len(ratings['Food_Id'].unique())
    n_users = len(ratings['User_Id'].unique())

    print(f"Number of ratings: {n_ratings}")
    print(f"Number of unique movieId's: {n_movies}")
    print(f"Number of unique users: {n_users}")
    print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
    print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

    user_freq = ratings[['User_Id', 'Food_Id']].groupby('User_Id').count().reset_index()
    user_freq.columns = ['User_Id', 'Ratings']
    #user_freq.head()


    # Find Lowest and Highest rated movies:
    mean_rating = ratings.groupby('Food_Id')[['Ratings']].mean()
    # Lowest rated movies
    lowest_rated = mean_rating['Ratings'].idxmin()
    movies.loc[movies['Id'] == lowest_rated]
    # Highest rated moviess
    highest_rated = mean_rating['Ratings'].idxmax()
    movies.loc[movies['Id'] == highest_rated]
    # show number of people who rated movies rated movie highest
    ratings[ratings['Food_Id']==highest_rated]
    # show number of people who rated movies rated movie lowest
    ratings[ratings['Food_Id']==lowest_rated]

    ## the above movies has very low dataset. We will use bayesian average
    movie_stats = ratings.groupby('Food_Id')[['Ratings']].agg(['count', 'mean'])
    movie_stats.columns = movie_stats.columns.droplevel()
    return ratings,movies

# Now, we create user-item matrix using scipy csr matrix


def create_matrix(df):
	
	N = len(df['User_Id'].unique())
	M = len(df['Food_Id'].unique())
	
	# Map Ids to indices
	user_mapper = dict(zip(np.unique(df["User_Id"]), list(range(N))))
	movie_mapper = dict(zip(np.unique(df["Food_Id"]), list(range(M))))
	
	# Map indices to IDs
	user_inv_mapper = dict(zip(list(range(N)), np.unique(df["User_Id"])))
	movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["Food_Id"])))
	
	user_index = [user_mapper[i] for i in df['User_Id']]
	movie_index = [movie_mapper[i] for i in df['Food_Id']]

	X = csr_matrix((df["Ratings"], (movie_index, user_index)), shape=(M, N))
	
	return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper



"""
Find similar movies using KNN
"""
def find_similar_movies(movie_id, movie_mapper,movie_inv_mapper,X, k, metric='cosine', show_distance=False):
	
	neighbour_ids = []
	
	movie_ind = movie_mapper[movie_id]
	movie_vec = X[movie_ind]
	k+=1
	kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
	kNN.fit(X)
	movie_vec = movie_vec.reshape(1,-1)
	neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
	for i in range(0,k):
		n = neighbour.item(i)
		neighbour_ids.append(movie_inv_mapper[n])
	neighbour_ids.pop(0)
	return neighbour_ids


def get_recommendations(food_id):
    ratings, movies = get_ratings()
    X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)
    movie_titles = dict(zip(movies['Id'], movies['name']))

    movie_id = food_id

    similar_ids = find_similar_movies(movie_id, movie_mapper,movie_inv_mapper,X, k=10)
    movie_title = movie_titles[movie_id]

    print(f"Since you ate {movie_title}")
    all_recommendations = []
    for i in similar_ids:
        all_recommendations.append(movie_titles[i])
    return all_recommendations

def get_detections(im):


    test_metadata = MetadataCatalog.get("./images/")



    with open('ClassInfo/annotations.json','r') as f:
        data = json.load(f)



    #for imageName in glob.glob('./images/*jpg'):
    #im = cv2.imread(image)
    outputs = predictor(im)
    classes = outputs['instances'].pred_classes.tolist()
    boxes = outputs['instances'].pred_boxes.to('cpu').tensor.tolist()
    all_boxes = []
    for b in boxes:
        box = [int(i) for i in b]
        #print(box)
        all_boxes.append(box)
    all_classes = []
    for c in classes:
        print("Found --> ",data['categories'][c]['name'])
        all_classes.append(data['categories'][c]['name'])
    print()
    print()
    #print(imageName,"===========================================",outputs['instances'].pred_classes.tolist())
    #print(data['categories'][2])
    v = Visualizer(im[:, :, ::-1], metadata=test_metadata,scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    outputImage = out.get_image()[:, :, ::-1]
    outputImage = cv2.resize(outputImage,(512,512))
    cv2.imwrite('output.jpg',outputImage)
    return {"boxes":all_boxes,"classes":all_classes}

detections = None

@app.post("/files/")
async def create_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    global detections
    detections = get_detections(img)
    return detections
    #return {"file_size": len(file)}


@app.get("/image_return/")
async def main():
    return FileResponse('output.jpg')
    
@app.get("/recommendations/")
async def recommendation_func():
    food_classes = detections['classes']
    df = pd.read_csv('FoodAndRatings/indian_food.csv')
    food_id = None
    if len(food_classes)>0:
        c = food_classes[0]
        for i in range(len(df)):
            if df['name'].iloc[i] == c:
                food_id = df['Id'].iloc[i]
                break
    else:
        return {"recommendations":"None"}
    if food_id is not None:
        recommendations = get_recommendations(food_id)
        return {"recommendations":recommendations}
    else:
        return {"recommendations":"None"}
    
@app.get("/ingredients/")
async def ingredients_func():
    food_classes = detections['classes']
    df = pd.read_csv('FoodAndRatings/indian_food.csv')
    food_ingredients = None
    if len(food_classes)>0:
        c = food_classes[0]
        for i in range(len(df)):
            if df['name'].iloc[i] == c:
                food_ingredients= df['ingredients'].iloc[i]
                break
        return {"ingredients":food_ingredients}
    else:
        return {"ingredients":"None"}
    

    
    
    
    
    
    
    
