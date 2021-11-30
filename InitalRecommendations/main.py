from typing import Optional
import pandas as pd
# from matplotlib import pyplot as plt
# from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    veg: Optional[str] = 'veg'
    cuisine: Optional[str] = 'Healthy Food'
    relevance: Optional[int] = 0
    Dish: Optional[str] = 'brown rice'


app = FastAPI()


def recommend(item):
    print("=================================Line 24============================")
    vegn = item.veg

    cuisine = item.cuisine

    val = item.relevance

    food = pd.read_csv("input/food.csv")
    ratings = pd.read_csv("input/ratings.csv")
    combined = pd.merge(ratings, food, on='Food_ID')
    #ans = food.loc[(food.C_Type == cuisine) & (food.Veg_Non == vegn),['Name','C_Type','Veg_Non']]

    ans = combined.loc[(combined.C_Type == cuisine) & (combined.Veg_Non == vegn) & (
        combined.Rating >= val), ['Name', 'C_Type', 'Veg_Non']]
    names = ans['Name'].tolist()
    x = np.array(names)
    ans1 = np.unique(x)
    finallist = item.Dish

    ##### IMPLEMENTING RECOMMENDER ######
    dataset = ratings.pivot_table(
        index='Food_ID', columns='User_ID', values='Rating')
    dataset.fillna(0, inplace=True)
    csr_dataset = csr_matrix(dataset.values)
    dataset.reset_index(inplace=True)

    model = NearestNeighbors(
        metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(csr_dataset)
    display = food_recommendation(finallist, model, dataset, food, csr_dataset)
    print(display)
    return display


def food_recommendation(Food_Name, model, dataset, food, csr_dataset):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]
    if len(FoodList):
        Foodi = FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances, indices = model.kneighbors(
            csr_dataset[Foodi], n_neighbors=n+1)
        Food_indices = sorted(list(zip(indices.squeeze().tolist(
        ), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append(
                {'Name': food.iloc[i]['Name'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(Recommendations, index=range(1, n+1))
        print(df['Name'])
        return df['Name']
    else:
        return "No Similar Foods."


# def get_recommendation(finallist, model, dataset, food):
#     display =
#     #names1 = display['Name'].tolist()
#     return display

    # x1 = np.array(names)
    # ans2 = np.unique(x1)
    # if bruh == True:
    #     bruh1 = st.checkbox("We also Recommend : ")
    #     if bruh1 == True:
    #         for i in display:
    #             st.write(i)


@app.post("/items/")
async def create_item(item: Item):
    # print("=============================Type========================",
    #       type(recommend(item)))
    # print(item.veg)
    return recommend(item)
