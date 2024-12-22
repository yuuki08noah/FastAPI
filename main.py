
from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from bs4 import BeautifulSoup
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Input, Dropout
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(128, activation='leaky_relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='leaky_relu', kernel_initializer='he_normal'))
model.add(Dense(2, activation='linear', kernel_initializer='he_normal'))
model.compile(optimizer=optimizers.Adam(), loss='mae', metrics=['mse', 'mae'])

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 날씨 데이터 생성
# def weather():
#     svg = open('Busan Districts.svg', 'r').read()
#     data = csv.reader(open('Summary Indicators Dec 3 2024.csv', 'r'), delimiter=",")
#     age = {}

#     for row in data:
#         gu = row[0]
#         try:
#             count = float(row[5])
#             age[gu] = count
#         except:
#             continue
#     result = []
#     soup = BeautifulSoup(svg)
#     paths = soup.findAll('path')
#     colors = ['#F1EEF6', '#D4B9BA', '#C994C7', '#DF65B0', '#DD1C77', '#980043']
#     path_style = 'fill:'

#     for p in paths:
#         if p['id']:
#             count = age[p['id']]
#             if count >= 52: level = 5
#             elif count >= 50: level = 4
#             elif count >= 48: level = 3
#             elif count >= 45: level = 2
#             elif count >= 42: level = 1
#             else: level = 0
#             p['style'] = path_style + colors[level]
#     print(soup.prettify())

def handle_outliers(data, threshold=1):
    data = data.flatten()
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    data_clean = data.copy()
    outliers = z_scores < threshold
    data_clean[outliers] = np.nan
    data_clean = pd.Series(data_clean).interpolate().values
    return data_clean.reshape(-1, 1)

def create_time_weights(length, decay_factor=0.01):
    weights = np.exp(decay_factor * np.arange(length))
    return weights / np.sum(weights)

def aero():
    csv_domestic = open('data/국내 항공 수송 실적 2023.csv', 'r', encoding='cp949').readlines()[2:]
    csv_international = open('data/국제 항공 수송 실적 2023.csv', 'r', encoding='cp949').readlines()[2:]
    data_domestic = csv.reader(csv_domestic, delimiter=",")
    data_international = csv.reader(csv_international, delimiter=",")
    result_domestic = []
    result_international = []
    x = list(range(1990, 2024))
    
    for row in data_domestic:
        result_domestic.append(int(row[1])/10000)
    
    for row in data_international:
        result_international.append(int(row[1])/10000)

    x_np = np.array(x)
    x_normalized = (x_np - 1990) / (2023-1990)
    x_normalized.reshape(-1, 1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.000001)
    result_domestic_normalized = np.array(result_domestic)
    result_domestic_normalized = scaler.fit_transform(result_domestic_normalized.reshape(-1, 1))
    result_international_normalized = np.array(result_international)
    result_international_normalized = scaler.fit_transform(result_international_normalized.reshape(-1, 1))
    
    y = np.column_stack([handle_outliers(result_domestic_normalized, 0.7), handle_outliers(result_international_normalized, 0.6)])
    model.fit(x_normalized, y, epochs=100, callbacks=[reduce_lr], sample_weight=create_time_weights(len(x)))

    future_year = np.array([(2024 - 1990) / (2023 - 1990)]).reshape(-1, 1)
    prediction = model.predict(future_year)
    prediction = scaler.inverse_transform(prediction)
    plt.plot(x, result_domestic, label='국내선')
    plt.plot(x, result_international, label='국제선')
    plt.xlabel("연도")
    plt.ylabel("여객(만명)")

    plt.title("연도별 국내선, 국제선 여객수 그래프")
    
    plt.savefig('static/travelers_statistics.png')
    plt.close()
    
    return prediction

aero()

def predict(x) :
    return scaler.inverse_transform(model.predict(np.array([(x - 1990) / (2023 - 1990)]).reshape(-1, 1)))

def bestFiveRoutes():
    csv_routes = open('data/한국공항공사 국내노선 이용률 2024년 12월.csv', 'r').readlines()[2:]
    data = list(csv.reader(csv_routes, delimiter=','))
    result = {}
    for row in data:
        if row[0] in result:
            result[row[0]] += int(row[5])
        else:
            result[row[0]] = int(row[5])
            
    return sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]

# 2. FastAPI 엔드포인트 작성하기
@app.get("/", response_class=HTMLResponse)
async def show_weather():
    return HTMLResponse(open('busan.html').read())

@app.get("/aero/travelers")
async def aero_travelers(request: Request):
    prediction = predict(2024).astype(int)
    return templates.TemplateResponse("travelers_statistics.html", {"request":request, "year":2024, "prediction":prediction, "best":bestFiveRoutes()})

@app.post("/aero/prediction")
async def aero_prediction(request: Request, year: int = Form(...)):
    prediction = predict(year).astype(int)
    return templates.TemplateResponse("travelers_statistics.html", {"request":request, "year":year, "prediction":prediction, "best":bestFiveRoutes()})