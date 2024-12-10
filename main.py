
from fastapi import FastAPI
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse
import random

app = FastAPI()

# 1. 날씨 데이터 생성
def gr():
    r = random.randint(10, 1000);
    nums = np.array([i/10 for i in range(r*10)])

    sin = np.sin(nums);
    cos = np.cos(nums);
    
    x = np.arange(0, r, 0.1);
    plt.clf()
    plt.plot(x, sin, color='r', label='sin')
    plt.plot(x, cos, color='g', label='cos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin, cos')
    
    plt.savefig('graph.png')

# 2. FastAPI 엔드포인트 작성하기
@app.get("/", response_class=FileResponse)
async def show_weather():
    gr();
    return FileResponse('graph.png')