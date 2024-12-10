
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import random

app = FastAPI()

# 1. 날씨 데이터 생성
def generate_weather_data():
    cities = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "수원"]
    weather_conditions = ["맑음", "구름 많음", "비", "눈", "흐림", "바람"]

    data = []
    for city in cities:
        temperature = round(random.uniform(-5, 35), 1)  # -5도에서 35도 사이
        humidity = random.randint(30, 90)  # 30%에서 90% 사이
        condition = random.choice(weather_conditions)

        # Python 3.10의 구조적 패턴 매칭을 사용하여 날씨 상태 분류
        match condition:
            case "맑음":
                icon = "☀️"
            case "구름 많음":
                icon = "☁️"
            case "비":
                icon = "🌧️"
            case "눈":
                icon = "❄️"
            case "흐림":
                icon = "🌥️"
            case "바람":
                icon = "💨"
            case _:
                icon = "❓"

        data.append({
            "도시": city,
            "온도 (°C)": temperature,
            "습도 (%)": humidity,
            "날씨": f"{condition} {icon}"  # 날씨 상태와 아이콘 결합
        })

    return pd.DataFrame(data)

# 2. FastAPI 엔드포인트 작성하기
@app.get("/", response_class=HTMLResponse)
async def show_weather():
    df = generate_weather_data()
    
    # HTML 테이블로 변환
    table_html = df.to_html(index=False, escape=False, justify="center", border=1)

    # HTML 페이지 생성
    html_content = f"""
    <html>
        <head>
            <title>대한민국 주요 도시 날씨</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; }}
                table {{ margin: 0 auto; border-collapse: collapse; width: 80%; }}
                th, td {{ padding: 10px; border: 1px solid #ddd; text-align: center; }}
                th {{ background-color: #f4f4f4; }}
            </style>
        </head>
        <body>
            <h1>대한민국 주요 도시 날씨 정보</h1>
            {table_html}
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)