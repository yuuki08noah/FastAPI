# 1. Python 3.10 이미지를 사용
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /myApi

# 3. 필요한 파일 복사
COPY main.py /myApi/main.py
COPY requirements.txt /myApi/requirements.txt

# 4. 필요한 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. Uvicorn으로 FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]