FROM python:3.11-slim

WORKDIR /app

COPY app/src/main/python/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "-u", "app/src/main/python/main.py"]