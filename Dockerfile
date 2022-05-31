# syntax=docker/dockerfile:1

FROM python:3.6-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python", "./predict" ]