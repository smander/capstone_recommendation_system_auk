# pull the official docker image
FROM python:3.9-slim
RUN apt-get update && apt -y install curl gnupg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m pip install mediapipe

# copy project
COPY . .
