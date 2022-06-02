# write some code to build your image
FROM python:3.8.13-buster

COPY beerscan /beerscan
COPY requirements.txt /requirements.txt
#COPY /Users/jbvandeneynde/code/jb-vde/gcp/wagon-data-bootcamp-346912-79b16cf34457.json /credentials.json

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn beerscan.api.fast:app --host 0.0.0.0 --port $PORT
