FROM python:3.6
WORKDIR /
COPY ./competition /competition
WORKDIR /competition
RUN pip install -r requirements.txt
