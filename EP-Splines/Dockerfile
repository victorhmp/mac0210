FROM python:3.6-slim

RUN pip3 install --upgrade pip && \
    pip3 install numpy && \
    pip3 install pandas && \
    pip3 install matplotlib

RUN apt-get update && \
    apt-get install -y --no-install-recommends tk-dev && \
    rm -r /var/lib/apt/lists/*

WORKDIR /mac0210
