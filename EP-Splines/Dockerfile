FROM python:3.6-slim

RUN pip3 install --upgrade pip && \
    pip3 install jupyter && \
    pip3 install pandas && \
    pip3 install numpy && \
    pip3 install matplotlib

WORKDIR /mac0210

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

