FROM python:3-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/make_dataset:$PYTHONPATH

COPY requirements.txt /opt/make_dataset/requirements.txt
RUN pip install -r /opt/make_dataset/requirements.txt

COPY . /opt/make_dataset/

WORKDIR /opt/make_dataset/
