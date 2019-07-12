FROM python:3-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/train_model:$PYTHONPATH

COPY requirements.txt /opt/train_model/requirements.txt
RUN pip install -r /opt/train_model/requirements.txt

COPY . /opt/train_model/

WORKDIR /opt/train_model/
