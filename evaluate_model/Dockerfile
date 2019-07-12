FROM python:3-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/evaluate_model:$PYTHONPATH

COPY requirements.txt /opt/evaluate_model/requirements.txt
RUN pip install -r /opt/evaluate_model/requirements.txt

COPY . /opt/evaluate_model/

WORKDIR /opt/evaluate_model/
