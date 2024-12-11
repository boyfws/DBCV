FROM python:3.12

WORKDIR /app

EXPOSE 8888

COPY requirements.txt /app/requirements.txt
COPY tests/requirements_test.txt /app/requirements_test.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements_test.txt

