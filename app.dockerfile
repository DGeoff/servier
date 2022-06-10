FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./api.py /code
COPY ./predictValue.py /code
COPY ./feature_extractor.py /code
COPY ./mymodel /code
CMD ["uvicorn", "app.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]