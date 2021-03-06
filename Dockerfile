FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY Pipfile .
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install pipenv
# RUN pipenv install
# RUN pipenv install --system --deploy --ignore-pipfile

COPY ./app /app
COPY ./models /models
COPY ./src /src
COPY ./data/processed /data/processed
COPY ./docs /docs

WORKDIR /app

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]

