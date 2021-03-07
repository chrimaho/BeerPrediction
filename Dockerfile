FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY Pipfile .

RUN pip install --upgrade pip
RUN pip install pipenv
RUN pip install torch
RUN pipenv install
RUN pipenv install --system --deploy --ignore-pipfile
RUN pipenv lock --requirements > requirements.txt

COPY ./app /app
COPY ./models /models
COPY ./src /src
COPY ./data/processed /data/processed
COPY ./docs /docs

WORKDIR /app

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]

