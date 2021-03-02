from fastapi import FastAPI

api = FastAPI()

@api.get("/")
def read_root():
    return {"Hello": "World"}

@api.get("/health", status_code=200)
def healthcheck():
    return "App is ready to go."