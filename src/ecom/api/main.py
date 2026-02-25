from fastapi import FastAPI

from ecom.api.routers.health import router as health_router
from ecom.api.routers.predict import router as predict_router

app = FastAPI(title='ecom-purchase-prediction')

app.include_router(health_router)
app.include_router(predict_router)