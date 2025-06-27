from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import app_config

model = None


# Модель входных данных
class InputData(BaseModel):
    features: list[float]


# Модель выходных данных
class OutputData(BaseModel):
    prediction: list[float]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    try:
        model = joblib.load(app_config.path_to_modelfile)
        print("Модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        raise SystemExit(1)
    yield
    print("Модель очищена")


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=OutputData)
async def predict(data: InputData) -> OutputData:
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Преобразование списка в массив
        input_data = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_data)
        return OutputData(prediction=prediction.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
