from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np


modelo = xgb.Booster()
modelo.load_model('modelo_casas_xgboost.json')


class DadosEntrada(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

app = FastAPI()
@app.post("/prever/")
async def prever(dados: DadosEntrada):
    try:
        dados__array = np.array([dados.bedrooms, dados. bathrooms, dados.sqft_living,
                                dados.sqft_lot, dados.floors, dados.waterfront,
                                dados.view, dados.condition, dados.grade,
                                dados.sqft_above, dados.sqft_basement,
                                dados.yr_built, dados.yr_renovated,
                                dados.lat, dados.long, dados.sqft_living15, dados.sqft_lot15])
        dmatrix = xgb.DMatrix(dados__array.reshape(1, -1))

        previsao = modelo.predict(dmatrix)

        return {f"Previsão": previsao.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)