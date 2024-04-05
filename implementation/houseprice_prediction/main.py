from fastapi import FastAPI, HTTPException, status
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
import pickle as pk
import numpy as np
import os

app = FastAPI()


model = pk.load(open("model/brians_custom_lr_model.pkl", "rb"))

class TVSalesPrediction(BaseModel):
    predicted_sales: float

def predict_tv_sales(tv, Model=model):
    X_pred = np.array([tv]).reshape(-1, 1)
    Y_pred_sklearn = Model.predict(X_pred)
    return Y_pred_sklearn[0]

@app.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction(prediction: TVSalesPrediction):
    try:
        tv_sales = predict_tv_sales(prediction.predicted_sales, model)
        return {"predicted_sales": tv_sales[0]}
    except:
        raise HTTPException(status_code=500, detail="ouch! something went wrong :(")

async def run_server():
    load_dotenv()  
    port = int(os.getenv("PORT", 8080)) 
    host = str(os.getenv("HOST", "127.0.0.1"))
    
    config = uvicorn.Config(app, host="127.0.0.1", port=port)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_server())