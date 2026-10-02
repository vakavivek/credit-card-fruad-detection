import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.predictor import FraudPredictor

app= FastAPI(
    title="Credit Card Fraud Detection API",
    version= "1.0.0",
    description= "Batch Fraud Detection Using XGBoost"
)

predictor= FraudPredictor()

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API"
    }

@app.get("/health")
def health():
    return {
        "status": "Healthy",
        "model_loaded": True
    }

@app.post("/predict")
async def predict(
    transaction_file: UploadFile= File(...),
    identity_file: UploadFile= File(...)):

    try:
        os.makedirs("temp",exist_ok=True)

        transaction_path= os.path.join("temp", transaction_file.filename)
        identity_path= os.path.join("temp",identity_file.filename)

        with open(transaction_path,"wb") as buffer:
            shutil.copyfileobj(transaction_file.file, buffer)

        with open(identity_path,"wb") as buffer:
            shutil.copyfileobj(identity_file.file, buffer)

        prediction_df= predictor.predict_csv(transaction_path,identity_path)

        output_path= "outputs/predictions.csv"

        predictor.save_predictions(prediction_df,output_path)

        os.remove(transaction_path)
        os.remove(identity_path)

        return JSONResponse(content={
            "message": "Prediction Completed",
            "output_file": output_path,
            "total_transactions": len(prediction_df),
            "fraud_transactions": int(prediction_df['prediction'].sum())
        })
    
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= str(e)
        )
    