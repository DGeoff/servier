from fastapi import FastAPI, File, HTTPException, UploadFile, Query
import uvicorn
import predictValue
import feature_extractor

app = FastAPI()
    
@app.post("/predict", response_model=Prediction)
async def prediction(smile_str : str):   
    response = predict_smiles(smile_str)
    return {
        "predictions": response,
    }
# if __name__ == "__main__":
    # uvicorn.run("main:app", host="127.0.0.1", port=5000)