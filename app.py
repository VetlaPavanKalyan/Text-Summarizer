from fastapi import FastAPI
import uvicorn

from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline
from textSummarizer.pipeline.train import TrainingPipeline
from textSummarizer.pipeline.bart_prediction import BartPredictionPipeline



text: str = "What is Text Summarization?"
app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        obj = TrainingPipeline()
        obj.main()
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    

@app.post("/bart-cnn-predict")
async def predict_bart(text, max_length = 128):
    try:
        obj = BartPredictionPipeline()
        text = obj.predict(text, max_length)
        return text
    except Exception as e:
        raise e


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=65535)