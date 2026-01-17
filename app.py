from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
import pandas as pd

from src.constants import APP_HOST, APP_PORT
from src.pipline.training_pipeline import TrainPipeline
from src.pipline.prediction_pipeline import DDoSPredictor

app = FastAPI(title="Open-Set DDoS Detection API")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- HOME (FORM) --------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "ddos_form.html",
        {"request": request, "result": None}
    )

# -------------------- TRAIN --------------------

@app.get("/train")
def train_model():
    try:
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        return {"status": "success", "message": "Model trained and pushed to S3"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------- PREDICT (FORM) --------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict_ddos(request: Request):
    try:
        form = await request.form()

        data = {key: float(form.get(key)) for key in form.keys()}
        df = pd.DataFrame([data])

        predictor = DDoSPredictor()
        prediction = predictor.predict(df)[0]

        return templates.TemplateResponse(
            "ddos_form.html",
            {"request": request, "result": prediction}
        )

    except Exception as e:
        return templates.TemplateResponse(
            "ddos_form.html",
            {"request": request, "result": f"Error: {e}"}
        )

# -------------------- MAIN --------------------

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)

