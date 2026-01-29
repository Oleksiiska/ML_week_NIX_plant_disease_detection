from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.model import Predictor  # <-- ВАЖЛИВО: так, не ".model"

BASE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="ML API", version="1.0.0")

predictor = Predictor()

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if not (file.content_type or "").lower().startswith("image/"):
        return JSONResponse({"error": "Please upload an image file"}, status_code=400)

    image_bytes = await file.read()
    result = predictor.predict_bytes(image_bytes, topk=3)

    if "text/html" in (request.headers.get("accept") or ""):
        return templates.TemplateResponse("index.html", {"request": request, "result": result})

    return JSONResponse(result)
