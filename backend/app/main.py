from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import AnalysisRequest, SummaryRequest, TranslateRequest
from . import services

app = FastAPI(title='Voice Reader API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/translate')
def translation(request: TranslateRequest):
    return {'translation': services.translate(request.text, request.target_language)}


@app.post('/summarize')
def summary(request: SummaryRequest):
    return {'summary': services.summarize(request.text, request.style)}


@app.post('/analysis/session')
def session_analysis(request: AnalysisRequest):
    return services.analyze(request.text, request.transcript, request.duration_seconds)
