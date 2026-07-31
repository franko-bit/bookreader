from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import AnalysisRequest, SummaryRequest, TranslateRequest
from . import services

app = FastAPI(title='Voice Reader API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r'https?://(localhost|127\.0\.0\.1|0\.0\.0\.0):\d+',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def root():
    return {
        'message': 'Voice Reader API is running',
        'docs': '/docs',
        'health': '/health',
    }


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/translate')
def translation(request: TranslateRequest):
    try:
        return {'translation': services.translate(request.text, request.source_language, request.target_language)}
    except services.TranslationServiceUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post('/summarize')
def summary(request: SummaryRequest):
    return {'summary': services.summarize(request.text, request.style)}


@app.post('/analysis/session')
def session_analysis(request: AnalysisRequest):
    return services.analyze(request.text, request.transcript, request.duration_seconds)
