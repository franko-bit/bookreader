from typing import Literal
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class TranslateRequest(TextRequest):
    source_language: Literal['English', 'French', 'Kinyarwanda'] = 'English'
    target_language: Literal['English', 'French', 'Kinyarwanda']


class SummaryRequest(TextRequest):
    style: Literal['short', 'bullets'] = 'short'


class AnalysisRequest(TextRequest):
    transcript: str = Field(default='', max_length=20_000)
    duration_seconds: float = Field(gt=0, le=7_200)
