"""Model-backed application services."""
from functools import lru_cache
import os
import re
import requests

NLLB_MODEL_ID = 'facebook/nllb-200-distilled-600M'
NLLB_LANGUAGE_CODES = {
    'English': 'eng_Latn',
    'French': 'fra_Latn',
    'Kinyarwanda': 'kin_Latn',
}


class TranslationServiceUnavailable(RuntimeError):
    """Raised when the local Hugging Face translation model cannot be loaded."""


def translate(text: str, source_language: str, target_language: str) -> str:
    """Translate a supported language pair with NLLB-200.

    The backend supports both local model inference and Hugging Face Inference API.
    Local inference is used by default; if local loading fails and an HF token is
    configured, the service falls back to the hosted API.
    """
    if source_language == target_language:
        return text

    use_remote = os.getenv('HF_INFERENCE_API', '0') == '1'
    if use_remote:
        return _hf_inference_translate(text, source_language, target_language)

    try:
        tokenizer, model, torch = _nllb_components()
        source_code = NLLB_LANGUAGE_CODES[source_language]
        target_code = NLLB_LANGUAGE_CODES[target_language]
        tokenizer.src_lang = source_code
        if callable(getattr(tokenizer, 'set_src_lang_special_tokens', None)):
            tokenizer.set_src_lang_special_tokens(source_code)
        if callable(getattr(tokenizer, 'set_tgt_lang_special_tokens', None)):
            tokenizer.set_tgt_lang_special_tokens(target_code)
        forced_bos_token_id = _nllb_language_id(tokenizer, target_code)
        if forced_bos_token_id is None:
            raise TranslationServiceUnavailable(
                f'Unable to resolve the target language token id for {target_code}.'
            )

        segments = _split_sentences(text)
        if len(segments) <= 1 or len(text) < 120:
            return _generate_translation(model, tokenizer, torch, text, forced_bos_token_id)

        return ' '.join(_generate_translation(model, tokenizer, torch, segment, forced_bos_token_id) for segment in segments)
    except TranslationServiceUnavailable as error:
        if os.getenv('HF_API_TOKEN'):
            return _hf_inference_translate(text, source_language, target_language)
        raise


def _generate_translation(model, tokenizer, torch, text: str, forced_bos_token_id: int) -> str:
    # Low-latency generation defaults. Can be tuned via environment variable
    FAST_MODE = (os.getenv('NLLB_FAST_MODE', '1') == '1')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    if FAST_MODE:
        num_beams = 1
        max_new_tokens = min(256, max(64, inputs.input_ids.shape[1] * 2))
        no_repeat_ngram_size = None
    else:
        num_beams = 4
        max_new_tokens = min(512, inputs.input_ids.shape[1] * 4)
        no_repeat_ngram_size = 3

    gen_kwargs = dict(
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    if not FAST_MODE:
        gen_kwargs['early_stopping'] = True
    if no_repeat_ngram_size:
        gen_kwargs['no_repeat_ngram_size'] = no_repeat_ngram_size

    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _nllb_language_id(tokenizer, language_code: str) -> int | None:
    try:
        return tokenizer.lang_code_to_id[language_code]
    except Exception:
        return tokenizer.convert_tokens_to_ids(language_code)


def _hf_inference_translate(text: str, source_language: str, target_language: str) -> str:
    api_token = os.getenv('HF_API_TOKEN')
    if not api_token:
        raise TranslationServiceUnavailable('HF_API_TOKEN is required for remote inference.')

    source_code = NLLB_LANGUAGE_CODES[source_language]
    target_code = NLLB_LANGUAGE_CODES[target_language]
    payload = {
        'inputs': text,
        'parameters': {
            'source_language': source_code,
            'target_language': target_code,
            'task': 'translation',
        },
    }
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json',
    }
    endpoint = os.getenv('HF_INFERENCE_URL', 'https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M')

    response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if not response.ok:
        raise TranslationServiceUnavailable(
            f'Hugging Face inference API returned {response.status_code}: {response.text}'
        )

    data = response.json()
    if isinstance(data, dict) and data.get('error'):
        raise TranslationServiceUnavailable(f'Hugging Face error: {data["error"]}')
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if 'translation_text' in data[0]:
            return data[0]['translation_text']
        if 'generated_text' in data[0]:
            return data[0]['generated_text']
    if isinstance(data, dict) and 'translation_text' in data:
        return data['translation_text']
    if isinstance(data, str):
        return data
    raise TranslationServiceUnavailable('Unexpected response from Hugging Face inference API.')


@lru_cache(maxsize=1)
def _nllb_components():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
        # Optional quantized load for lower-memory, faster CPU inference when supported
        quantize = (os.getenv('NLLB_QUANTIZE', '0') == '1')
        model = None
        if quantize:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID, device_map='auto', load_in_8bit=True)
            except Exception:
                # bitsandbytes/load_in_8bit not available or failed; fall back
                model = None
        if model is None:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID, device_map='auto')
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_ID)
        model.eval()
        # Optional lightweight warm-up to reduce first-inference latency.
        try:
            if os.getenv('NLLB_WARMUP', '1') == '1':
                tiny = tokenizer('Hello', return_tensors='pt')
                with torch.inference_mode():
                    _ = model.generate(**tiny, max_new_tokens=1)
        except Exception:
            # Warm-up is best-effort; ignore failures.
            pass
        return tokenizer, model, torch
    except Exception as error:
        raise TranslationServiceUnavailable(
            f'Unable to load {NLLB_MODEL_ID}. Install backend requirements and ensure '
            f'the server can download Hugging Face model files. Details: {error}'
        ) from error


def summarize(text: str, style: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = [sentence for sentence in sentences if sentence][:3]
    if style == 'bullets':
        return '\n'.join(f'• {sentence}' for sentence in selected)
    return ' '.join(selected)


def analyze(text: str, transcript: str, duration_seconds: float) -> dict:
    expected = _words(text)
    spoken = _words(transcript or text)
    skipped = _unique_in_order(word for word in expected if word not in spoken)[:8]
    extra = _unique_in_order(word for word in spoken if word not in expected)[:8]
    wpm = round(len(spoken) / duration_seconds * 60)
    pace = 'A little slow' if wpm < 100 else 'A little fast' if wpm > 170 else 'Comfortable'
    score = max(50, 96 - len(skipped) * 5 - len(extra) * 3)
    recommendations = [
        'Try connecting short phrases more smoothly.' if wpm < 100 else 'Slow down slightly at sentence endings.' if wpm > 170 else 'Your pace is well balanced.',
        'Pause briefly at punctuation and before long clauses.',
        'Emphasize the key nouns and verbs in each sentence.',
    ]
    return {'wpm': wpm, 'pace': pace, 'score': score, 'skipped_words': skipped, 'extra_words': extra, 'recommendations': recommendations}


def _words(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ÿ']+", text.lower())


def _unique_in_order(words):
    return list(dict.fromkeys(words))
