"""MVP service implementations. Replace these functions with model providers in production."""
import re


def translate(text: str, target_language: str) -> str:
    # A deliberately transparent fallback until NLLB or a translation API is configured.
    return f'[{target_language} translation service not configured] {text}'


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
