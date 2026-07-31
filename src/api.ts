const baseUrl = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function post<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  })
  if (!response.ok) throw new Error(`Request failed (${response.status})`)
  return response.json() as Promise<T>
}

export const ai = {
  translate: (text: string, target_language: string) => post<{ translation: string }>('/translate', { text, target_language }),
  summarize: (text: string, style: 'short' | 'bullets') => post<{ summary: string }>('/summarize', { text, style }),
  analyze: (text: string, transcript: string, duration_seconds: number) => post<Analysis>('/analysis/session', { text, transcript, duration_seconds }),
}

export type Analysis = { wpm: number; pace: string; score: number; skipped_words: string[]; extra_words: string[]; recommendations: string[] }
