import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import { BookMarked, ChevronLeft, ChevronRight, FileUp, Highlighter, Mic, Moon, Pause, Play, RotateCcw, Sparkles, Square, Trash2, Volume2 } from 'lucide-react'
import { ai, type Analysis } from './api'
import type { Bookmark, Recording, ToolResult } from './types'

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()

const sampleText = `Welcome to Voice Reader. Upload a text-based PDF to begin reading with helpful study and speech-coaching tools. Select any passage to translate it, create a summary, listen to it, or practice reading it aloud.\n\nAs you read, focus on natural pauses at commas and full stops. Let important words carry a little more energy, and leave enough room to breathe before long sentences.`
const storeKey = 'voice-reader-state'

function normalise(text: string) { return text.toLowerCase().replace(/[^a-zÀ-ÿ'\s]/g, '').split(/\s+/).filter(Boolean) }
function fallbackAnalysis(text: string, transcript: string, duration: number): Analysis {
  const expected = normalise(text), actual = normalise(transcript)
  const skipped_words = expected.filter((word, i) => !actual.includes(word) && !expected.slice(0, i).includes(word)).slice(0, 8)
  const extra_words = actual.filter((word) => !expected.includes(word)).slice(0, 8)
  const wpm = duration ? Math.round(actual.length / duration * 60) : 0
  const pace = wpm < 100 ? 'A little slow' : wpm > 170 ? 'A little fast' : 'Comfortable'
  return { wpm, pace, score: Math.max(50, 96 - skipped_words.length * 5 - extra_words.length * 3), skipped_words, extra_words, recommendations: [wpm < 100 ? 'Try connecting short phrases more smoothly.' : wpm > 170 ? 'Slow down slightly at sentence endings.' : 'Your pace is well balanced.', 'Pause briefly at punctuation and before long clauses.', 'Emphasize the key nouns and verbs in each sentence.'] }
}

export default function App() {
  const [pages, setPages] = useState<string[]>([sampleText])
  const [fileName, setFileName] = useState('Getting started')
  const [page, setPage] = useState(1)
  const [selectedText, setSelectedText] = useState('')
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([])
  const [result, setResult] = useState<ToolResult | null>(null)
  const [sourceLanguage, setSourceLanguage] = useState('English')
  const [language, setLanguage] = useState('French')
  const [dark, setDark] = useState(true)
  const [recording, setRecording] = useState<Recording | null>(null)
  const [recordingActive, setRecordingActive] = useState(false)
  const [recordingPaused, setRecordingPaused] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  const startedAt = useRef(0)
  const duration = useRef(0)

  useEffect(() => {
    const saved = localStorage.getItem(storeKey)
    if (saved) { const state = JSON.parse(saved) as { page?: number; bookmarks?: Bookmark[] }; setPage(state.page ?? 1); setBookmarks(state.bookmarks ?? []) }
  }, [])
  useEffect(() => localStorage.setItem(storeKey, JSON.stringify({ page, bookmarks })), [page, bookmarks])
  const currentText = pages[page - 1] ?? ''
  const focusText = selectedText || currentText
  const currentBookmark = useMemo(() => bookmarks.some((bookmark) => bookmark.page === page), [bookmarks, page])

  async function uploadPdf(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; if (!file) return
    try {
      const source = new Uint8Array(await file.arrayBuffer())
      const pdf = await pdfjsLib.getDocument({ data: source }).promise
      const extracted: string[] = []
      for (let i = 1; i <= pdf.numPages; i += 1) {
        const content = await (await pdf.getPage(i)).getTextContent()
        extracted.push(content.items.map((item) => ('str' in item ? item.str : '')).join(' ').replace(/\s+/g, ' ').trim() || '[This page has no selectable text.]')
      }
      setPages(extracted); setFileName(file.name); setPage(1); setSelectedText(''); setResult(null); setAnalysis(null)
    } catch { setResult({ title: 'Could not read this PDF', body: 'Please choose a text-based PDF. Scanned PDFs need OCR, which is planned for a later release.', kind: 'coach' }) }
  }

  function captureSelection() {
    const selected = window.getSelection()?.toString().trim() ?? ''
    if (selected) setSelectedText(selected)
  }
  function bookmark() { setBookmarks((items) => currentBookmark ? items.filter((item) => item.page !== page) : [...items, { page, label: `Page ${page}` }]) }
  function speak() { const utterance = new SpeechSynthesisUtterance(focusText); utterance.rate = 0.9; speechSynthesis.cancel(); speechSynthesis.speak(utterance) }
  async function runTool(kind: 'translation' | 'summary', style?: 'short' | 'bullets') {
    if (!focusText.trim()) return
    setResult({ title: 'Working…', body: '', kind })
    try {
      if (kind === 'translation') { const response = await ai.translate(focusText, sourceLanguage, language); setResult({ title: `${language} translation`, body: response.translation, kind }) }
      else { const response = await ai.summarize(focusText, style ?? 'short'); setResult({ title: style === 'bullets' ? 'Key points' : 'Short summary', body: response.summary, kind }) }
    } catch {
      const sentence = focusText.split(/(?<=[.!?])\s+/).slice(0, 2).join(' ')
      setResult({ title: kind === 'translation' ? `${language} translation` : 'Short summary', body: kind === 'translation' ? 'Connect the FastAPI server to generate a translation. The selected passage is ready to send.' : sentence || 'Select some text first.', kind })
    }
  }
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const media = new MediaRecorder(stream); chunks.current = []; startedAt.current = Date.now(); duration.current = 0
      media.ondataavailable = (event) => { if (event.data.size) chunks.current.push(event.data) }
      media.onstop = () => { const blob = new Blob(chunks.current, { type: media.mimeType }); duration.current += (Date.now() - startedAt.current) / 1000; setRecording({ blob, url: URL.createObjectURL(blob), duration: duration.current }); stream.getTracks().forEach((track) => track.stop()) }
      recorder.current = media; media.start(); setRecordingActive(true); setRecordingPaused(false)
    } catch { setResult({ title: 'Microphone unavailable', body: 'Allow microphone access in your browser, then try recording again.', kind: 'coach' }) }
  }
  function pauseRecording() { if (!recorder.current) return; if (recordingPaused) { startedAt.current = Date.now(); recorder.current.resume(); setRecordingPaused(false) } else { duration.current += (Date.now() - startedAt.current) / 1000; recorder.current.pause(); setRecordingPaused(true) } }
  function stopRecording() { recorder.current?.stop(); setRecordingActive(false); setRecordingPaused(false) }
  async function getFeedback() {
    if (!recording) return
    const effectiveTranscript = transcript || focusText
    try { setAnalysis(await ai.analyze(focusText, effectiveTranscript, recording.duration)) } catch { setAnalysis(fallbackAnalysis(focusText, effectiveTranscript, recording.duration)) }
  }

  return <div className={dark ? 'app-shell' : 'app-shell bg-slate-100 text-slate-900'}>
    <header className="flex items-center justify-between border-b border-slate-700 px-4 py-3">
      <div className="flex items-center gap-3"><div className="rounded-lg bg-blue-600 p-2"><Sparkles size={19} /></div><div><h1 className="m-0 text-lg font-bold">Voice Reader</h1><p className="m-0 text-xs muted">Read with clarity. Speak with confidence.</p></div></div>
      <div className="flex items-center gap-2"><label className="toolbar-button primary"><FileUp size={16} /> Upload PDF<input className="hidden" type="file" accept="application/pdf" onChange={uploadPdf} /></label><button className="toolbar-button" onClick={() => setDark(!dark)} aria-label="Toggle theme"><Moon size={16} /></button></div>
    </header>
    <main className="grid min-h-[calc(100vh-73px)] grid-cols-1 gap-4 p-4 xl:grid-cols-[220px_minmax(0,1fr)_340px]">
      <aside className="panel p-4"><h2 className="mb-1 text-sm font-semibold">Document</h2><p className="mb-5 truncate text-sm muted">{fileName}</p><div className="space-y-1"><p className="text-xs font-semibold uppercase tracking-wide muted">Pages</p>{pages.slice(0, 12).map((_, index) => <button key={index} onClick={() => setPage(index + 1)} className={`block w-full rounded px-3 py-2 text-left text-sm ${page === index + 1 ? 'bg-blue-600 text-white' : 'hover:bg-slate-700'}`}>Page {index + 1}</button>)}</div><div className="mt-6"><p className="text-xs font-semibold uppercase tracking-wide muted">Bookmarks</p>{bookmarks.length ? bookmarks.map((item) => <button key={item.page} className="mt-2 block text-sm text-blue-300" onClick={() => setPage(item.page)}><BookMarked className="mr-1 inline" size={14} />{item.label}</button>) : <p className="mt-2 text-sm muted">No bookmarks yet.</p>}</div></aside>
      <section className="panel flex min-w-0 flex-col overflow-hidden"><div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-700 p-3"><div className="flex gap-2"><button className="toolbar-button" onClick={() => setPage(Math.max(1, page - 1))} disabled={page === 1}><ChevronLeft size={16} /></button><span className="self-center text-sm muted">Page {page} of {pages.length}</span><button className="toolbar-button" onClick={() => setPage(Math.min(pages.length, page + 1))} disabled={page === pages.length}><ChevronRight size={16} /></button></div><div className="flex gap-2"><button className="toolbar-button" onClick={bookmark}><BookMarked size={16} fill={currentBookmark ? 'currentColor' : 'none'} /> Bookmark</button><button className="toolbar-button" onClick={captureSelection}><Highlighter size={16} /> Use selection</button></div></div><article className="flex-1 overflow-auto p-7" onMouseUp={captureSelection}><p className="reader-text m-0">{currentText}</p></article><div className="border-t border-slate-700 px-5 py-3 text-xs muted">{selectedText ? `Selected: “${selectedText.slice(0, 95)}${selectedText.length > 95 ? '…' : ''}”` : 'Select text in the reader, then use a tool.'}</div></section>
      <aside className="space-y-4"><section className="panel p-4"><h2 className="mb-3 text-sm font-semibold">Selected text tools</h2><div className="grid grid-cols-2 gap-2"><button className="toolbar-button justify-center" onClick={() => runTool('summary', 'short')}>Summary</button><button className="toolbar-button justify-center" onClick={() => runTool('summary', 'bullets')}>Key points</button><select aria-label="Source language" value={sourceLanguage} onChange={(event) => setSourceLanguage(event.target.value)} className="rounded border border-slate-600 bg-slate-800 px-2 text-sm"><option>English</option><option>French</option><option>Kinyarwanda</option></select><select aria-label="Translation language" value={language} onChange={(event) => setLanguage(event.target.value)} className="rounded border border-slate-600 bg-slate-800 px-2 text-sm"><option>French</option><option>Kinyarwanda</option><option>English</option></select><button className="toolbar-button col-span-2 justify-center" onClick={() => runTool('translation')}>Translate</button></div><button className="toolbar-button mt-2 w-full justify-center" onClick={speak}><Volume2 size={16} /> Read aloud</button>{result && <div className="mt-4 rounded-lg bg-slate-800 p-3"><p className="m-0 text-sm font-semibold">{result.title}</p><p className="mt-2 whitespace-pre-line text-sm leading-6 text-slate-300">{result.body}</p></div>}</section>
        <section className="panel p-4"><h2 className="mb-1 text-sm font-semibold">Practice recording</h2><p className="mb-3 text-xs muted">Record the selected passage, then review your feedback.</p><div className="flex flex-wrap gap-2">{!recordingActive ? <button className="toolbar-button primary" onClick={startRecording}><Mic size={16} /> Record</button> : <><button className="toolbar-button" onClick={pauseRecording}><Pause size={16} /> {recordingPaused ? 'Resume' : 'Pause'}</button><button className="toolbar-button danger" onClick={stopRecording}><Square size={16} /> Stop</button></>} {recording && <><audio controls src={recording.url} className="h-9 max-w-full" /><button className="toolbar-button danger" onClick={() => { URL.revokeObjectURL(recording.url); setRecording(null); setAnalysis(null) }}><Trash2 size={16} /></button></>}</div>{recording && <><label className="mt-3 block text-xs muted">Transcript (optional; transcription API will populate this)</label><textarea value={transcript} onChange={(event) => setTranscript(event.target.value)} placeholder="Paste or type what you said…" className="mt-1 min-h-20 w-full rounded border border-slate-600 bg-slate-800 p-2 text-sm" /><button className="toolbar-button primary mt-2 w-full justify-center" onClick={getFeedback}><Sparkles size={16} /> Analyze reading</button></>}</section>
        {analysis && <section className="panel p-4"><div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Session feedback</h2><strong className="text-xl text-blue-300">{analysis.score}</strong></div><div className="mt-3 grid grid-cols-2 gap-2 text-sm"><div className="rounded bg-slate-800 p-2"><span className="muted">Pace</span><br />{analysis.pace}</div><div className="rounded bg-slate-800 p-2"><span className="muted">Speed</span><br />{analysis.wpm} WPM</div></div><ul className="mt-3 space-y-1 pl-4 text-sm text-slate-300">{analysis.recommendations.map((item) => <li key={item}>{item}</li>)}</ul>{analysis.skipped_words.length > 0 && <p className="mt-3 text-xs text-amber-200">Review: {analysis.skipped_words.join(', ')}</p>}</section>}</aside>
    </main>
    <footer className="fixed bottom-4 left-1/2 flex -translate-x-1/2 gap-2 rounded-xl border border-slate-600 bg-slate-900/95 p-2 shadow-xl"><button className="toolbar-button" onClick={speak}><Play size={16} /> Listen</button><button className="toolbar-button" onClick={() => { speechSynthesis.cancel(); setSelectedText('') }}><RotateCcw size={16} /> Clear</button></footer>
  </div>
}
