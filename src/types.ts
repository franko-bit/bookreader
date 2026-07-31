export type ToolResult = { title: string; body: string; kind: 'translation' | 'summary' | 'coach' }
export type Bookmark = { page: number; label: string }
export type Recording = { url: string; duration: number; blob: Blob }
