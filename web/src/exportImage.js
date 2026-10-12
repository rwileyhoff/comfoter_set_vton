// Client-side export: convert a FOLIO frame to PNG / JPEG / WEBP at a
// chosen size, copy it to the clipboard, or zip a whole reel.
import { pad4, VIEW_NAMES } from './constants.js';
import { makeZip } from './zip.js';

export const FORMATS = {
  png: { mime: 'image/png', ext: 'png', label: 'PNG', lossy: false },
  jpeg: { mime: 'image/jpeg', ext: 'jpg', label: 'JPEG', lossy: true },
  webp: { mime: 'image/webp', ext: 'webp', label: 'WEBP', lossy: true },
};
export const SIZES = [
  { id: 'full', label: 'Full' },
  { id: '1024', label: '1024', edge: 1024 },
  { id: 'web', label: 'Web 1200', width: 1200 },
];
export const DEFAULT_EXPORT = { format: 'png', size: 'full', quality: 0.92 };

const EXPORT_KEY = 'fabric.export.v1';
export function loadExportPrefs() {
  try { return { ...DEFAULT_EXPORT, ...JSON.parse(window.localStorage.getItem(EXPORT_KEY) || '{}') }; } catch (e) { return { ...DEFAULT_EXPORT }; }
}
export function saveExportPrefs(p) {
  try { window.localStorage.setItem(EXPORT_KEY, JSON.stringify(p)); } catch (e) { /* ignore */ }
}
export function prefsLabel(p) {
  const f = FORMATS[p.format] || FORMATS.png;
  const s = SIZES.find((x) => x.id === p.size) || SIZES[0];
  return `${f.label} · ${s.label}${f.lossy ? ` · q${Math.round(p.quality * 100)}` : ''}`;
}

export const slug = (s) => String(s || '').replace(/\.[a-z0-9]{2,4}$/i, '').replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '').slice(0, 40);

export function baseName(f) {
  const parts = [`R${pad4(f.n)}`, f.view];
  if (slug(f.fabric)) parts.push(slug(f.fabric));
  if (slug(f.name)) parts.push(slug(f.name));
  return parts.join('_');
}

export function frameRecord(f, reelTitle) {
  return {
    frame: `R${pad4(f.n)}`, name: f.name || '', note: f.note || '', tag: f.tag || '',
    reel: reelTitle || '', view: f.view, view_name: VIEW_NAMES[f.view], engine: f.engine,
    size: f.size, fabric: f.fabric || '', seconds: f.seconds, settings: f.settings || null,
  };
}

function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error('Could not read the render image'));
    im.src = url;
  });
}

export async function convertFrame(f, prefs = DEFAULT_EXPORT) {
  const fmt = FORMATS[prefs.format] || FORMATS.png;
  const sz = SIZES.find((s) => s.id === prefs.size) || SIZES[0];
  if (fmt.mime === 'image/png' && sz.id === 'full') {
    const r = await fetch(f.url);
    return { blob: await r.blob(), name: `${baseName(f)}.png` };
  }
  const im = await loadImage(f.url);
  const w = im.naturalWidth; const h = im.naturalHeight;
  let scale = 1;
  if (sz.edge) scale = Math.min(1, sz.edge / Math.max(w, h));
  if (sz.width) scale = Math.min(1, sz.width / w);
  const cw = Math.max(1, Math.round(w * scale)); const ch = Math.max(1, Math.round(h * scale));
  const c = document.createElement('canvas');
  c.width = cw; c.height = ch;
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
  if (fmt.lossy) { g.fillStyle = '#ffffff'; g.fillRect(0, 0, cw, ch); }
  g.drawImage(im, 0, 0, cw, ch);
  const blob = await new Promise((res, rej) => c.toBlob(
    (b) => (b ? res(b) : rej(new Error(`This browser cannot write ${fmt.label}`))),
    fmt.mime, fmt.lossy ? prefs.quality : undefined,
  ));
  if (blob.type && blob.type !== fmt.mime) throw new Error(`This browser cannot write ${fmt.label}`);
  return { blob, name: `${baseName(f)}${sz.id === 'full' ? '' : `_${cw}w`}.${fmt.ext}` };
}

export function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export async function copyFrameImage(f) {
  if (!navigator.clipboard || typeof window.ClipboardItem === 'undefined') {
    throw new Error('This browser cannot copy images. Use Export instead.');
  }
  const png = fetch(f.url).then((r) => r.blob()).then((b) => (b.type === 'image/png' ? b : convertFrame(f, DEFAULT_EXPORT).then((x) => x.blob)));
  await navigator.clipboard.write([new window.ClipboardItem({ 'image/png': png })]);
}

export async function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) { await navigator.clipboard.writeText(text); return; }
  const ta = document.createElement('textarea');
  ta.value = text; document.body.appendChild(ta); ta.select();
  document.execCommand('copy'); ta.remove();
}

export async function exportReelZip(reel, frames, prefs) {
  const files = [];
  for (const f of frames) {
    const { blob, name } = await convertFrame(f, prefs);
    files.push({ name, data: blob });
  }
  const meta = {
    reel: reel.title, exported: new Date().toISOString(), format: prefsLabel(prefs),
    frames: frames.map((f) => frameRecord(f, reel.title)),
  };
  files.push({ name: 'settings.json', data: JSON.stringify(meta, null, 2) });
  const zip = await makeZip(files);
  const d = new Date();
  const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  downloadBlob(zip, `${slug(reel.title) || 'reel'}_${stamp}.zip`);
  return frames.length;
}
