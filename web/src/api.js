// Thin client for main.py, reached through the Vite proxy at /api.

export async function getRooms() {
  const r = await fetch('/api/rooms');
  if (!r.ok) throw new Error(`rooms: HTTP ${r.status}`);
  return r.json();
}

export async function getHealth() {
  const r = await fetch('/api/health');
  if (!r.ok) throw new Error(`health: HTTP ${r.status}`);
  return r.json();
}

function b64ToUrl(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
}

/**
 * fabricA: { file, inches }   shams: optional { file, inches }
 * Returns { settings, failed, views: { view: { url, mode, seconds, warnings, width, height } } }
 */
export async function postRender({ fabricA, shams, views, size, options, seed }) {
  const fd = new FormData();
  fd.append('pattern', fabricA.file, fabricA.file.name || 'pattern.png');
  fd.append('file_inches', String(fabricA.inches));
  fd.append('views', views.join(','));
  fd.append('size', String(size));
  fd.append('options', JSON.stringify(options));
  if (seed !== undefined && seed !== null && seed !== '') fd.append('seed', String(seed));
  if (shams) {
    fd.append('shams_pattern', shams.file, shams.file.name || 'shams.png');
    fd.append('shams_file_inches', String(shams.inches));
  }
  const r = await fetch('/api/render', { method: 'POST', body: fd });
  let data;
  try { data = await r.json(); } catch (e) { throw new Error(`render: HTTP ${r.status}`); }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : (d && d.error) || `render: HTTP ${r.status}`);
  }
  const out = {};
  Object.entries(data.views).forEach(([v, x]) => {
    out[v] = { url: b64ToUrl(x.png_base64), mode: x.mode, seconds: x.seconds, warnings: x.warnings, width: x.width, height: x.height };
  });
  return { settings: data.settings, failed: data.failed, views: out };
}
