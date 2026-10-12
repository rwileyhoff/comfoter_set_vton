// FOLIO (render bin) persistence in the browser's IndexedDB.
// Two stores: 'blobs' holds each render image once (key = blob key),
// 'meta' holds one record, 'state': frames (without blob URLs), reels, counters.
// Everything stays on this machine, in this browser profile.

const DB = 'fabric-bin';
const VERSION = 1;

let dbPromise = null;
function open() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') { reject(new Error('This browser has no IndexedDB')); return; }
    const req = indexedDB.open(DB, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('blobs')) db.createObjectStore('blobs');
      if (!db.objectStoreNames.contains('meta')) db.createObjectStore('meta');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('Could not open the bin store'));
  });
  return dbPromise;
}

function tx(store, mode, fn) {
  return open().then((db) => new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const s = t.objectStore(store);
    let out;
    Promise.resolve(fn(s)).then((v) => { out = v; });
    t.oncomplete = () => resolve(out);
    t.onerror = () => reject(t.error || new Error('Bin store write failed'));
    t.onabort = () => reject(t.error || new Error('Bin store write aborted (disk full?)'));
  }));
}
const req2p = (r) => new Promise((res, rej) => { r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });

export function putBlob(key, blob) { return tx('blobs', 'readwrite', (s) => { s.put(blob, key); }); }
export function deleteBlobs(keys) { return keys.length ? tx('blobs', 'readwrite', (s) => { keys.forEach((k) => s.delete(k)); }) : Promise.resolve(); }
export function saveState(state) { return tx('meta', 'readwrite', (s) => { s.put(state, 'state'); }); }

// -> { state, blobs: Map(key -> Blob) } or null when nothing was saved
export async function loadAll() {
  const db = await open();
  const t = db.transaction(['meta', 'blobs'], 'readonly');
  const state = await req2p(t.objectStore('meta').get('state'));
  const keys = await req2p(t.objectStore('blobs').getAllKeys());
  const vals = await req2p(t.objectStore('blobs').getAll());
  const blobs = new Map(keys.map((k, i) => [k, vals[i]]));
  return { state: state || null, blobs };
}

export async function clearAll() {
  await tx('blobs', 'readwrite', (s) => { s.clear(); });
  await tx('meta', 'readwrite', (s) => { s.clear(); });
}

export async function usage() {
  try {
    if (navigator.storage && navigator.storage.estimate) {
      const e = await navigator.storage.estimate();
      return { used: e.usage || 0, quota: e.quota || 0 };
    }
  } catch (e) { /* ignore */ }
  return null;
}

// ask the browser not to evict the bin under storage pressure
export function askPersist() {
  try { if (navigator.storage && navigator.storage.persist) navigator.storage.persist(); } catch (e) { /* ignore */ }
}

export function fmtBytes(n) {
  if (!n) return '0 MB';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(n / 1e6))} MB`;
}
