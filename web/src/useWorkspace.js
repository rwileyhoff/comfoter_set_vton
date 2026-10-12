import { useCallback, useEffect, useRef, useState } from 'react';

// Tear-off palettes: each palette is docked left, docked right, or floating
// at x/y/w. A workspace is that whole layout, saved by name.

const KEY = 'fabric.workspace.v1';
const BUILTINS = ['Default', 'Wide canvas', 'Swapped'];

export function preset(name) {
  if (name === 'Wide canvas') return { left: [], right: ['results'], float: { brief: { x: 24, y: 24, w: 310, z: 2 }, bin: { x: 360, y: 560, w: 290, z: 3 } } };
  if (name === 'Swapped') return { left: ['results', 'bin'], right: ['brief'], float: {} };
  return { left: ['brief'], right: ['results', 'bin'], float: {} };
}

const home = (id) => (id === 'brief' ? 'left' : 'right');

function without(L, id) {
  const f = { ...L.float }; delete f[id];
  return { left: L.left.filter((x) => x !== id), right: L.right.filter((x) => x !== id), float: f };
}

function load() {
  try {
    const d = JSON.parse(window.localStorage.getItem(KEY) || 'null');
    if (d && d.layout && Array.isArray(d.layout.left) && Array.isArray(d.layout.right)) return d;
  } catch (e) { /* ignore */ }
  return null;
}

export function useWorkspace() {
  const saved0 = useRef(load());
  const [layout, setLayout] = useState(saved0.current ? saved0.current.layout : preset('Default'));
  const [saved, setSaved] = useState(saved0.current ? saved0.current.saved || {} : {});
  const [wsName, setWsName] = useState(saved0.current ? saved0.current.wsName || 'Default' : 'Default');
  const [drag, setDrag] = useState(null); // { id, snap }
  const wsRef = useRef(null);
  const zTop = useRef(10);

  useEffect(() => {
    if (drag) return;
    try { window.localStorage.setItem(KEY, JSON.stringify({ layout, saved, wsName })); } catch (e) { /* ignore */ }
  }, [layout, saved, wsName, drag]);

  const toFloat = (L, id, x, y, w) => {
    const b = without(L, id);
    zTop.current += 1;
    b.float[id] = { x: Math.max(0, Math.round(x)), y: Math.max(0, Math.round(y)), w: Math.round(w), z: zTop.current };
    return b;
  };
  const toDock = (L, id, slot, idx) => {
    const b = without(L, id);
    const list = b[slot].slice();
    list.splice(Math.max(0, Math.min(idx, list.length)), 0, id);
    b[slot] = list;
    return b;
  };

  const floatOut = useCallback((id) => {
    setLayout((L) => {
      const n = Object.keys(L.float).length;
      return toFloat(L, id, 340 + 32 * n, 40 + 32 * n, id === 'brief' ? 310 : 290);
    });
  }, []);

  const dockHome = useCallback((id) => setLayout((L) => toDock(L, id, home(id), 99)), []);

  const dropIndex = (slot, clientY) => {
    if (!wsRef.current) return 99;
    let n = 0;
    wsRef.current.querySelectorAll(`[data-dock="${slot}"] [data-panel]`).forEach((el) => {
      const r = el.getBoundingClientRect();
      if (clientY > r.top + r.height / 2) n += 1;
    });
    return n;
  };

  const startDrag = useCallback((id, e) => {
    if (e.button !== undefined && e.button !== 0) return;
    if (e.target.closest && e.target.closest('button')) return;
    const ws = wsRef.current;
    if (!ws) return;
    e.preventDefault();
    const r = ws.getBoundingClientRect();
    const pr = e.currentTarget.parentElement.getBoundingClientRect();
    const offX = e.clientX - pr.left; const offY = e.clientY - pr.top;
    const w = Math.max(260, Math.min(pr.width, 340));
    const zone = Math.min(160, r.width / 6);
    setLayout((L) => toFloat(L, id, pr.left - r.left, pr.top - r.top, w));
    setDrag({ id, snap: null });
    let snap = null;
    const move = (ev) => {
      const x = Math.min(Math.max(0, ev.clientX - r.left - offX), Math.max(0, r.width - 120));
      const y = Math.max(0, ev.clientY - r.top - offY);
      const lx = ev.clientX - r.left;
      snap = lx < zone ? 'left' : (r.width - lx < zone ? 'right' : null);
      setDrag({ id, snap });
      setLayout((L) => (L.float[id] ? { ...L, float: { ...L.float, [id]: { ...L.float[id], x: Math.round(x), y: Math.round(y) } } } : L));
    };
    const up = (ev) => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      if (snap) {
        const idx = dropIndex(snap, ev.clientY);
        setLayout((L) => toDock(L, id, snap, idx));
      }
      setDrag(null);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
  }, []);

  const loadWorkspace = (name) => {
    setLayout(saved[name] ? JSON.parse(JSON.stringify(saved[name])) : preset(name));
    setWsName(name);
  };
  const saveWorkspace = (raw) => {
    let name = (raw || '').trim();
    if (!name) name = BUILTINS.includes(wsName) ? `My workspace ${Object.keys(saved).length + 1}` : wsName;
    setSaved((s) => ({ ...s, [name]: JSON.parse(JSON.stringify(layout)) }));
    setWsName(name);
  };
  const names = BUILTINS.concat(Object.keys(saved).filter((n) => !BUILTINS.includes(n)));

  return { layout, wsRef, drag, startDrag, floatOut, dockHome, wsName, names, loadWorkspace, saveWorkspace };
}
