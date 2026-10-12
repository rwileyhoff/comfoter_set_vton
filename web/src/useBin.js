import { useCallback, useEffect, useRef, useState } from 'react';
import { REEL_COLORS } from './constants.js';

// Render Bin state. The reel model follows LATTICE COMPOSE's composeReducer:
// reels hold ordered frame ids; frames hold the render and its settings.

const VIEWER_KEY = 'fabric.viewer.v2';

export function useBin() {
  const [frames, setFrames] = useState({});
  const [reels, setReels] = useState([
    { id: 'reel_1', title: 'Renders', color: REEL_COLORS[0], collapsed: false, items: [] },
    { id: 'reel_2', title: 'Selects', color: REEL_COLORS[1], collapsed: false, items: [] },
  ]);
  const counters = useRef({ frame: 0, reel: 2, dup: 0 });
  const [sel, setSel] = useState(null);
  const [cursor, setCursor] = useState({});
  const [playing, setPlaying] = useState(null);
  const [viewerOpen, setViewerOpen] = useState(() => {
    try { return window.localStorage.getItem(VIEWER_KEY) === 'shown'; } catch (e) { return false; }
  });
  const [drag, setDrag] = useState(null);       // { kind: 'frame'|'reel', id }
  const [dropReel, setDropReel] = useState(null);
  const [compareA, setCompareA] = useState(null); // frame id held as compare A
  const timer = useRef(null);
  const stateRef = useRef({});
  stateRef.current = { frames, reels, sel, cursor };

  // add one batch of renders to the "Renders" reel
  const addBatch = useCallback((items) => {
    const made = items.map((it) => {
      counters.current.frame += 1;
      return { ...it, id: `f${counters.current.frame}`, n: counters.current.frame };
    });
    setFrames((f) => { const o = { ...f }; made.forEach((m) => { o[m.id] = m; }); return o; });
    setReels((rs) => {
      let list = rs.map((r) => ({ ...r, items: r.items.slice() }));
      let target = list.find((r) => r.title === 'Renders') || list[0];
      if (!target) {
        counters.current.reel += 1;
        target = { id: `reel_${counters.current.reel}`, title: 'Renders', color: REEL_COLORS[0], collapsed: false, items: [] };
        list = [target].concat(list);
      }
      target.items = target.items.concat(made.map((m) => m.id));
      return list;
    });
    return made;
  }, []);

  const newReel = () => {
    counters.current.reel += 1;
    const n = counters.current.reel;
    setReels((rs) => rs.concat([{ id: `reel_${n}`, title: `Reel ${n}`, color: REEL_COLORS[(n - 1) % REEL_COLORS.length], collapsed: false, items: [] }]));
  };
  const updReel = (id, fn) => setReels((rs) => rs.map((r) => (r.id === id ? fn(r) : r)));
  const renameReel = (id, title) => { const t = title.trim(); if (t) updReel(id, (r) => ({ ...r, title: t })); };
  const toggleReel = (id) => updReel(id, (r) => ({ ...r, collapsed: !r.collapsed }));

  // duplicates share the source blob URL, so only revoke a URL nothing else uses
  const dropFrames = (ids) => {
    setFrames((f) => {
      const o = { ...f };
      ids.forEach((x) => { delete o[x]; });
      const live = new Set(Object.values(o).map((m) => m.url));
      ids.forEach((x) => { const m = f[x]; if (m && !live.has(m.url)) { try { URL.revokeObjectURL(m.url); } catch (e) { /* ignore */ } } });
      return o;
    });
    setSel((s) => (ids.includes(s) ? null : s));
    setCompareA((a) => (ids.includes(a) ? null : a));
  };
  const updateFrame = (fid, patch) => setFrames((f) => (f[fid] ? { ...f, [fid]: { ...f[fid], ...patch } } : f));
  // copy a frame (same render, same R number) onto a reel; lands after the source if same reel
  const duplicateFrame = (fid, toReel) => {
    const src = stateRef.current.frames[fid];
    if (!src) return null;
    counters.current.dup += 1;
    const id = `d${counters.current.dup}`;
    const copy = { ...src, id, dupOf: src.dupOf || fid, name: src.name ? `${src.name} copy` : 'copy' };
    setFrames((f) => ({ ...f, [id]: copy }));
    setReels((rs) => rs.map((r) => {
      if (r.id !== toReel) return r;
      const items = r.items.slice();
      const at = items.indexOf(fid);
      if (at >= 0) items.splice(at + 1, 0, id); else items.push(id);
      return { ...r, items };
    }));
    return id;
  };
  const deleteReel = (id) => {
    const r = stateRef.current.reels.find((x) => x.id === id);
    setReels((rs) => rs.filter((x) => x.id !== id));
    if (r) dropFrames(r.items);
  };
  const removeFrame = (fid) => {
    setReels((rs) => rs.map((r) => ({ ...r, items: r.items.filter((x) => x !== fid) })));
    dropFrames([fid]);
  };
  const moveFrame = (fid, toReel, beforeFid) => {
    if (fid === beforeFid) return;
    setReels((rs) => {
      const list = rs.map((r) => ({ ...r, items: r.items.filter((x) => x !== fid) }));
      const t = list.find((r) => r.id === toReel);
      if (!t) return rs;
      const at = beforeFid ? t.items.indexOf(beforeFid) : -1;
      if (at >= 0) t.items.splice(at, 0, fid); else t.items.push(fid);
      return list;
    });
  };
  const moveReel = (rid, beforeRid) => {
    if (rid === beforeRid) return;
    setReels((rs) => {
      const m = rs.find((r) => r.id === rid);
      if (!m) return rs;
      const rest = rs.filter((r) => r.id !== rid);
      const at = rest.findIndex((r) => r.id === beforeRid);
      rest.splice(at < 0 ? rest.length : at, 0, m);
      return rest;
    });
  };
  const endDrag = () => { setDrag(null); setDropReel(null); };

  // transport
  const liveIds = (r) => r.items.filter((x) => stateRef.current.frames[x]);
  const curIndex = (r) => {
    const ids = liveIds(r);
    const at = ids.indexOf(stateRef.current.sel);
    if (at >= 0) return at;
    return Math.min(stateRef.current.cursor[r.id] || 0, Math.max(0, ids.length - 1));
  };
  const scrollTo = (fid) => {
    requestAnimationFrame(() => {
      const el = document.querySelector(`[data-frame="${fid}"]`);
      if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    });
  };
  const step = (rid, to) => {
    const r = stateRef.current.reels.find((x) => x.id === rid);
    if (!r) return;
    const ids = liveIds(r);
    if (!ids.length) return;
    let i = typeof to === 'function' ? to(curIndex(r), ids.length) : to;
    i = Math.max(0, Math.min(ids.length - 1, i));
    setCursor((c) => ({ ...c, [rid]: i }));
    setSel(ids[i]);
    scrollTo(ids[i]);
  };
  const stopPlay = () => { if (timer.current) { clearInterval(timer.current); timer.current = null; } setPlaying(null); };
  const togglePlay = (rid) => {
    if (playing === rid) { stopPlay(); return; }
    stopPlay();
    setPlaying(rid);
    step(rid, (c) => c);
    timer.current = setInterval(() => step(rid, (c, n) => (c + 1) % n), 700);
  };
  useEffect(() => () => { if (timer.current) clearInterval(timer.current); }, []);

  const selectFrame = (rid, fid) => {
    if (stateRef.current.sel === fid) { setSel(null); return; }
    const r = stateRef.current.reels.find((x) => x.id === rid);
    if (r) setCursor((c) => ({ ...c, [rid]: liveIds(r).indexOf(fid) }));
    setSel(fid);
  };
  const stepSelected = (delta) => {
    const r = stateRef.current.reels.find((x) => x.items.includes(stateRef.current.sel)) || stateRef.current.reels[0];
    if (r) step(r.id, (c) => c + delta);
  };
  const toggleViewer = () => {
    setViewerOpen((v) => {
      try { window.localStorage.setItem(VIEWER_KEY, v ? 'hidden' : 'shown'); } catch (e) { /* ignore */ }
      return !v;
    });
  };

  return {
    frames, reels, sel, setSel, cursor, playing, viewerOpen, drag, dropReel, compareA, setCompareA,
    updateFrame, duplicateFrame,
    setDrag, setDropReel, endDrag,
    addBatch, newReel, renameReel, toggleReel, deleteReel, removeFrame, moveFrame, moveReel,
    step, togglePlay, stopPlay, selectFrame, stepSelected, toggleViewer, curIndex, liveIds,
  };
}
