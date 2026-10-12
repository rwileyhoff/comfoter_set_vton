import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getRooms, postRender } from './api.js';
import {
  ENGINES, MODEL_VARIANTS, VIEW_ORDER, VIEW_NAMES, VIEW_FILLS, PARAMS, DEFAULT_PARAMS, PIECE_COUNTS, PALETTE_TITLES, pad4,
} from './constants.js';
import { useWorkspace } from './useWorkspace.js';
import { useBin } from './useBin.js';
import Brief from './components/Brief.jsx';
import RenderBin from './components/RenderBin.jsx';
import { Grip, FloatOut, DockIn, Pin } from './components/Icons.jsx';

function readImageSize(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => resolve({ w: 0, h: 0 });
    img.src = url;
  });
}

export default function App() {
  // engines
  const [engine, setEngine] = useState('warp');
  const [variant, setVariant] = useState('m1');
  // project + set
  const [pieceCount, setPieceCount] = useState('3 PC');
  const [size, setSize] = useState(2048);
  // rooms / views
  const [rooms, setRooms] = useState(null);
  const [apiError, setApiError] = useState('');
  const [viewsChecked, setViewsChecked] = useState(VIEW_ORDER.slice());
  const [selectedView, setSelectedView] = useState('quarter');
  // fabrics
  const [fabricA, setFabricA] = useState(null);
  const [fabricB, setFabricB] = useState(null);
  const [activeSlot, setActiveSlot] = useState('A');
  const [shamsFabric, setShamsFabric] = useState('A');
  // settings
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [weaveOn, setWeaveOn] = useState(true);
  const [aoOn, setAoOn] = useState(true);
  // ui
  const [tab, setTab] = useState('fabrics');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [latest, setLatest] = useState({}); // view -> frame
  const [pinned, setPinned] = useState({});
  const [binOpen, setBinOpen] = useState(false);
  const [draftWs, setDraftWs] = useState('');

  const ws = useWorkspace();
  const bin = useBin();

  // load rooms once
  useEffect(() => {
    getRooms()
      .then((d) => { setRooms(d.rooms[0]); setApiError(''); })
      .catch(() => setApiError('Cannot reach the render API. Start it with: uvicorn main:app --port 8010'));
  }, []);

  const setFabricFile = useCallback(async (slot, file) => {
    const url = URL.createObjectURL(file);
    const { w, h } = await readImageSize(url);
    const name = (file.name || 'pasted').replace(/\.[^.]+$/, '');
    const set = slot === 'A' ? setFabricA : setFabricB;
    set((prev) => {
      if (prev && prev.url) { try { URL.revokeObjectURL(prev.url); } catch (e) { /* ignore */ } }
      return { file, url, name, w, h, inches: prev ? prev.inches : 35.8 };
    });
  }, []);
  const setFabricInches = (slot, v) => (slot === 'A' ? setFabricA : setFabricB)((p) => (p ? { ...p, inches: v } : p));
  const clearFabricB = () => { setFabricB(null); setShamsFabric('A'); };

  // paste a screenshot into the active fabric slot
  useEffect(() => {
    const onPaste = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      const items = (e.clipboardData && e.clipboardData.items) || [];
      for (const it of items) {
        if (it.type && it.type.startsWith('image/')) {
          const blob = it.getAsFile();
          if (blob) {
            const file = new File([blob], `pasted-${Date.now()}.png`, { type: blob.type });
            setFabricFile(activeSlot, file);
            setTab('fabrics');
            e.preventDefault();
          }
          return;
        }
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [activeSlot, setFabricFile]);

  const effectiveOptions = useMemo(() => ({
    ...params,
    weave_amount: weaveOn ? params.weave_amount : 0,
    ao: aoOn ? params.ao : 0,
  }), [params, weaveOn, aoOn]);

  const command = useMemo(() => {
    const v = viewsChecked.join(' ');
    return `--views ${v} --file-inches ${fabricA ? fabricA.inches : '[in]'} ` +
      PARAMS.map((p) => `${p.flag} ${Number(effectiveOptions[p.key]).toFixed(p.dec)}`).join(' ');
  }, [viewsChecked, fabricA, effectiveOptions]);

  const engineDef = ENGINES.find((e) => e.id === engine);
  const engineNote = engine === 'model' ? MODEL_VARIANTS.find((m) => m.id === variant).note : engineDef.note;
  const canRender = engineDef.ready && !!fabricA && Number(fabricA.inches) > 0 && !busy;

  const doRender = useCallback(async (views) => {
    if (!engineDef.ready) { setError(`${engineDef.label} is not wired yet. Only WARP renders today.`); return; }
    if (!fabricA) { setError('Add Fabric A first (Fabrics tab: upload or paste a repeat).'); setTab('fabrics'); return; }
    if (!views.length) { setError('Tick at least one view in the Room tab.'); return; }
    setBusy(true); setError('');
    try {
      const shams = shamsFabric === 'B' && fabricB ? { file: fabricB.file, inches: Number(fabricB.inches) } : null;
      const res = await postRender({ fabricA: { file: fabricA.file, inches: Number(fabricA.inches) }, shams, views, size, options: effectiveOptions });
      const order = VIEW_ORDER.filter((v) => res.views[v]);
      const made = bin.addBatch(order.map((v) => ({ view: v, url: res.views[v].url, settings: res.settings, engine, size, seconds: res.views[v].seconds, fabric: fabricA.name })));
      setLatest((L) => { const o = { ...L }; made.forEach((m) => { o[m.view] = m; }); return o; });
      if (!order.includes(selectedView) && order.length) setSelectedView(order[0]);
      const failed = Object.entries(res.failed || {});
      if (failed.length) setError(`Some views failed: ${failed.map(([v, m]) => `${VIEW_NAMES[v]}: ${m}`).join('; ')}`);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }, [engineDef, fabricA, fabricB, shamsFabric, size, effectiveOptions, engine, bin, selectedView]);

  const renderAll = () => doRender(VIEW_ORDER.filter((v) => viewsChecked.includes(v)));
  const renderSelected = () => doRender([selectedView]);

  const sendToRender = (f) => {
    const o = (f.settings && f.settings.options) || {};
    setParams((p) => ({ ...p, ...Object.fromEntries(PARAMS.map((x) => [x.key, o[x.key] ?? p[x.key]])) }));
    setWeaveOn(true); setAoOn(true);
    setSelectedView(f.view);
    setEngine(f.engine || 'warp');
    setSize(f.size || 2048);
    setBinOpen(false);
    setTab('render');
  };

  // keyboard: ~ (Backquote) toggles the bin, Esc closes, arrows step the reel
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target; const tag = (t && t.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      if (e.code === 'Backquote' && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); bin.stopPlay(); setBinOpen((b) => !b); return; }
      if (e.key === 'Escape' && binOpen) { e.preventDefault(); bin.stopPlay(); setBinOpen(false); return; }
      if (binOpen && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) { e.preventDefault(); bin.stepSelected(e.key === 'ArrowRight' ? 1 : -1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [binOpen, bin]);

  const frameCount = Object.keys(bin.frames).length;
  const roomViews = rooms ? rooms.views : [];
  const viewInfo = (v) => roomViews.find((x) => x.view === v) || {};
  const current = latest[selectedView] && bin.frames[latest[selectedView].id] ? latest[selectedView] : null;
  const cur = viewInfo(selectedView);

  // ---------- palettes ----------
  const recent = Object.values(bin.frames).sort((a, b) => b.n - a.n).slice(0, 6);
  const paletteBody = (id) => {
    if (id === 'brief') {
      return (
        <Brief tab={tab} setTab={setTab} pieceCount={pieceCount} fabricA={fabricA} fabricB={fabricB}
          activeSlot={activeSlot} setActiveSlot={setActiveSlot} setFabricFile={setFabricFile} setFabricInches={setFabricInches}
          clearFabricB={clearFabricB} shamsFabric={shamsFabric} setShamsFabric={setShamsFabric} rooms={rooms}
          viewsChecked={viewsChecked} toggleView={(v) => setViewsChecked((s) => (s.includes(v) ? s.filter((x) => x !== v) : s.concat([v])))}
          params={params} setParam={(k, v) => setParams((p) => ({ ...p, [k]: v }))} resetParams={() => setParams(DEFAULT_PARAMS)}
          command={command} selectedView={selectedView} renderSelected={renderSelected} />
      );
    }
    if (id === 'results') {
      const rows = VIEW_ORDER.filter((v) => latest[v] && bin.frames[latest[v].id]).map((v) => latest[v]);
      return (
        <div className="list">
          {rows.length === 0 && <div className="hint">No renders yet. Add Fabric A, then Render.</div>}
          {rows.map((f) => (
            <div key={f.id} className="result">
              <button type="button" onClick={() => setSelectedView(f.view)} style={{ padding: 0, border: 'none', background: 'none', cursor: 'pointer' }} aria-label={`Show ${VIEW_NAMES[f.view]}`}>
                <img className="thumb" src={f.url} alt="" />
              </button>
              <div className="meta">
                <div style={{ fontSize: 12.5, color: 'var(--ink-hi)' }}>{VIEW_NAMES[f.view]}</div>
                <div className="hint mono" style={{ fontSize: 11 }}>R{pad4(f.n)} · {f.size}px · {f.seconds}s</div>
              </div>
              <button type="button" className="icon-btn" style={{ width: 44, height: 44, color: pinned[f.id] ? '#7fb0ee' : 'var(--ink-3)' }}
                aria-pressed={!!pinned[f.id]} aria-label={`${pinned[f.id] ? 'Unpin' : 'Pin'} ${VIEW_NAMES[f.view]}`}
                onClick={() => setPinned((p) => ({ ...p, [f.id]: !p[f.id] }))}><Pin /></button>
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className="list">
        <div className="hint mono">{frameCount} renders · {bin.reels.length} reels</div>
        {recent.length > 0 ? (
          <div className="mini-grid">
            {recent.map((f) => (
              <button key={f.id} type="button" className="mini" aria-label={`Open R${pad4(f.n)} in the Render Bin`}
                onClick={() => { bin.setSel(f.id); setBinOpen(true); }}>
                <img src={f.url} alt="" /><span className="code">R{pad4(f.n)}</span>
              </button>
            ))}
          </div>
        ) : <div className="swatch-drop" style={{ height: 80, cursor: 'default' }}>Renders land here automatically.</div>}
        <button type="button" className="btn-gold" onClick={() => setBinOpen(true)}>Open Render Bin <span className="kbd" style={{ color: 'var(--gold)', borderColor: '#5a4a38' }}>~</span></button>
      </div>
    );
  };

  // a plain function, not a component: a component defined in here would
  // remount on every render and drop focus from the inputs inside it
  const renderPalette = (id, where) => {
    const f = ws.layout.float[id] || {};
    const floating = where === 'float';
    return (
      <div key={id} data-panel={id} className={`palette ${floating ? 'floating' : ''}`}
        style={floating ? { left: f.x, top: f.y, width: f.w, zIndex: f.z } : undefined}>
        <div className="palette-head" onPointerDown={(e) => ws.startDrag(id, e)}>
          <Grip />
          <div className="title">{PALETTE_TITLES[id]}</div>
          <button type="button" className="icon-btn" aria-label={`${floating ? 'Dock' : 'Float'} ${PALETTE_TITLES[id].toLowerCase()}`}
            title={floating ? 'Dock' : 'Float'} onClick={() => (floating ? ws.dockHome(id) : ws.floatOut(id))}>
            {floating ? <DockIn /> : <FloatOut />}
          </button>
        </div>
        <div className={floating ? 'body-scroll' : undefined}>{paletteBody(id)}</div>
      </div>
    );
  };

  // ---------- render ----------
  return (
    <div className="app">
      <div className="frame">
        <div className="titlebar">
          <div className="brand">FABRIC</div>
          <div className="project">
            <div className="name">{fabricA ? fabricA.name : 'New set'} — Queen {pieceCount}</div>
            <div className="meta">PDP · {viewsChecked.length} views · 2 pieces · {rooms ? 'render_v2' : 'API offline'}</div>
          </div>
          <div className="engines" role="group" aria-label="Render engine">
            {ENGINES.map((en) => {
              const on = engine === en.id;
              return (
                <button key={en.id} type="button" className="engine" aria-pressed={on} onClick={() => setEngine(en.id)}
                  title={en.ready ? en.note : `${en.note} — not wired yet`}
                  style={on ? { background: en.fill, color: en.ink, borderColor: '#f0f0f0' } : { color: en.line, borderColor: en.line }}>
                  {en.label}
                </button>
              );
            })}
            {engine === 'model' && (
              <div className="variant" role="group" aria-label="Model">
                {MODEL_VARIANTS.map((m) => (
                  <button key={m.id} type="button" aria-pressed={variant === m.id} className={variant === m.id ? 'on' : ''} onClick={() => setVariant(m.id)}>{m.label}</button>
                ))}
              </div>
            )}
          </div>
          <div className="actions">
            <button type="button" className={`btn btn-action btn-bin ${binOpen ? 'on' : ''}`} aria-pressed={binOpen} title="Render Bin (~)"
              onClick={() => { bin.stopPlay(); setBinOpen((b) => !b); }}>
              Render Bin <span className="count">{frameCount}</span><span className="kbd">~</span>
            </button>
            <button type="button" className="btn btn-action" title="Blueprint JSON: later">Blueprint JSON</button>
            <button type="button" className="btn btn-action btn-primary" onClick={renderAll} disabled={!canRender}
              title={!engineDef.ready ? `${engineDef.label} is not wired yet` : !fabricA ? 'Add Fabric A first' : ''}>
              {busy ? 'Rendering…' : `Render ${viewsChecked.length} view${viewsChecked.length === 1 ? '' : 's'}`}
            </button>
          </div>
        </div>

        {binOpen ? (
          <RenderBin bin={bin} onClose={() => { bin.stopPlay(); setBinOpen(false); }} onSendToRender={sendToRender} />
        ) : (
          <>
            <div className="toolbar">
              <label>Piece count
                <select className="select" value={pieceCount} onChange={(e) => setPieceCount(e.target.value)}>
                  {Object.keys(PIECE_COUNTS).map((k) => <option key={k}>{k}</option>)}
                </select>
              </label>
              <label>Size<select className="select" defaultValue="QUEEN 90 × 90"><option>QUEEN 90 × 90</option><option>KING 106 × 90</option></select></label>
              <label>Room<select className="select"><option>{rooms ? rooms.name : 'Base room 1'}</option></select></label>
              <label>Output
                <select className="select" value={size} onChange={(e) => setSize(Number(e.target.value))}>
                  <option value={2048}>2048 px</option><option value={1024}>1024 px</option>
                </select>
              </label>
              <label><input type="checkbox" checked={weaveOn} onChange={(e) => setWeaveOn(e.target.checked)} /> Weave</label>
              <label><input type="checkbox" checked={aoOn} onChange={(e) => setAoOn(e.target.checked)} /> AO</label>
              <div className="right">
                <label>Workspace
                  <select className="select" value={ws.wsName} onChange={(e) => ws.loadWorkspace(e.target.value)}>
                    {ws.names.map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
                <label style={{ fontSize: 12, color: 'var(--ink-3)' }}>Save as
                  <input className="field" style={{ width: 110 }} placeholder="name" value={draftWs} onChange={(e) => setDraftWs(e.target.value)} />
                </label>
                <button type="button" className="btn" onClick={() => { ws.saveWorkspace(draftWs); setDraftWs(''); }}>Save</button>
              </div>
              <div className="engine-note">Engine: {engineNote}</div>
            </div>

            <div className="workspace" ref={ws.wsRef}>
              {ws.layout.left.length > 0 && (
                <div className="dock dock-left" data-dock="left">
                  {ws.layout.left.map((id) => renderPalette(id, 'left'))}
                </div>
              )}

              <div className="centre">
                <div className="ruler"><span>VIEWS</span><span>{(rooms ? rooms.name : 'BASE ROOM 1').toUpperCase()} · {size} PX</span></div>
                <div className="strip" role="tablist" aria-label="Views">
                  {VIEW_ORDER.map((v, i) => {
                    const info = viewInfo(v); const f = latest[v] && bin.frames[latest[v].id] ? latest[v] : null;
                    return (
                      <button key={v} type="button" role="tab" aria-selected={selectedView === v} className={`shot ${selectedView === v ? 'on' : ''}`}
                        onClick={() => setSelectedView(v)}
                        style={{ background: f ? undefined : VIEW_FILLS[v], backgroundImage: f ? `url(${f.url})` : undefined }}>
                        {f && <div className="shade" />}
                        <div className="txt">
                          <div className="t1">#{i + 1} {VIEW_NAMES[v]}</div>
                          <div>ppi {info.comforter_ppi ?? '—'}</div>
                          <div>{info.mode || '—'}{info.gridded && info.gridded.length ? ' · gridded' : ''}</div>
                          <div>{viewsChecked.includes(v) ? `fabric A${shamsFabric === 'B' ? ' + B' : ''}` : 'not ticked'}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>

                <div className="canvas-wrap">
                  <div className="canvas">
                    {current ? <img src={current.url} alt={`${VIEW_NAMES[selectedView]} render R${pad4(current.n)}`} /> : (
                      <div className="empty">
                        <div>[{VIEW_NAMES[selectedView]} render · {size} px]</div>
                        <div className="sub">{fabricA ? 'press Render' : 'add Fabric A: Fabrics tab, upload or paste'}</div>
                      </div>
                    )}
                    {busy && <div className="busy">Rendering on the warp…</div>}
                  </div>
                </div>
                {apiError && <div className="banner err">{apiError}</div>}
                {error && <div className="banner err" style={{ marginTop: 8 }}>{error}</div>}
                {!engineDef.ready && <div className="banner info" style={{ marginTop: 8 }}>{engineDef.label} is a placeholder until its GPU worker exists. Switch to WARP to render.</div>}

                <div className="pieces">
                  <div>
                    <div className="section-head">Comforter — Fabric A</div>
                    <div className="hint">{VIEW_NAMES[selectedView]} surface from the saved grid.</div>
                    <div className="kv">
                      <div>comforter_ppi</div><div className="v">{cur.comforter_ppi ?? '—'}</div>
                      <div>Mode</div><div className="v">{cur.mode || '—'}</div>
                      <div>Fabric A width (in)</div><div className="v">{fabricA ? fabricA.inches : '—'}</div>
                    </div>
                  </div>
                  <div>
                    <div className="section-head muted">Shams — Fabric {shamsFabric}</div>
                    <div className="hint">{cur.gridded && cur.gridded.includes('shams') ? 'Gridded.' : 'No grid yet. Tiles at sham_ppi.'}</div>
                    <div className="kv">
                      <div>sham_ppi</div><div className="v">{cur.sham_ppi ?? '—'}</div>
                      <div>Size (in)</div><div className="v">26 × 20</div>
                    </div>
                  </div>
                </div>
              </div>

              {ws.layout.right.length > 0 && (
                <div className="dock dock-right" data-dock="right">
                  {ws.layout.right.map((id) => renderPalette(id, 'right'))}
                </div>
              )}

              {ws.drag && (
                <>
                  <div className={`snap-zone ${ws.drag.snap === 'left' ? 'hot' : ''}`} style={{ left: 0 }}>DOCK LEFT</div>
                  <div className={`snap-zone ${ws.drag.snap === 'right' ? 'hot' : ''}`} style={{ right: 0 }}>DOCK RIGHT</div>
                </>
              )}
              {Object.keys(ws.layout.float).map((id) => renderPalette(id, 'float'))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
