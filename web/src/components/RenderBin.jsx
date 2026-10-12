import { useEffect, useRef, useState } from 'react';
import { Grip, T, Lock } from './Icons.jsx';
import { VIEW_NAMES, PARAMS, TAGS, pad4, tagOf } from '../constants.js';
import { prefsLabel } from '../exportImage.js';
import { fmtBytes } from '../binStore.js';
import { useFrameActions, FINAL_SIZE } from './FrameActions.jsx';

export function settingsLine(f) {
  const o = (f.settings && f.settings.options) || {};
  const opts = PARAMS.map((p) => `${p.flag} ${Number(o[p.key] ?? 0).toFixed(p.dec)}`).join(' ');
  const seed = f.settings ? f.settings.seed : '';
  return `${f.engine} · ${VIEW_NAMES[f.view]} · ${f.size}px · seed ${seed} · ${opts}`;
}

function ReelStrip({ reel, bin, actions, hover }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(reel.title);
  const frames = reel.items.map((id) => bin.frames[id]).filter(Boolean);
  const cur = frames.length ? bin.curIndex(reel) : 0;
  const hot = bin.dropReel === reel.id;

  const onOver = (e) => {
    if (!bin.drag) return;
    e.preventDefault();
    if (bin.dropReel !== reel.id) bin.setDropReel(reel.id);
  };
  const onDrop = (e) => {
    if (!bin.drag) return;
    e.preventDefault();
    if (bin.drag.kind === 'frame') bin.moveFrame(bin.drag.id, reel.id, null);
    else bin.moveReel(bin.drag.id, reel.id);
    bin.endDrag();
  };
  const commit = () => { setEditing(false); bin.renameReel(reel.id, draft); };
  const tb = (Icon, label, fn, extra) => (
    <button type="button" className={`tbtn ${extra || ''}`} aria-label={label} title={label} onClick={fn}><Icon /></button>
  );

  return (
    <div className={`reel ${reel.locked ? 'locked' : ''}`} onDragOver={onOver} onDrop={onDrop} style={{ borderColor: hot ? reel.color : undefined }}
      onMouseEnter={() => { hover.current.reel = reel.id; }} onMouseLeave={() => { hover.current.reel = null; hover.current.frame = null; }}>
      <div className="reel-head" style={{ borderLeftColor: reel.color, borderBottom: reel.collapsed ? 'none' : '1px solid rgba(255,255,255,0.04)' }}
        onContextMenu={(e) => actions.openReelMenu(e, reel.id)}>
        <div
          className="grip" draggable title="Drag to reorder reels"
          onDragStart={(e) => { try { e.dataTransfer.setData('text/plain', `reel:${reel.id}`); e.dataTransfer.effectAllowed = 'move'; } catch (x) { /* ignore */ } bin.setDrag({ kind: 'reel', id: reel.id }); }}
          onDragEnd={bin.endDrag}
        ><Grip /></div>
        <button type="button" className="plain" aria-label={`${reel.collapsed ? 'Expand' : 'Collapse'} ${reel.title}`}
          onClick={() => bin.toggleReel(reel.id)} style={{ transform: reel.collapsed ? 'rotate(-90deg)' : 'none', fontSize: 12 }}>▾</button>
        {editing ? (
          <input className="reel-title-input" aria-label="Reel name" style={{ borderColor: reel.color }} autoFocus value={draft}
            onChange={(e) => setDraft(e.target.value)} onBlur={commit}
            onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setEditing(false); setDraft(reel.title); } }} />
        ) : (
          <div className="reel-title" title="Double-click to rename" onDoubleClick={() => { setDraft(reel.title); setEditing(true); }}>{reel.title}</div>
        )}
        <button type="button" className={`lock-btn ${reel.locked ? 'on' : ''}`} aria-pressed={!!reel.locked}
          aria-label={`${reel.locked ? 'Unlock' : 'Lock'} ${reel.title}`} title={reel.locked ? 'Locked: steps with the other locked reels (L)' : 'Lock to step with other locked reels (L)'}
          onClick={() => bin.toggleLock(reel.id)}><Lock open={!reel.locked} />{reel.locked ? 'Locked' : 'Lock'}</button>
        <span className="reel-count">{frames.length}</span>
        <button type="button" className="plain" aria-label={`Delete reel ${reel.title}`} title="Delete reel" style={{ fontSize: 15 }} onClick={() => bin.deleteReel(reel.id)}>×</button>
      </div>
      {!reel.collapsed && (
        <>
          <div className="reel-scroll">
            <div className="reel-row">
              {frames.length === 0 && <div className="reel-empty">Drop renders here</div>}
              {frames.map((f) => {
                const on = bin.sel === f.id;
                const tag = tagOf(f.tag);
                const isA = bin.compareA === f.id;
                return (
                  <div key={f.id} className="frame-tile" data-frame={f.id} draggable
                    onContextMenu={(e) => actions.openFrameMenu(e, f.id)}
                    onMouseEnter={() => { hover.current.frame = f.id; }} onMouseLeave={() => { hover.current.frame = null; }}
                    onDragStart={(e) => { try { e.dataTransfer.setData('text/plain', `frame:${f.id}`); e.dataTransfer.effectAllowed = 'move'; } catch (x) { /* ignore */ } bin.setDrag({ kind: 'frame', id: f.id }); }}
                    onDragEnd={bin.endDrag}
                    onDragOver={(e) => { if (bin.drag && bin.drag.kind === 'frame') { e.preventDefault(); e.stopPropagation(); if (bin.dropReel !== reel.id) bin.setDropReel(reel.id); } }}
                    onDrop={(e) => { if (bin.drag && bin.drag.kind === 'frame') { e.preventDefault(); e.stopPropagation(); bin.moveFrame(bin.drag.id, reel.id, f.id); bin.endDrag(); } }}
                  >
                    <button type="button" className="frame-btn" aria-pressed={on}
                      aria-label={`${f.name || VIEW_NAMES[f.view]} render R${pad4(f.n)}${tag ? `, tagged ${tag.label}` : ''}${isA ? ', compare A' : ''}. Right-click or menu key for options`}
                      title={f.note || undefined}
                      onClick={() => bin.selectFrame(reel.id, f.id)}
                      style={{ border: on ? '2px solid #D4A574' : tag ? `2px solid ${tag.color}` : '1px solid rgba(255,255,255,0.12)', boxShadow: on ? '0 0 0 3px rgba(212,165,116,0.18)' : 'none' }}>
                      <img src={f.url} alt="" draggable={false} />
                      {tag && <span className="tag-corner" style={{ background: tag.color }} aria-hidden="true" />}
                      <span className="code">R{pad4(f.n)}{f.note && <span className="note-dot" />}</span>
                      {isA && <span className="ab-badge" aria-hidden="true">A</span>}
                      <span className="name">{f.name || VIEW_NAMES[f.view]}</span>
                      {f.size < FINAL_SIZE && <span className="proxy-badge" title={`Proxy at ${f.size} px. Right-click → Re-render at ${FINAL_SIZE}`}>{f.size}</span>}
                    </button>
                    <button type="button" className="frame-x" aria-label={`Remove R${pad4(f.n)}`} onClick={(e) => { e.stopPropagation(); bin.removeFrame(f.id); }}>×</button>
                  </div>
                );
              })}
            </div>
          </div>
          {frames.length > 0 && (
            <div className="transport-wrap">
              <div className="transport" role="group" aria-label={`Step through ${reel.title}`}>
                <div className="pos">{reel.locked && <span className="pos-lock" title="Locked"><Lock /></span>}R{pad4(frames[cur].n)}  {cur + 1} / {frames.length}</div>
                {tb(T.first, 'First frame', () => bin.step(reel.id, 0))}
                {tb(T.back, 'Back 5 frames', () => bin.step(reel.id, (c) => c - 5))}
                {tb(T.prev, 'Previous frame', () => bin.step(reel.id, (c) => c - 1))}
                {bin.playing === reel.id
                  ? tb(T.pause, 'Pause', () => bin.togglePlay(reel.id), 'playing')
                  : tb(T.play, 'Play through the reel', () => bin.togglePlay(reel.id))}
                {tb(T.next, 'Next frame', () => bin.step(reel.id, (c) => c + 1))}
                {tb(T.fwd, 'Forward 5 frames', () => bin.step(reel.id, (c) => c + 5))}
                {tb(T.last, 'Last frame', () => bin.step(reel.id, (c, n) => n - 1))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

const isTyping = (t) => { const tag = (t && t.tagName) || ''; return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable); };

export default function RenderBin({ bin, onClose, onSendToRender, onRerender, busy }) {
  const count = Object.keys(bin.frames).length;
  const f = bin.sel ? bin.frames[bin.sel] : null;
  const reelOf = f ? bin.reels.find((r) => r.items.includes(f.id)) : null;
  const label = f ? `R${pad4(f.n)} · ${f.name || VIEW_NAMES[f.view]} · ${reelOf ? reelOf.title : ''}` : '';
  const fTag = f ? tagOf(f.tag) : null;
  const actions = useFrameActions({ bin, where: 'bin', onSendToRender, onRerender });
  const hover = useRef({ reel: null, frame: null });
  const [goTo, setGoTo] = useState(null); // string being typed after G, or null
  const locked = bin.lockedSet();
  const multi = bin.viewerOpen && f && reelOf && reelOf.locked && locked.length > 1;

  // Flame-style desktop hotkeys. They act on the frame or reel under the mouse,
  // falling back to the selected frame.
  const live = useRef({});
  live.current = { bin, actions, f, reelOf };
  useEffect(() => {
    const onKey = (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      if (document.querySelector('.modal-scrim, .cmenu-root')) return;
      const { bin: b, actions: a, f: sel, reelOf: selReel } = live.current;
      const fid = (hover.current.frame && b.frames[hover.current.frame]) ? hover.current.frame : sel ? sel.id : null;
      const fr = fid ? b.frames[fid] : null;
      const rid = hover.current.reel || (fr ? (b.reels.find((r) => r.items.includes(fid)) || {}).id : null) || (selReel && selReel.id);
      const k = e.key.toLowerCase();
      if (k === 'c' && rid) { e.preventDefault(); b.toggleReel(rid); return; }
      if (k === 'l' && rid) { e.preventDefault(); b.toggleLock(rid); return; }
      if (k === 'g') { e.preventDefault(); setGoTo(''); return; }
      const tag = { 1: 'approved', 2: 'revise', 3: 'reject', 0: '' }[e.key];
      if (tag !== undefined && fr) { e.preventDefault(); b.updateFrame(fr.id, { tag }); a.say(tag ? `R${pad4(fr.n)} tagged ${tagOf(tag).label}` : `R${pad4(fr.n)} tag cleared`); return; }
      if (e.key === 'Delete' && fr) { e.preventDefault(); b.removeFrame(fr.id); a.say(`Deleted R${pad4(fr.n)}`); return; }
      if (e.key === ' ' && rid && (e.target === document.body || !e.target.closest('button'))) { e.preventDefault(); b.togglePlay(rid); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const jump = (txt) => {
    const n = parseInt(txt, 10);
    setGoTo(null);
    if (!n) return;
    const hit = Object.values(bin.frames).filter((x) => x.n === n).sort((x, y) => (x.dupOf ? 1 : 0) - (y.dupOf ? 1 : 0))[0];
    if (!hit) { actions.say(`No R${pad4(n)} in the bin`, 'err'); return; }
    const r = bin.reels.find((x) => x.items.includes(hit.id));
    if (r && r.collapsed) bin.toggleReel(r.id);
    if (r && bin.sel !== hit.id) bin.selectFrame(r.id, hit.id);
    setTimeout(() => {
      const el = document.querySelector(`[data-frame="${hit.id}"]`);
      if (el) { el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' }); const btn = el.querySelector('.frame-btn'); if (btn) btn.focus({ preventScroll: true }); }
    }, 60);
  };
  const { prefs, exportOne } = actions;
  const viewerMenu = (e) => { if (f) actions.openFrameMenu(e, f.id); };
  const [confirmClear, setConfirmClear] = useState(false);
  const st = bin.store;
  const storeLine = !st.ok ? `not saved: ${st.error}` : st.error ? st.error : `saved in this browser · ${fmtBytes(st.used)}`;

  return (
    <div className={`bin ${bin.viewerOpen ? 'frozen' : 'flow'}`}>
      <div className="bin-head">
        <div>
          <div className="bin-title">FOLIO</div>
          <div className="bin-sub">{count} renders on {bin.reels.length} reels · drag frames between reels, drag a reel by its grip to reorder · right-click for export, tags, compare and re-render</div>
          <div className="hotkeys" aria-label="Hotkeys, over a frame or reel">
            <span><b>C</b> collapse</span><span><b>L</b> lock</span><span><b>G</b> go to R#</span>
            {TAGS.map((t, i) => <span key={t.id}><b>{i + 1}</b> {t.label.toLowerCase()}</span>)}<span><b>0</b> untag</span>
            <span><b>Del</b> delete</span><span><b>Space</b> play</span><span><b>← →</b> step</span>
          </div>
          <div className={`bin-store ${!st.ok || st.error ? 'err' : ''}`}>{bin.loaded ? storeLine : 'loading saved renders…'}</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button type="button" className="lbtn" aria-pressed={bin.viewerOpen} onClick={bin.toggleViewer}>{bin.viewerOpen ? 'Hide viewer' : 'Show viewer'}</button>
          <button type="button" className="lbtn" onClick={bin.newReel}>+ New Reel</button>
          {confirmClear ? (
            <span className="confirm-clear" role="group" aria-label="Confirm clearing the bin">
              Delete all {count} renders?
              <button type="button" className="lbtn danger" onClick={() => { bin.clearBin(); setConfirmClear(false); }}>Delete all</button>
              <button type="button" className="lbtn" onClick={() => setConfirmClear(false)}>Keep</button>
            </span>
          ) : (
            <button type="button" className="lbtn" disabled={!count} onClick={() => setConfirmClear(true)}>Clear bin</button>
          )}
          <button type="button" className="lbtn gold" onClick={onClose}>Back to workspace <span className="kbd" style={{ color: 'inherit', borderColor: 'rgba(212,165,116,0.45)', marginLeft: 4 }}>~</span></button>
        </div>
      </div>

      {goTo !== null && (
        <form className="goto" onSubmit={(e) => { e.preventDefault(); jump(goTo); }}>
          <label htmlFor="goto-input">Go to R</label>
          <input id="goto-input" className="field" autoFocus inputMode="numeric" value={goTo} placeholder="12"
            onChange={(e) => setGoTo(e.target.value.replace(/\D/g, '').slice(0, 4))}
            onKeyDown={(e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); setGoTo(null); } }}
            onBlur={() => setGoTo(null)} />
          <span className="hint">Enter to jump · Esc to cancel</span>
        </form>
      )}
      {busy && <div className="bin-busy" role="status">Rendering…</div>}

      <div className="reels">
        {bin.reels.length === 0 && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: 'rgba(240,237,232,0.45)' }}>
            <div style={{ fontFamily: 'var(--serif)', fontSize: 20, marginBottom: 8 }}>No reels yet</div>
            <div style={{ fontSize: 12 }}>Render something, or create a reel.</div>
          </div>
        )}
        {bin.reels.map((r) => <ReelStrip key={r.id} reel={r} bin={bin} actions={actions} hover={hover} />)}
      </div>

      {bin.viewerOpen && !f && (
        <div className="viewer" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 13, color: 'rgba(240,237,232,0.6)' }}>Viewer is on. Click a frame, or use the arrows under a reel, to see it large here.</div>
          <button type="button" className="lbtn" onClick={bin.toggleViewer}>Hide viewer</button>
        </div>
      )}
      {multi && (
        <div className="viewer multi">
          <div className="multi-head">
            <span><Lock /> {locked.length} locked reels · position {locked.find((x) => x.reel.id === reelOf.id).index + 1}</span>
            <span className="hint">Step any locked reel and they all move. Click a frame to make it current.</span>
            <button type="button" className="lbtn" onClick={bin.toggleViewer}>Hide viewer</button>
          </div>
          <div className="multi-grid" style={{ gridTemplateColumns: `repeat(${Math.min(locked.length, 4)}, minmax(0, 1fr))` }}>
            {locked.map(({ reel, frame }) => {
              const t = tagOf(frame.tag);
              const on = frame.id === f.id;
              return (
                <figure key={reel.id} className={`multi-cell ${on ? 'on' : ''}`} style={{ borderColor: on ? '#D4A574' : t ? t.color : undefined }}
                  onContextMenu={(e) => actions.openFrameMenu(e, frame.id)}>
                  <button type="button" onClick={() => bin.selectFrame(reel.id, frame.id)} aria-label={`${reel.title}: R${pad4(frame.n)} ${VIEW_NAMES[frame.view]}`}>
                    <img src={frame.url} alt="" />
                  </button>
                  <figcaption>
                    <span className="swatch" style={{ background: reel.color }} />{reel.title} · R{pad4(frame.n)} · {frame.name || VIEW_NAMES[frame.view]}
                    {frame.size < FINAL_SIZE && <span className="proxy-badge inline">{frame.size}</span>}
                    {t && <span className="tag-chip" style={{ borderColor: t.color, color: t.color }}>{t.label}</span>}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </div>
      )}
      {bin.viewerOpen && f && !multi && (
        <div className="viewer">
          <div className="big" onContextMenu={viewerMenu}><img src={f.url} alt={`${VIEW_NAMES[f.view]} render R${pad4(f.n)}, large preview`} /></div>
          <div className="info">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <div style={{ fontSize: 15, color: '#F0EDE8', fontWeight: 500 }}>
                {label}
                {fTag && <span className="tag-chip" style={{ borderColor: fTag.color, color: fTag.color }}>{fTag.label}</span>}
              </div>
              <button type="button" className="lbtn" onClick={bin.toggleViewer}>Hide viewer</button>
            </div>
            {f.note && <div className="note-box">{f.note}</div>}
            <div className="line">{settingsLine(f)}</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button type="button" className="btn btn-primary" onClick={() => onSendToRender(f)}>Send to render</button>
              <button type="button" className="lbtn" onClick={() => exportOne(f)} title="Change format in the right-click Export menu">Export · {prefsLabel(prefs)}</button>
              <button type="button" className="lbtn" onClick={() => bin.removeFrame(f.id)}>Remove</button>
            </div>
          </div>
        </div>
      )}
      {!bin.viewerOpen && f && (
        <div className="slim">
          <img src={f.url} alt="" onContextMenu={viewerMenu} />
          <div style={{ flex: '1 1 320px', minWidth: 0 }}>
            <div style={{ fontSize: 13.5, color: '#F0EDE8', fontWeight: 500 }}>
              {label}
              {fTag && <span className="tag-chip" style={{ borderColor: fTag.color, color: fTag.color }}>{fTag.label}</span>}
            </div>
            <div className="line">{settingsLine(f)}</div>
          </div>
          <button type="button" className="lbtn gold" onClick={bin.toggleViewer}>Show viewer</button>
          <button type="button" className="btn btn-primary" onClick={() => onSendToRender(f)}>Send to render</button>
          <button type="button" className="lbtn" onClick={() => bin.removeFrame(f.id)}>Remove</button>
        </div>
      )}

      {actions.overlays}
    </div>
  );
}
