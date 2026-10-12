import { useEffect, useMemo, useRef, useState } from 'react';
import { PARAMS, VIEW_NAMES, pad4, tagOf } from '../constants.js';
import { frameRecord } from '../exportImage.js';

// Modal shell: Esc closes, focus moves in and returns on close, keys stay inside.
export function Modal({ title, onClose, children, wide, className }) {
  const ref = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const back = document.activeElement;
    const first = ref.current && ref.current.querySelector('[data-autofocus], input, textarea, button');
    if (first) first.focus();
    const key = (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); closeRef.current(); } };
    window.addEventListener('keydown', key, true);
    return () => { window.removeEventListener('keydown', key, true); if (back && back.focus && document.contains(back)) back.focus(); };
  }, []);
  const trap = (e) => {
    e.stopPropagation(); // keep ~ and arrows away from the app shortcuts
    if (e.key !== 'Tab' || !ref.current) return;
    const els = Array.from(ref.current.querySelectorAll('button:not([disabled]), input, textarea, select, [tabindex="0"]'));
    if (!els.length) return;
    const a = els[0]; const z = els[els.length - 1];
    if (e.shiftKey && document.activeElement === a) { e.preventDefault(); z.focus(); }
    else if (!e.shiftKey && document.activeElement === z) { e.preventDefault(); a.focus(); }
  };
  return (
    <div className="modal-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={ref} className={`modal ${wide ? 'wide' : ''} ${className || ''}`} role="dialog" aria-modal="true" aria-label={title} onKeyDown={trap}>
        <div className="modal-head">
          <div className="modal-title">{title}</div>
          <button type="button" className="plain" aria-label="Close" style={{ fontSize: 18 }} onClick={onClose}>×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function RenameDialog({ frame, onSave, onClose }) {
  const [name, setName] = useState(frame.name || '');
  const [note, setNote] = useState(frame.note || '');
  const save = (e) => { e.preventDefault(); onSave({ name: name.trim(), note: note.trim() }); onClose(); };
  return (
    <Modal title={`Rename R${pad4(frame.n)}`} onClose={onClose}>
      <form className="modal-body" onSubmit={save}>
        <label className="mfield">Name
          <input className="field" data-autofocus value={name} placeholder={VIEW_NAMES[frame.view]} onChange={(e) => setName(e.target.value)} maxLength={60} />
        </label>
        <label className="mfield">Note
          <textarea className="field" rows={3} value={note} placeholder="e.g. Jesse's pick, shams read too large" onChange={(e) => setNote(e.target.value)} maxLength={400} />
        </label>
        <div className="hint">The name shows on the frame and goes into exported file names.</div>
        <div className="modal-foot">
          <button type="button" className="lbtn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-primary">Save</button>
        </div>
      </form>
    </Modal>
  );
}

export function InfoDialog({ frame, reelTitle, onCopy, onClose }) {
  const rec = frameRecord(frame, reelTitle);
  const o = (frame.settings && frame.settings.options) || {};
  const tag = tagOf(frame.tag);
  const rows = [
    ['Frame', `R${pad4(frame.n)}${frame.dupOf ? ' (copy)' : ''}`],
    ['Name', frame.name || '—'],
    ['Reel', reelTitle || '—'],
    ['Tag', tag ? tag.label : '—'],
    ['View', VIEW_NAMES[frame.view]],
    ['Engine', String(frame.engine || '').toUpperCase()],
    ['Output', `${frame.size} px`],
    ['Fabric', frame.fabric || '—'],
    ['Seed', frame.settings ? String(frame.settings.seed) : '—'],
    ['Render time', frame.seconds != null ? `${Number(frame.seconds).toFixed(1)} s` : '—'],
  ];
  return (
    <Modal title={`Info · R${pad4(frame.n)}`} onClose={onClose} wide>
      <div className="modal-body">
        <div className="info-grid">
          {rows.map(([k, v]) => (<div key={k} style={{ display: 'contents' }}><div className="hint">{k}</div><div className="mono">{v}</div></div>))}
          {PARAMS.map((p) => (
            <div key={p.key} style={{ display: 'contents' }}><div className="hint">{p.label}</div><div className="mono">{o[p.key] != null ? Number(o[p.key]).toFixed(p.dec) : '—'}</div></div>
          ))}
        </div>
        {frame.note && <div className="note-box">{frame.note}</div>}
        <details>
          <summary className="hint" style={{ cursor: 'pointer' }}>Full settings JSON</summary>
          <pre className="cmd" style={{ maxHeight: 220, overflow: 'auto', margin: '8px 0 0' }}>{JSON.stringify(rec, null, 2)}</pre>
        </details>
        <div className="modal-foot">
          <button type="button" className="lbtn" onClick={() => onCopy(JSON.stringify(rec, null, 2))}>Copy settings</button>
          <button type="button" className="btn btn-primary" onClick={onClose}>Done</button>
        </div>
      </div>
    </Modal>
  );
}

// what differs between two frames, for the compare footer
function diffs(a, b) {
  const out = [];
  const add = (label, x, y) => { if (String(x) !== String(y)) out.push(`${label} ${x} → ${y}`); };
  add('view', VIEW_NAMES[a.view], VIEW_NAMES[b.view]);
  add('size', a.size, b.size);
  add('fabric', a.fabric || '—', b.fabric || '—');
  add('seed', a.settings ? a.settings.seed : '—', b.settings ? b.settings.seed : '—');
  const oa = (a.settings && a.settings.options) || {}; const ob = (b.settings && b.settings.options) || {};
  PARAMS.forEach((p) => {
    if (oa[p.key] == null && ob[p.key] == null) return;
    add(p.label.replace(/ \(.*\)$/, '').toLowerCase(), Number(oa[p.key] ?? 0).toFixed(p.dec), Number(ob[p.key] ?? 0).toFixed(p.dec));
  });
  const sa = a.settings || {}; const sb = b.settings || {};
  const inch = (s, k) => {
    const x = k === 'a' ? s.fabric_a : s.pieces && s.pieces[k];
    return x && x.file_inches != null ? `${x.file_inches}in` : '—';
  };
  add('fabric A width', inch(sa, 'a'), inch(sb, 'a'));
  add('shams fabric width', inch(sa, 'shams'), inch(sb, 'shams'));
  return out;
}

export function CompareView({ a: a0, b: b0, onClose }) {
  const [pair, setPair] = useState([a0, b0]);
  const [a, b] = pair;
  const [pos, setPos] = useState(50);
  const [mode, setMode] = useState('wipe');
  const box = useRef(null);
  const dragging = useRef(false);
  const list = useMemo(() => diffs(a, b), [a, b]);
  const name = (f) => `R${pad4(f.n)} · ${f.name || VIEW_NAMES[f.view]}`;

  const fromX = (clientX) => {
    const r = box.current.getBoundingClientRect();
    setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  const onKey = (e) => {
    if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && mode === 'wipe' && e.target.tagName !== 'INPUT') {
      e.preventDefault();
      const d = (e.shiftKey ? 10 : 2) * (e.key === 'ArrowRight' ? 1 : -1);
      setPos((p) => Math.max(0, Math.min(100, p + d)));
    }
  };

  return (
    <Modal title="Compare A / B" onClose={onClose} className="compare">
      <div className="modal-body" onKeyDown={onKey}>
        <div className="compare-bar">
          <div className="variant" role="group" aria-label="Compare mode">
            <button type="button" className={mode === 'wipe' ? 'on' : ''} aria-pressed={mode === 'wipe'} onClick={() => setMode('wipe')}>Wipe</button>
            <button type="button" className={mode === 'side' ? 'on' : ''} aria-pressed={mode === 'side'} onClick={() => setMode('side')}>Side by side</button>
          </div>
          <button type="button" className="lbtn" onClick={() => setPair([b, a])}>Swap A / B</button>
          {mode === 'wipe' && (
            <label className="row" style={{ gap: 8, color: 'rgba(240,237,232,0.7)' }}>Wipe
              <input type="range" min="0" max="100" step="0.5" value={pos} onChange={(e) => setPos(parseFloat(e.target.value))} style={{ width: 160 }} />
              <span className="mono" style={{ width: 40 }}>{Math.round(pos)}%</span>
            </label>
          )}
          <span className="hint" style={{ marginLeft: 'auto' }}>Drag the line, or ← → (Shift for big steps). Esc closes.</span>
        </div>

        {mode === 'wipe' ? (
          <div ref={box} className="wipe" tabIndex={0} aria-label={`Wipe between A ${name(a)} and B ${name(b)}, ${Math.round(pos)} percent A`}
            onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); fromX(e.clientX); }}
            onPointerMove={(e) => { if (dragging.current) fromX(e.clientX); }}
            onPointerUp={() => { dragging.current = false; }} onPointerCancel={() => { dragging.current = false; }}>
            <img src={b.url} alt="" draggable={false} />
            <img src={a.url} alt="" draggable={false} style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
            <div className="wipe-line" style={{ left: `${pos}%` }}><span /></div>
            <div className="wipe-tag left">A · {name(a)}</div>
            <div className="wipe-tag right">B · {name(b)}</div>
          </div>
        ) : (
          <div className="side">
            {[['A', a], ['B', b]].map(([k, f]) => (
              <figure key={k}>
                <img src={f.url} alt={`${k}: ${name(f)}`} />
                <figcaption>{k} · {name(f)}</figcaption>
              </figure>
            ))}
          </div>
        )}

        <div className="compare-diff">
          <span className="hint">Differences A → B:</span>{' '}
          <span className="mono">{list.length ? list.join(' · ') : 'none in the saved settings'}</span>
        </div>
      </div>
    </Modal>
  );
}
