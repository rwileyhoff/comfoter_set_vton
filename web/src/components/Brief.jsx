import { useRef } from 'react';
import { PARAMS, PIECE_COUNTS, VIEW_NAMES } from '../constants.js';

const TABS = [['set', 'Set'], ['fabrics', 'Fabrics'], ['room', 'Room'], ['render', 'Render'], ['tune', 'Fine tune']];

function FabricSlot({ letter, fabric, active, onPick, onFile, onInches, onClear, optional }) {
  const input = useRef(null);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className={`section-head ${fabric ? '' : 'muted'}`}>Fabric {letter}{fabric ? ` — ${fabric.name}` : optional ? ' — optional' : ''}</div>
      <button type="button" className={`swatch-drop ${active ? 'active' : ''}`} onClick={() => { onPick(); input.current && input.current.click(); }}
        aria-label={`Fabric ${letter} repeat: click to upload, or click then paste`}>
        {fabric ? <img src={fabric.url} alt={`Fabric ${letter} repeat`} /> : <span>Click to upload, or click here then paste (Ctrl+V)</span>}
      </button>
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp" style={{ display: 'none' }}
        onChange={(e) => { const f = e.target.files && e.target.files[0]; if (f) onFile(f); e.target.value = ''; }} />
      {fabric && (
        <>
          <label className="row">File width (in)
            <input className="field" style={{ width: 90, textAlign: 'right' }} type="number" min="0.1" step="0.1" value={fabric.inches}
              onChange={(e) => onInches(e.target.value)} />
          </label>
          <div className="hint mono">{fabric.w} × {fabric.h} px · {(fabric.w / Math.max(0.1, Number(fabric.inches) || 0.1)).toFixed(2)} ppi</div>
          {optional && <button type="button" className="btn" style={{ alignSelf: 'flex-start' }} onClick={onClear}>Remove fabric {letter}</button>}
        </>
      )}
    </div>
  );
}

export default function Brief(props) {
  const {
    tab, setTab, pieceCount, fabricA, fabricB, activeSlot, setActiveSlot, setFabricFile, setFabricInches, clearFabricB,
    shamsFabric, setShamsFabric, rooms, viewsChecked, toggleView, params, setParam, resetParams, command, selectedView,
  } = props;
  const pieces = PIECE_COUNTS[pieceCount] || PIECE_COUNTS['3 PC'];
  const views = rooms ? rooms.views : [];

  return (
    <div>
      <div className="tabs" role="tablist" aria-label="Brief">
        {TABS.map(([id, label]) => (
          <button key={id} type="button" role="tab" aria-selected={tab === id} className={`tab ${tab === id ? 'on' : ''}`} onClick={() => setTab(id)}>{label}</button>
        ))}
      </div>

      {tab === 'set' && (
        <div className="pane">
          <div className="section-head">Set — {pieceCount}</div>
          <div className="hint">Piece count loads the default pieces. The current room shows a comforter and 2 shams.</div>
          <div className="grid3">
            <div className="hint">Piece</div><div className="hint">Size</div><div className="hint">Fabric</div>
            {pieces.map((p, i) => (
              <div key={`${p}${i}`} style={{ display: 'contents' }}>
                <div>{p}</div>
                <div className="mono">{p === 'Comforter' ? '90 × 90' : p === 'Sham' ? '26 × 20' : '—'}</div>
                <div className="mono">{p === 'Sham' ? shamsFabric : 'A'}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'fabrics' && (
        <div className="pane">
          <FabricSlot letter="A" fabric={fabricA} active={activeSlot === 'A'} onPick={() => setActiveSlot('A')}
            onFile={(f) => setFabricFile('A', f)} onInches={(v) => setFabricInches('A', v)} />
          <div style={{ borderTop: '1px solid var(--line)' }} />
          <FabricSlot letter="B" optional fabric={fabricB} active={activeSlot === 'B'} onPick={() => setActiveSlot('B')}
            onFile={(f) => setFabricFile('B', f)} onInches={(v) => setFabricInches('B', v)} onClear={clearFabricB} />
          <label className="row">Shams fabric
            <select className="select" value={shamsFabric} onChange={(e) => setShamsFabric(e.target.value)}>
              <option value="A">A</option>
              <option value="B" disabled={!fabricB}>B</option>
            </select>
          </label>
        </div>
      )}

      {tab === 'room' && (
        <div className="pane">
          <div className="section-head">Room — {rooms ? rooms.name : 'loading'}</div>
          <div className="hint">Shows a comforter and 2 shams. Tick the views to render.</div>
          {views.map((v) => (
            <label key={v.view} className="check">
              <input type="checkbox" checked={viewsChecked.includes(v.view)} onChange={() => toggleView(v.view)} />
              {VIEW_NAMES[v.view]}
              <span className="hint mono" style={{ marginLeft: 'auto' }}>{v.mode} · {v.gridded.join(', ') || 'no grid'}</span>
            </label>
          ))}
        </div>
      )}

      {tab === 'render' && (
        <div className="pane">
          <div className="section-head">Render — settings</div>
          <div className="params">
            {PARAMS.map((p) => (
              <div key={p.key} style={{ display: 'contents' }}>
                <label htmlFor={`p-${p.key}`} style={{ color: '#c8c8c8' }}>{p.label}</label>
                <input id={`p-${p.key}`} type="range" min={p.min} max={p.max} step={p.step} value={params[p.key]}
                  onChange={(e) => setParam(p.key, parseFloat(e.target.value))} style={{ width: 96 }} />
                <div className="num">{Number(params[p.key]).toFixed(p.dec)}</div>
              </div>
            ))}
          </div>
          <button type="button" className="btn" style={{ alignSelf: 'flex-start' }} onClick={resetParams}>Reset to working settings</button>
          <div className="hint">Command</div>
          <div className="cmd">{command}</div>
        </div>
      )}

      {tab === 'tune' && (
        <div className="pane">
          <div className="section-head">05 Fine tune — {VIEW_NAMES[selectedView] || ''}</div>
          <div className="hint">Reassign a piece's fabric, then re-render this view.</div>
          <label className="row">Shams fabric
            <select className="select" value={shamsFabric} onChange={(e) => setShamsFabric(e.target.value)}>
              <option value="A">A</option>
              <option value="B" disabled={!fabricB}>B</option>
            </select>
          </label>
          <div style={{ borderTop: '1px solid var(--line)', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div className="hint">CONSTRUCTION · MODEL, LATER</div>
            {['Edge', 'Flange', 'Embroidery colour'].map((x) => (
              <label key={x} className="row" style={{ color: 'var(--ink-3)' }}>{x}
                <select className="select" disabled><option>—</option></select>
              </label>
            ))}
          </div>
          <button type="button" className="btn btn-primary" style={{ alignSelf: 'flex-start' }} onClick={props.renderSelected}>
            Re-render {VIEW_NAMES[selectedView] || 'view'}
          </button>
        </div>
      )}
    </div>
  );
}
