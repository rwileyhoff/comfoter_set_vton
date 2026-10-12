import { useCallback, useEffect, useRef, useState } from 'react';
import { Grip, T } from './Icons.jsx';
import { VIEW_NAMES, PARAMS, TAGS, pad4, tagOf } from '../constants.js';
import ContextMenu from './ContextMenu.jsx';
import { CompareView, InfoDialog, RenameDialog } from './BinDialogs.jsx';
import {
  FORMATS, SIZES, loadExportPrefs, saveExportPrefs, prefsLabel, convertFrame, downloadBlob,
  copyFrameImage, copyText, exportReelZip, frameRecord,
} from '../exportImage.js';

export function settingsLine(f) {
  const o = (f.settings && f.settings.options) || {};
  const opts = PARAMS.map((p) => `${p.flag} ${Number(o[p.key] ?? 0).toFixed(p.dec)}`).join(' ');
  const seed = f.settings ? f.settings.seed : '';
  return `${f.engine} · ${VIEW_NAMES[f.view]} · ${f.size}px · seed ${seed} · ${opts}`;
}

// position for a menu opened by mouse, or by the keyboard menu key / Shift+F10
function menuPoint(e) {
  if (e.clientX || e.clientY) return { x: e.clientX, y: e.clientY };
  const r = e.currentTarget.getBoundingClientRect();
  return { x: r.left + 24, y: r.top + 24 };
}

function ReelStrip({ reel, bin, openMenu }) {
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
    <div className="reel" onDragOver={onOver} onDrop={onDrop} style={{ borderColor: hot ? reel.color : undefined }}>
      <div className="reel-head" style={{ borderLeftColor: reel.color, borderBottom: reel.collapsed ? 'none' : '1px solid rgba(255,255,255,0.04)' }}
        onContextMenu={(e) => { e.preventDefault(); openMenu({ kind: 'reel', rid: reel.id, ...menuPoint(e) }); }}>
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
                    onContextMenu={(e) => { e.preventDefault(); bin.setSel(f.id); openMenu({ kind: 'frame', fid: f.id, rid: reel.id, ...menuPoint(e) }); }}
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
                <div className="pos">R{pad4(frames[cur].n)}  {cur + 1} / {frames.length}</div>
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

function ExportPanel({ prefs, setPrefs, onExport, frameLabel }) {
  const fmt = FORMATS[prefs.format];
  const seg = (list, key, cur) => (
    <div className="variant" role="group">
      {list.map(([id, label]) => (
        <button key={id} type="button" className={cur === id ? 'on' : ''} aria-pressed={cur === id}
          onClick={() => setPrefs({ ...prefs, [key]: id })}>{label}</button>
      ))}
    </div>
  );
  return (
    <div className="export-panel">
      <div className="hint">Format</div>
      {seg(Object.entries(FORMATS).map(([id, f]) => [id, f.label]), 'format', prefs.format)}
      <div className="hint">Size</div>
      {seg(SIZES.map((x) => [x.id, x.label]), 'size', prefs.size)}
      {fmt.lossy && (
        <label className="row" style={{ color: 'rgba(240,237,232,0.75)' }}>Quality
          <input type="range" min="0.5" max="1" step="0.01" value={prefs.quality}
            onChange={(e) => setPrefs({ ...prefs, quality: parseFloat(e.target.value) })} style={{ width: 110 }} />
          <span className="mono" style={{ width: 30, textAlign: 'right' }}>{Math.round(prefs.quality * 100)}</span>
        </label>
      )}
      <button type="button" className="btn btn-primary" onClick={onExport}>Export {frameLabel}</button>
      <div className="hint" style={{ fontSize: 11 }}>Also used for reel ZIPs. Full keeps the render size; JPEG fills white.</div>
    </div>
  );
}

export default function RenderBin({ bin, onClose, onSendToRender }) {
  const count = Object.keys(bin.frames).length;
  const f = bin.sel ? bin.frames[bin.sel] : null;
  const reelOf = f ? bin.reels.find((r) => r.items.includes(f.id)) : null;
  const label = f ? `R${pad4(f.n)} · ${f.name || VIEW_NAMES[f.view]} · ${reelOf ? reelOf.title : ''}` : '';
  const fTag = f ? tagOf(f.tag) : null;

  const [menu, setMenu] = useState(null);       // { kind, x, y, fid?, rid }
  const [dialog, setDialog] = useState(null);   // { kind: 'rename'|'info', fid }
  const [compare, setCompare] = useState(null); // { a, b } frame ids
  const [prefs, setPrefsState] = useState(loadExportPrefs);
  const [notice, setNotice] = useState(null);
  const noticeTimer = useRef(null);

  const setPrefs = (p) => { setPrefsState(p); saveExportPrefs(p); };
  const closeMenu = useCallback(() => setMenu(null), []);
  const say = (text, kind = 'ok') => {
    setNotice({ text, kind });
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), kind === 'err' ? 6000 : 3200);
  };
  useEffect(() => () => { if (noticeTimer.current) clearTimeout(noticeTimer.current); }, []);
  const run = async (work, done) => {
    try { const r = await work(); say(typeof done === 'function' ? done(r) : done); } catch (e) { say(e.message || String(e), 'err'); }
  };
  const live = (r) => r.items.map((id) => bin.frames[id]).filter(Boolean);
  const exportOne = (fr) => run(async () => { const { blob, name } = await convertFrame(fr, prefs); downloadBlob(blob, name); return name; }, (name) => `Exported ${name}`);
  const zipReel = (r) => {
    const frs = live(r);
    if (!frs.length) { say(`${r.title} has no frames`, 'err'); return; }
    say(`Building ${r.title} ZIP (${frs.length} frames)…`);
    run(() => exportReelZip(r, frs, prefs), (n) => `Exported ${r.title}: ${n} frames, ${prefsLabel(prefs)}, plus settings.json`);
  };
  const openViewer = (fr) => { bin.setSel(fr.id); if (!bin.viewerOpen) bin.toggleViewer(); };

  const frameItems = (fr, r) => {
    const A = bin.compareA && bin.frames[bin.compareA] ? bin.frames[bin.compareA] : null;
    const selOther = bin.sel && bin.sel !== fr.id && bin.frames[bin.sel] && (!A || A.id !== bin.sel) ? bin.frames[bin.sel] : null;
    const cmp = [];
    if (A && A.id !== fr.id) cmp.push({ label: `Compare with A · R${pad4(A.n)}`, onSelect: () => setCompare({ a: A.id, b: fr.id }) });
    if (selOther) cmp.push({ label: `Compare with selected · R${pad4(selOther.n)}`, onSelect: () => setCompare({ a: selOther.id, b: fr.id }) });
    if (!A || A.id !== fr.id) cmp.push({ label: A ? 'Set as compare A instead' : 'Set as compare A', onSelect: () => { bin.setCompareA(fr.id); say(`R${pad4(fr.n)} is compare A. Right-click another frame to compare.`); } });
    else cmp.push({ label: 'Clear compare A', onSelect: () => bin.setCompareA(null) });
    return [
      { label: 'Open in viewer', disabled: bin.viewerOpen && bin.sel === fr.id, onSelect: () => openViewer(fr) },
      { label: 'Send to render', onSelect: () => onSendToRender(fr) },
      'sep',
      {
        label: 'Export', hint: prefsLabel(prefs),
        panel: (close) => <ExportPanel prefs={prefs} setPrefs={setPrefs} frameLabel={`R${pad4(fr.n)}`} onExport={() => { close(); exportOne(fr); }} />,
      },
      { label: 'Copy image', hint: 'PNG', onSelect: () => run(() => copyFrameImage(fr), 'Image copied. Paste it into an email, deck or CADENCE.') },
      { label: 'Export reel as ZIP', hint: `${live(r).length} frames`, onSelect: () => zipReel(r) },
      'sep',
      {
        label: 'Colour tag',
        submenu: TAGS.map((t) => ({ label: t.label, swatch: t.color, checked: fr.tag === t.id, onSelect: () => bin.updateFrame(fr.id, { tag: t.id }) }))
          .concat([{ label: 'No tag', checked: !fr.tag, onSelect: () => bin.updateFrame(fr.id, { tag: '' }) }]),
      },
      { label: 'Rename / note…', onSelect: () => setDialog({ kind: 'rename', fid: fr.id }) },
      {
        label: 'Move to reel',
        submenu: bin.reels.map((x) => ({ label: x.id === r.id ? `${x.title} (here)` : x.title, swatch: x.color, disabled: x.id === r.id, onSelect: () => { bin.moveFrame(fr.id, x.id, null); say(`Moved R${pad4(fr.n)} to ${x.title}`); } })),
      },
      {
        label: 'Duplicate to reel',
        submenu: bin.reels.map((x) => ({ label: x.id === r.id ? `${x.title} (here)` : x.title, swatch: x.color, onSelect: () => { bin.duplicateFrame(fr.id, x.id); say(`Copied R${pad4(fr.n)} to ${x.title}`); } })),
      },
      'sep',
      ...cmp,
      'sep',
      { label: 'Info…', onSelect: () => setDialog({ kind: 'info', fid: fr.id }) },
      { label: 'Copy settings', hint: 'JSON', onSelect: () => run(() => copyText(JSON.stringify(frameRecord(fr, r.title), null, 2)), 'Settings copied') },
      'sep',
      { label: 'Delete', danger: true, onSelect: () => bin.removeFrame(fr.id) },
    ];
  };
  const reelItems = (r) => [
    { label: 'Export reel as ZIP', hint: `${live(r).length} · ${prefsLabel(prefs)}`, disabled: !live(r).length, onSelect: () => zipReel(r) },
    { label: r.collapsed ? 'Expand reel' : 'Collapse reel', onSelect: () => bin.toggleReel(r.id) },
    { label: 'New reel', onSelect: bin.newReel },
    'sep',
    { label: 'Delete reel', danger: true, onSelect: () => bin.deleteReel(r.id) },
  ];

  let menuItems = null;
  if (menu) {
    const r = bin.reels.find((x) => x.id === menu.rid);
    if (menu.kind === 'frame' && r && bin.frames[menu.fid]) menuItems = frameItems(bin.frames[menu.fid], r);
    if (menu.kind === 'reel' && r) menuItems = reelItems(r);
  }
  const dFrame = dialog ? bin.frames[dialog.fid] : null;
  const dReel = dFrame ? bin.reels.find((x) => x.items.includes(dFrame.id)) : null;
  const cA = compare ? bin.frames[compare.a] : null;
  const cB = compare ? bin.frames[compare.b] : null;
  const viewerMenu = (e) => {
    if (!f || !reelOf) return;
    e.preventDefault();
    setMenu({ kind: 'frame', fid: f.id, rid: reelOf.id, ...menuPoint(e) });
  };

  return (
    <div className={`bin ${bin.viewerOpen ? 'frozen' : 'flow'}`}>
      <div className="bin-head">
        <div>
          <div className="bin-title">Render Bin</div>
          <div className="bin-sub">{count} renders on {bin.reels.length} reels · every render lands here · drag frames between reels, drag a reel by its grip to reorder · right-click a frame for export, tags and compare</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <button type="button" className="lbtn" aria-pressed={bin.viewerOpen} onClick={bin.toggleViewer}>{bin.viewerOpen ? 'Hide viewer' : 'Show viewer'}</button>
          <button type="button" className="lbtn" onClick={bin.newReel}>+ New Reel</button>
          <button type="button" className="lbtn gold" onClick={onClose}>Back to workspace <span className="kbd" style={{ color: 'inherit', borderColor: 'rgba(212,165,116,0.45)', marginLeft: 4 }}>~</span></button>
        </div>
      </div>

      <div className="reels">
        {bin.reels.length === 0 && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: 'rgba(240,237,232,0.45)' }}>
            <div style={{ fontFamily: 'var(--serif)', fontSize: 20, marginBottom: 8 }}>No reels yet</div>
            <div style={{ fontSize: 12 }}>Render something, or create a reel.</div>
          </div>
        )}
        {bin.reels.map((r) => <ReelStrip key={r.id} reel={r} bin={bin} openMenu={setMenu} />)}
      </div>

      {bin.viewerOpen && !f && (
        <div className="viewer" style={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 13, color: 'rgba(240,237,232,0.6)' }}>Viewer is on. Click a frame, or use the arrows under a reel, to see it large here.</div>
          <button type="button" className="lbtn" onClick={bin.toggleViewer}>Hide viewer</button>
        </div>
      )}
      {bin.viewerOpen && f && (
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

      {notice && <div className={`bin-toast ${notice.kind}`} role="status">{notice.text}</div>}
      {menuItems && <ContextMenu x={menu.x} y={menu.y} items={menuItems} label={menu.kind === 'frame' ? 'Frame options' : 'Reel options'} onClose={closeMenu} />}
      {dialog && dFrame && dialog.kind === 'rename' && (
        <RenameDialog frame={dFrame} onSave={(p) => bin.updateFrame(dFrame.id, p)} onClose={() => setDialog(null)} />
      )}
      {dialog && dFrame && dialog.kind === 'info' && (
        <InfoDialog frame={dFrame} reelTitle={dReel ? dReel.title : ''} onClose={() => setDialog(null)}
          onCopy={(t) => run(() => copyText(t), 'Settings copied')} />
      )}
      {cA && cB && <CompareView a={cA} b={cB} onClose={() => setCompare(null)} />}
    </div>
  );
}
