import { useCallback, useEffect, useRef, useState } from 'react';
import { TAGS, pad4 } from '../constants.js';
import ContextMenu from './ContextMenu.jsx';
import { CompareView, InfoDialog, RenameDialog } from './BinDialogs.jsx';
import {
  FORMATS, SIZES, loadExportPrefs, saveExportPrefs, prefsLabel, convertFrame, downloadBlob,
  copyFrameImage, copyText, exportReelZip, frameRecord,
} from '../exportImage.js';

// One right-click menu for a render frame, wherever it appears: the FOLIO
// reels, the bin viewer, the workspace canvas and the view tiles.

// position for a menu opened by mouse, or by the keyboard menu key / Shift+F10
export function menuPoint(e) {
  if (e.clientX || e.clientY) return { x: e.clientX, y: e.clientY };
  const r = e.currentTarget.getBoundingClientRect();
  return { x: r.left + 24, y: r.top + 24 };
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

// where: 'bin' (inside the FOLIO) or 'workspace' (canvas / view tiles)
export const FINAL_SIZE = 2048;

export function useFrameActions({ bin, where, onSendToRender, onOpenInBin, onRerender }) {
  const [menu, setMenu] = useState(null);       // { kind, x, y, fid?, rid }
  const [dialog, setDialog] = useState(null);   // { kind: 'rename'|'info', fid }
  const [compare, setCompare] = useState(null); // { a, b } frame ids
  const [prefs, setPrefsState] = useState(loadExportPrefs);
  const [notice, setNotice] = useState(null);
  const noticeTimer = useRef(null);

  const setPrefs = (p) => { setPrefsState(p); saveExportPrefs(p); };
  const closeMenu = useCallback(() => setMenu(null), []);
  const say = useCallback((text, kind = 'ok') => {
    setNotice({ text, kind });
    if (noticeTimer.current) clearTimeout(noticeTimer.current);
    noticeTimer.current = setTimeout(() => setNotice(null), kind === 'err' ? 6000 : 3200);
  }, []);
  useEffect(() => () => { if (noticeTimer.current) clearTimeout(noticeTimer.current); }, []);

  const run = async (work, done) => {
    try { const r = await work(); say(typeof done === 'function' ? done(r) : done); } catch (e) { say(e.message || String(e), 'err'); }
  };
  const live = (r) => r.items.map((id) => bin.frames[id]).filter(Boolean);
  const reelOf = (fid) => bin.reels.find((r) => r.items.includes(fid)) || null;

  const exportOne = (fr) => run(async () => { const { blob, name } = await convertFrame(fr, prefs); downloadBlob(blob, name); return name; }, (name) => `Exported ${name}`);
  const zipReel = (r) => {
    const frs = live(r);
    if (!frs.length) { say(`${r.title} has no frames`, 'err'); return; }
    say(`Building ${r.title} ZIP (${frs.length} frames)…`);
    run(() => exportReelZip(r, frs, prefs), (n) => `Exported ${r.title}: ${n} frames, ${prefsLabel(prefs)}, plus settings.json`);
  };

  const rerender = (frs) => {
    const todo = frs.filter((f) => f.size < FINAL_SIZE);
    if (!todo.length) { say('Already full size'); return; }
    if (!onRerender) return;
    say(`Re-rendering ${todo.length} at ${FINAL_SIZE}…`);
    run(() => onRerender(todo, FINAL_SIZE), (n) => `${n} render${n === 1 ? '' : 's'} now ${FINAL_SIZE} px, same seed and settings`);
  };

  const frameItems = (fr, r) => {
    const A = bin.compareA && bin.frames[bin.compareA] ? bin.frames[bin.compareA] : null;
    const selOther = bin.sel && bin.sel !== fr.id && bin.frames[bin.sel] && (!A || A.id !== bin.sel) ? bin.frames[bin.sel] : null;
    const cmp = [];
    if (A && A.id !== fr.id) cmp.push({ label: `Compare with A · R${pad4(A.n)}`, onSelect: () => setCompare({ a: A.id, b: fr.id }) });
    if (selOther && where === 'bin') cmp.push({ label: `Compare with selected · R${pad4(selOther.n)}`, onSelect: () => setCompare({ a: selOther.id, b: fr.id }) });
    if (!A || A.id !== fr.id) cmp.push({ label: A ? 'Set as compare A instead' : 'Set as compare A', onSelect: () => { bin.setCompareA(fr.id); say(`R${pad4(fr.n)} is compare A. Right-click another render to compare.`); } });
    else cmp.push({ label: 'Clear compare A', onSelect: () => bin.setCompareA(null) });

    const open = where === 'bin'
      ? [{ label: 'Open in viewer', disabled: bin.viewerOpen && bin.sel === fr.id, onSelect: () => { bin.setSel(fr.id); if (!bin.viewerOpen) bin.toggleViewer(); } }]
      : [{ label: 'Show in FOLIO', hint: '~', onSelect: () => onOpenInBin(fr) }];
    const send = where === 'bin' ? [{ label: 'Send to render', onSelect: () => onSendToRender(fr) }] : [];

    const proxy = fr.size < FINAL_SIZE;
    return [
      ...open,
      ...send,
      { label: `Re-render at ${FINAL_SIZE}`, hint: proxy ? `${fr.size} → ${FINAL_SIZE}` : 'already full', disabled: !proxy || !onRerender, onSelect: () => rerender([fr]) },
      'sep',
      {
        label: 'Export', hint: prefsLabel(prefs),
        panel: (close) => <ExportPanel prefs={prefs} setPrefs={setPrefs} frameLabel={`R${pad4(fr.n)}`} onExport={() => { close(); exportOne(fr); }} />,
      },
      { label: 'Copy image', hint: 'PNG', onSelect: () => run(() => copyFrameImage(fr), 'Image copied. Paste it into an email, deck or CADENCE.') },
      ...(r ? [{ label: `Export reel as ZIP`, hint: `${r.title} · ${live(r).length}`, onSelect: () => zipReel(r) }] : []),
      'sep',
      {
        label: 'Colour tag',
        submenu: TAGS.map((t, i) => ({ label: t.label, hint: String(i + 1), swatch: t.color, checked: fr.tag === t.id, onSelect: () => bin.updateFrame(fr.id, { tag: t.id }) }))
          .concat([{ label: 'No tag', hint: '0', checked: !fr.tag, onSelect: () => bin.updateFrame(fr.id, { tag: '' }) }]),
      },
      { label: 'Rename / note…', onSelect: () => setDialog({ kind: 'rename', fid: fr.id }) },
      {
        label: 'Move to reel',
        submenu: bin.reels.map((x) => ({ label: r && x.id === r.id ? `${x.title} (here)` : x.title, swatch: x.color, disabled: !!r && x.id === r.id, onSelect: () => { bin.moveFrame(fr.id, x.id, null); say(`Moved R${pad4(fr.n)} to ${x.title}`); } })),
      },
      {
        label: 'Duplicate to reel',
        submenu: bin.reels.map((x) => ({ label: r && x.id === r.id ? `${x.title} (here)` : x.title, swatch: x.color, onSelect: () => { bin.duplicateFrame(fr.id, x.id); say(`Copied R${pad4(fr.n)} to ${x.title}`); } })),
      },
      'sep',
      ...cmp,
      'sep',
      { label: 'Info…', onSelect: () => setDialog({ kind: 'info', fid: fr.id }) },
      { label: 'Copy settings', hint: 'JSON', onSelect: () => run(() => copyText(JSON.stringify(frameRecord(fr, r ? r.title : ''), null, 2)), 'Settings copied') },
      'sep',
      { label: 'Delete', hint: where === 'bin' ? 'Del' : undefined, danger: true, onSelect: () => { bin.removeFrame(fr.id); say(`Deleted R${pad4(fr.n)}`); } },
    ];
  };
  const reelItems = (r) => [
    { label: r.locked ? 'Unlock reel' : 'Lock reel', hint: 'L', onSelect: () => bin.toggleLock(r.id) },
    {
      label: `Re-render reel at ${FINAL_SIZE}`,
      hint: `${live(r).filter((f) => f.size < FINAL_SIZE).length} proxies`,
      disabled: !onRerender || !live(r).some((f) => f.size < FINAL_SIZE),
      onSelect: () => rerender(live(r)),
    },
    'sep',
    { label: 'Export reel as ZIP', hint: `${live(r).length} · ${prefsLabel(prefs)}`, disabled: !live(r).length, onSelect: () => zipReel(r) },
    { label: r.collapsed ? 'Expand reel' : 'Collapse reel', hint: 'C', onSelect: () => bin.toggleReel(r.id) },
    { label: 'New reel', onSelect: bin.newReel },
    'sep',
    { label: 'Delete reel', danger: true, onSelect: () => bin.deleteReel(r.id) },
  ];

  // open helpers, for onContextMenu handlers
  const openFrameMenu = (e, fid) => {
    e.preventDefault();
    if (!bin.frames[fid]) return;
    if (where === 'bin') bin.setSel(fid);
    const r = reelOf(fid);
    setMenu({ kind: 'frame', fid, rid: r ? r.id : null, ...menuPoint(e) });
  };
  const openReelMenu = (e, rid) => { e.preventDefault(); setMenu({ kind: 'reel', rid, ...menuPoint(e) }); };

  let menuItems = null;
  if (menu) {
    const r = menu.rid ? bin.reels.find((x) => x.id === menu.rid) : reelOf(menu.fid);
    if (menu.kind === 'frame' && bin.frames[menu.fid]) menuItems = frameItems(bin.frames[menu.fid], r);
    if (menu.kind === 'reel' && r) menuItems = reelItems(r);
  }
  const dFrame = dialog ? bin.frames[dialog.fid] : null;
  const dReel = dFrame ? reelOf(dFrame.id) : null;
  const cA = compare ? bin.frames[compare.a] : null;
  const cB = compare ? bin.frames[compare.b] : null;

  const overlays = (
    <>
      {notice && <div className={`bin-toast ${notice.kind}`} role="status">{notice.text}</div>}
      {menuItems && <ContextMenu x={menu.x} y={menu.y} items={menuItems} label={menu.kind === 'frame' ? 'Render options' : 'Reel options'} onClose={closeMenu} />}
      {dialog && dFrame && dialog.kind === 'rename' && (
        <RenameDialog frame={dFrame} onSave={(p) => bin.updateFrame(dFrame.id, p)} onClose={() => setDialog(null)} />
      )}
      {dialog && dFrame && dialog.kind === 'info' && (
        <InfoDialog frame={dFrame} reelTitle={dReel ? dReel.title : ''} onClose={() => setDialog(null)}
          onCopy={(t) => run(() => copyText(t), 'Settings copied')} />
      )}
      {cA && cB && <CompareView a={cA} b={cB} onClose={() => setCompare(null)} />}
    </>
  );

  return { openFrameMenu, openReelMenu, exportOne, prefs, say, overlays, rerender, run };
}
