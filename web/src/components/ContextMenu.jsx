import { useEffect, useLayoutEffect, useRef, useState } from 'react';

// Right-click menu for the Render Bin. Items:
//   { label, onSelect, hint, danger, disabled, swatch, checked }
//   { label, submenu: [items] }          nested list
//   { label, panel: (close) => node }    nested custom panel (export settings)
//   'sep'
// Keyboard: arrows move, Right/Enter opens a submenu, Left/Esc backs out.

const W = 236;

function useClampedPosition(ref, x, y, deps) {
  const [pos, setPos] = useState({ left: x, top: y });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const left = Math.max(8, Math.min(x, window.innerWidth - r.width - 8));
    const top = Math.max(8, Math.min(y, window.innerHeight - r.height - 8));
    setPos({ left, top });
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return pos;
}

function focusables(list) {
  return list ? Array.from(list.querySelectorAll(':scope > li > [role^="menuitem"]:not([disabled])')) : [];
}

function MenuList({ items, close, label, flip, depth, onBack }) {
  const ref = useRef(null);
  const [open, setOpen] = useState(-1);
  const [shift, setShift] = useState(0);

  useEffect(() => {
    if (depth > 0) { const f = focusables(ref.current)[0]; if (f) f.focus(); else { const p = ref.current && ref.current.querySelector('button,input'); if (p) p.focus(); } }
  }, [depth]);
  useLayoutEffect(() => {
    if (depth === 0 || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const over = r.bottom - (window.innerHeight - 8);
    if (over > 0) setShift(-Math.min(over, r.top - 8));
  }, [depth]);

  const onKey = (e) => {
    const list = focusables(ref.current);
    const i = list.indexOf(document.activeElement);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (i < 0 && document.activeElement && ref.current.contains(document.activeElement) && !document.activeElement.matches('[role^="menuitem"]')) return;
      e.preventDefault(); e.stopPropagation();
      const n = list.length; if (!n) return;
      const next = e.key === 'ArrowDown' ? (i + 1) % n : (i - 1 + n) % n;
      list[next].focus();
    } else if (e.key === 'ArrowLeft' && depth > 0) {
      e.preventDefault(); e.stopPropagation(); onBack();
    } else if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault(); e.stopPropagation();
      const t = e.key === 'Home' ? list[0] : list[list.length - 1]; if (t) t.focus();
    }
  };

  let k = -1;
  return (
    <ul ref={ref} role="menu" aria-label={label} className={`cmenu ${depth ? 'sub' : ''} ${flip ? 'flip' : ''}`}
      style={depth ? { top: shift - 5 } : undefined} onKeyDown={onKey}>
      {items.map((it, idx) => {
        if (it === 'sep') return <li key={`s${idx}`} role="separator" className="cmenu-sep" />;
        k += 1;
        const my = k;
        const nested = it.submenu || it.panel;
        const trigger = (e) => {
          if (it.disabled) return;
          if (nested) { setOpen(my); return; }
          close(); it.onSelect && it.onSelect(e);
        };
        return (
          <li key={`${idx}-${it.label}`} role="none" className="cmenu-li" onMouseEnter={() => setOpen(nested ? my : -1)}>
            <button type="button" role={it.checked !== undefined ? 'menuitemradio' : 'menuitem'}
              aria-checked={it.checked !== undefined ? !!it.checked : undefined}
              aria-haspopup={nested ? 'menu' : undefined} aria-expanded={nested ? open === my : undefined}
              disabled={it.disabled} className={`cmenu-item ${it.danger ? 'danger' : ''} ${open === my ? 'open' : ''}`}
              onClick={trigger}
              onKeyDown={(e) => { if (nested && (e.key === 'ArrowRight' || e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); e.stopPropagation(); setOpen(my); } }}>
              <span className="cmenu-mark" aria-hidden="true">
                {it.swatch ? <span className="cmenu-swatch" style={{ background: it.swatch }} /> : it.checked ? '✓' : ''}
              </span>
              <span className="cmenu-label">{it.label}</span>
              {it.hint && <span className="cmenu-hint">{it.hint}</span>}
              {nested && <span className="cmenu-arrow" aria-hidden="true">▸</span>}
            </button>
            {nested && open === my && (it.submenu ? (
              <MenuList items={it.submenu} close={close} label={it.label} flip={flip} depth={depth + 1}
                onBack={() => { setOpen(-1); const t = focusables(ref.current)[my]; if (t) t.focus(); }} />
            ) : (
              <div className={`cmenu sub panel ${flip ? 'flip' : ''}`} role="group" aria-label={it.label} style={{ top: -5 }}
                onKeyDown={(e) => { if (e.key === 'ArrowLeft' && e.target.tagName !== 'INPUT') { e.preventDefault(); e.stopPropagation(); setOpen(-1); const t = focusables(ref.current)[my]; if (t) t.focus(); } }}>
                <PanelFocus>{it.panel(close)}</PanelFocus>
              </div>
            ))}
          </li>
        );
      })}
    </ul>
  );
}

function PanelFocus({ children }) {
  const ref = useRef(null);
  useEffect(() => { const b = ref.current && ref.current.querySelector('button,input'); if (b) b.focus(); }, []);
  return <div ref={ref}>{children}</div>;
}

export default function ContextMenu({ x, y, items, label, onClose }) {
  const ref = useRef(null);
  const pos = useClampedPosition(ref, x, y, [x, y]);
  const flip = pos.left + W * 2 + 16 > window.innerWidth;
  const back = useRef(document.activeElement);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    const first = ref.current && ref.current.querySelector('[role^="menuitem"]:not([disabled])');
    if (first) first.focus();
    const restore = back.current;
    const close = () => closeRef.current();
    const down = (e) => { if (ref.current && !ref.current.contains(e.target)) close(); };
    const key = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); close(); }
    };
    const away = () => close();
    document.addEventListener('mousedown', down, true);
    window.addEventListener('keydown', key, true);
    window.addEventListener('resize', away);
    window.addEventListener('blur', away);
    return () => {
      document.removeEventListener('mousedown', down, true);
      window.removeEventListener('keydown', key, true);
      window.removeEventListener('resize', away);
      window.removeEventListener('blur', away);
      if (restore && restore.focus && document.contains(restore)) restore.focus();
    };
  }, []);

  return (
    <div ref={ref} className="cmenu-root" style={{ left: pos.left, top: pos.top }} onContextMenu={(e) => e.preventDefault()}
      onKeyDown={(e) => { if (e.key === 'Tab' && !(ref.current && ref.current.querySelector('.panel'))) e.preventDefault(); e.stopPropagation(); }}>
      <MenuList items={items} close={() => closeRef.current()} label={label} flip={flip} depth={0} onBack={() => {}} />
    </div>
  );
}
