// Small stroke/fill icons. Decorative: always aria-hidden; buttons carry labels.
const P = { 'aria-hidden': true };

export const Grip = () => (
  <svg {...P} width="12" height="16" viewBox="0 0 12 16" fill="#6c6c6c">
    <circle cx="3" cy="3" r="1.4" /><circle cx="9" cy="3" r="1.4" /><circle cx="3" cy="8" r="1.4" />
    <circle cx="9" cy="8" r="1.4" /><circle cx="3" cy="13" r="1.4" /><circle cx="9" cy="13" r="1.4" />
  </svg>
);
export const FloatOut = () => (
  <svg {...P} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="7" width="14" height="14" rx="2" /><path d="M10 3h11v11" /><path d="M21 3l-9 9" />
  </svg>
);
export const DockIn = () => (
  <svg {...P} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18" />
  </svg>
);
export const Pin = () => (
  <svg {...P} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 17v5" /><path d="M9 3h6l-1 6 4 4H6l4-4z" />
  </svg>
);
export const T = {
  first: () => <svg {...P} width="16" height="14" viewBox="0 0 16 14" fill="currentColor"><rect x="1" y="2" width="2" height="10" /><path d="M9 2L3.5 7L9 12z" /><path d="M15 2L9.5 7L15 12z" /></svg>,
  back: () => <svg {...P} width="16" height="14" viewBox="0 0 16 14" fill="currentColor"><path d="M8 2L2 7L8 12z" /><path d="M15 2L9 7L15 12z" /></svg>,
  prev: () => <svg {...P} width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="2" y="2" width="2" height="10" /><path d="M12 2L5 7L12 12z" /></svg>,
  play: () => <svg {...P} width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M3 1.5L12 7L3 12.5z" /></svg>,
  pause: () => <svg {...P} width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="3" y="2" width="3" height="10" /><rect x="8" y="2" width="3" height="10" /></svg>,
  next: () => <svg {...P} width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><path d="M2 2L9 7L2 12z" /><rect x="10" y="2" width="2" height="10" /></svg>,
  fwd: () => <svg {...P} width="16" height="14" viewBox="0 0 16 14" fill="currentColor"><path d="M1 2L7 7L1 12z" /><path d="M8 2L14 7L8 12z" /></svg>,
  last: () => <svg {...P} width="16" height="14" viewBox="0 0 16 14" fill="currentColor"><path d="M1 2L6.5 7L1 12z" /><path d="M7 2L12.5 7L7 12z" /><rect x="13" y="2" width="2" height="10" /></svg>,
};
export const Lock = ({ open }) => (
  <svg {...P} width="11" height="12" viewBox="0 0 24 26" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="11" width="16" height="12" rx="2" />{open ? <path d="M8 11V7a4 4 0 0 1 7.5-2" /> : <path d="M8 11V7a4 4 0 0 1 8 0v4" />}
  </svg>
);
