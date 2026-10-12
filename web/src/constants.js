// Engines on the title bar. Only WARP is wired to /render today; the others
// show as switches so the layout is final, and refuse to render until their
// GPU worker exists.
export const ENGINES = [
  { id: 'warp', label: 'WARP', fill: '#0b52d6', ink: '#e8f0ff', line: '#5b9be6', note: 'Warp · exact print · CPU', ready: true },
  { id: 'library', label: 'LIBRARY', fill: '#19e3a6', ink: '#06382b', line: '#19e3a6', note: 'Library · base rooms · exact print', ready: false },
  { id: 'model', label: 'MODEL', fill: '#7cf3f2', ink: '#06403f', line: '#7cf3f2', note: 'Model · klein · sketch', ready: false },
];

export const MODEL_VARIANTS = [
  { id: 'm1', label: 'M1 Base', note: 'Model M1 · klein base · sketch' },
  { id: 'm2', label: 'M2 HMC', note: 'Model M2 · klein HMC · sketch' },
];

export const VIEW_ORDER = ['headon', 'quarter', 'side', 'overhead'];
export const VIEW_NAMES = { headon: 'Head-on', quarter: 'Quarter', side: 'Side', overhead: 'Overhead' };
export const VIEW_FILLS = { headon: '#2a5560', quarter: '#2f6a63', side: '#367a5d', overhead: '#3e8a55' };

// Render settings: key = RenderOptions field in render.py
export const PARAMS = [
  { key: 'fuzz', label: 'Fuzz', min: 0, max: 0.5, step: 0.01, dec: 2, flag: '--fuzz' },
  { key: 'threads', label: 'Threads', min: 20, max: 300, step: 10, dec: 0, flag: '--threads' },
  { key: 'weave_amount', label: 'Weave amount', min: 0, max: 0.5, step: 0.01, dec: 2, flag: '--weave-amount' },
  { key: 'ao', label: 'AO', min: 0, max: 1, step: 0.05, dec: 2, flag: '--ao' },
  { key: 'feather', label: 'Feather (px)', min: 0, max: 10, step: 1, dec: 0, flag: '--feather' },
  { key: 'specular', label: 'Specular', min: 0, max: 0.5, step: 0.01, dec: 2, flag: '--specular' },
  { key: 'disp_max', label: 'Disp max (px)', min: 0, max: 20, step: 1, dec: 0, flag: '--disp-max' },
];

export const DEFAULT_PARAMS = { fuzz: 0.12, threads: 140, weave_amount: 0.12, ao: 0.45, feather: 1, specular: 0.10, disp_max: 6 };

export const REEL_COLORS = ['#D4A574', '#7EB8C9', '#C9627A', '#8BC9A0', '#C9B07E', '#A78BDB'];

export const PIECE_COUNTS = {
  '3 PC': ['Comforter', 'Sham', 'Sham'],
  '5 PC': ['Comforter', 'Sham', 'Sham', 'Dec pillow', 'Dec pillow'],
  '7 PC': ['Comforter', 'Sham', 'Sham', 'Dec pillow', 'Dec pillow', 'Euro', 'Euro'],
  '9 PC': ['Comforter', 'Sham', 'Sham', 'Dec pillow', 'Dec pillow', 'Euro', 'Euro', 'Bed skirt', 'Throw'],
};

export const PALETTE_TITLES = { brief: 'BRIEF', results: 'RESULTS', bin: 'RENDER BIN' };

export const pad4 = (n) => ('000' + n).slice(-4);

// Render Bin colour tags (Flame clip colours)
export const TAGS = [
  { id: 'approved', label: 'Approved', color: '#8BC9A0' },
  { id: 'revise', label: 'Revise', color: '#E0B04A' },
  { id: 'reject', label: 'Reject', color: '#C9627A' },
];
export const tagOf = (id) => TAGS.find((t) => t.id === id) || null;
