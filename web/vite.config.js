import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The render API (uvicorn main:app --port 8010) is reached through /api,
// so the browser never needs CORS. FABRIC_API overrides the target.
const API = process.env.FABRIC_API || 'http://127.0.0.1:8010';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    proxy: { '/api': { target: API, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
  preview: {
    port: 5180,
    proxy: { '/api': { target: API, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
});
