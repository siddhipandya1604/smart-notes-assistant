import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and proxies API calls to the FastAPI server.
// Point it elsewhere with:  VITE_API_TARGET=http://127.0.0.1:8765 npm run dev
const target = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": target, "/classify": target } },
});
