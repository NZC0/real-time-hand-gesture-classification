import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { copyFileSync, mkdirSync, existsSync } from "fs";

/**
 * Copy ONNX Runtime Web WASM files to public/ so they are served
 * at the root path expected by ort.env.wasm.wasmPaths = "/".
 */
function copyOrtWasm() {
  return {
    name: "copy-ort-wasm",
    buildStart() {
      const wasmSrc = resolve(
        __dirname,
        "node_modules/onnxruntime-web/dist"
      );
      const wasmDest = resolve(__dirname, "public");
      if (!existsSync(wasmDest)) mkdirSync(wasmDest, { recursive: true });

      const files = [
        "ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd.wasm",
        "ort-wasm.wasm",
        "ort-wasm-threaded.wasm",
      ];

      for (const file of files) {
        const src = resolve(wasmSrc, file);
        const dest = resolve(wasmDest, file);
        try {
          copyFileSync(src, dest);
        } catch {
          // File may not exist in all ort versions — skip silently
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyOrtWasm()],

  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },

  server: {
    headers: {
      // Required for SharedArrayBuffer (multi-threaded WASM)
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },

  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks: {
          ort: ["onnxruntime-web"],
          mediapipe: ["@mediapipe/tasks-vision"],
        },
      },
    },
  },
});
