import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  base: "/pwa/",
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      // Custom service worker (src/sw.ts) so we can handle Web Push while
      // keeping Workbox precaching for offline support.
      strategies: "injectManifest",
      srcDir: "src",
      filename: "sw.ts",
      includeAssets: ["favicon.svg"],
      manifest: {
        name: "Animus",
        short_name: "Animus",
        description: "Personal AI exocortex — mobile companion",
        theme_color: "#0f172a",
        background_color: "#0f172a",
        display: "standalone",
        orientation: "portrait",
        scope: "/pwa/",
        start_url: "/pwa/",
        // OS share sheet → opens the app at start_url with the shared text as
        // query params; App.tsx routes that into the Capture view.
        share_target: {
          action: "/pwa/",
          method: "GET",
          params: { title: "title", text: "text", url: "url" },
        },
        icons: [
          {
            src: "icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "icon-512.png",
            sizes: "512x512",
            type: "image/png",
          },
        ],
      },
      injectManifest: {
        globPatterns: ["**/*.{js,css,html,svg,png}"],
      },
    }),
  ],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:7700",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://localhost:7700",
        ws: true,
      },
    },
  },
});
