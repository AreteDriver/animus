# Animus PWA

**Progressive web app interface for Animus — accessible from any device.**

Built with Vite, React, and TypeScript. Provides a lightweight, installable interface to the Animus system without requiring a native app store.

## Features

- **Installable** — Works offline after first load, add to home screen
- **Responsive** — Desktop, tablet, and mobile layouts
- **Fast** — Vite-powered build with instant HMR during development
- **Type-safe** — Full TypeScript coverage

## Tech Stack

- **Build tool**: Vite
- **Framework**: React
- **Language**: TypeScript
- **Styling**: CSS (custom properties)

## Development

```bash
git clone git@github.com:AreteDriver/animus.git
cd animus/packages/pwa

npm install
npm run dev        # Start dev server (usually localhost:5173)
npm run build      # Production build to dist/
npm run preview    # Preview production build
```

## Configuration

The PWA connects to the Bootstrap API by default at `/api` (same-origin when served by the Bootstrap dashboard). In development, the Vite proxy forwards `/api` and `/ws` to `http://localhost:7700`.

Override at build time via environment variables:

```bash
VITE_API_BASE_URL=/api   npm run build
VITE_WS_BASE_URL=/ws     npm run build
```

## Deployment

The `dist/` directory is a static site. Deploy to any static host:

```bash
npm run build
# Deploy dist/ to Netlify, Vercel, GitHub Pages, S3, etc.
```

## Part of the Animus Monorepo

- [Animus Core](https://github.com/AreteDriver/animus/tree/main/packages/core) — operating environment engine
- [Animus Forge](https://github.com/AreteDriver/animus/tree/main/packages/forge) — multi-agent orchestration
- [Animus Bootstrap](https://github.com/AreteDriver/animus/tree/main/packages/bootstrap) — system daemon and dashboard

## License

MIT — 2026, AreteDriver
