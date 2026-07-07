/** Runtime configuration for the Animus PWA.
 *
 * Centralises API base URL, WebSocket endpoint, and feature flags so the
 * same build works in development (Vite proxy) and production (served by
 * Bootstrap dashboard).
 */

/** API base path — must match the Bootstrap dashboard mount point. */
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

/** WebSocket base path — used for the real-time chat channel. */
export const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL ?? "/ws";

/** How many times to retry a failed API request before surfacing the error. */
export const API_RETRY_COUNT = Number(import.meta.env.VITE_API_RETRY_COUNT ?? "1");

/** Debounce interval (ms) for auto-saving the session checkpoint. */
export const CHECKPOINT_SAVE_DEBOUNCE_MS = Number(
  import.meta.env.VITE_CHECKPOINT_SAVE_DEBOUNCE_MS ?? "5000",
);
