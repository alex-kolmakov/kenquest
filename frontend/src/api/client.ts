/**
 * API client helpers — fetch wrappers and SSE utilities.
 */

const BASE = "/api";

export async function apiFetch<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Subscribe to a Server-Sent Events endpoint.
 * Returns a cleanup function to close the connection.
 */
export function subscribeSSE(
  path: string,
  onMessage: (data: string) => void,
  onError?: (err: Event) => void
): () => void {
  const es = new EventSource(`${BASE}${path}`);
  es.onmessage = (e) => onMessage(e.data);
  if (onError) es.onerror = onError;
  return () => es.close();
}
