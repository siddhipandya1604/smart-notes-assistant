// Thin wrapper over the FastAPI backend. Every failure becomes an Error with a human message.
async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Can't reach the server. Is it running?");
  }
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.message || `Request failed (${response.status}).`);
  return body;
}

export const api = {
  classify: (text) => request("/classify", { method: "POST", body: JSON.stringify({ text }) }),
  listNotes: () => request("/api/notes"),
  addNote: (text) => request("/api/notes", { method: "POST", body: JSON.stringify({ text }) }),
  deleteNote: (id) => request(`/api/notes/${id}`, { method: "DELETE" }),
  loadStarter: () => request("/api/notes/load-starter", { method: "POST" }),
  search: (q) => request(`/api/search?q=${encodeURIComponent(q)}`),
};

export const CATEGORIES = ["Study", "Money", "Travel", "Shopping"];
