import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client.js";
import Composer from "./components/Composer.jsx";
import FilterBar from "./components/FilterBar.jsx";
import NoteCard from "./components/NoteCard.jsx";
import Toasts from "./components/Toasts.jsx";
import { useDebounced } from "./hooks/useDebounced.js";

function matchesFilter(note, filter) {
  if (filter === "All") return true;
  if (filter === "Uncertain") return note.uncertain;
  return note.category === filter; // uncertain notes show under their best-guess category
}

export default function App() {
  const [notes, setNotes] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [filter, setFilter] = useState("All");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState(null); // { query, results, message } | null
  const [toasts, setToasts] = useState([]);
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "auto");
  const debouncedQuery = useDebounced(query, 300);

  const toast = useCallback((message, kind = "info") => {
    const id = crypto.randomUUID();
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
  }, []);

  const refresh = useCallback(async () => {
    try {
      setNotes(await api.listNotes());
    } catch (e) {
      toast(e.message, "error");
    } finally {
      setLoaded(true);
    }
  }, [toast]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Search runs against the server whenever the query (or the set of notes) changes.
  useEffect(() => {
    let current = true;
    const q = debouncedQuery.trim();
    if (!q) {
      setSearch(null);
      return;
    }
    api.search(q).then((r) => current && setSearch(r)).catch((e) => current && toast(e.message, "error"));
    return () => { current = false; };
  }, [debouncedQuery, notes, toast]);

  async function saveNote(text) {
    const note = await api.addNote(text);
    setNotes((prev) => [...prev, note]);
    toast(note.uncertain ? `Saved as ${note.id}, marked uncertain (best guess: ${note.category})` : `Saved as ${note.id} · ${note.category}`,
      note.uncertain ? "warn" : "ok");
  }

  async function removeNote(id) {
    try {
      await api.deleteNote(id);
      setNotes((prev) => prev.filter((n) => n.id !== id));
      toast(`Deleted ${id}`);
    } catch (e) {
      toast(e.message, "error");
    }
  }

  async function loadStarter() {
    try {
      const r = await api.loadStarter();
      await refresh();
      toast(r.added ? `Loaded ${r.added} starter notes` : "Starter notes are already loaded", r.added ? "ok" : "info");
    } catch (e) {
      toast(e.message, "error");
    }
  }

  const counts = useMemo(() => {
    const c = { All: notes.length, Uncertain: notes.filter((n) => n.uncertain).length };
    for (const n of notes) c[n.category] = (c[n.category] || 0) + 1;
    return { Study: 0, Money: 0, Travel: 0, Shopping: 0, ...c };
  }, [notes]);

  const searching = query.trim().length > 0;
  const shown = useMemo(() => {
    if (searching) {
      if (!search || search.query !== debouncedQuery.trim()) return null; // still fetching
      return search.results.filter((r) => matchesFilter(r.note, filter)).map((r) => ({ note: r.note, result: r }));
    }
    return [...notes].reverse().filter((n) => matchesFilter(n, filter)).map((note) => ({ note }));
  }, [searching, search, debouncedQuery, notes, filter]);

  const themeIcon = { auto: "◐", light: "☀", dark: "☾" }[theme];
  const nextTheme = { auto: "light", light: "dark", dark: "auto" }[theme];

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <span className="logo" aria-hidden="true">📝</span>
          <div>
            <h1>Smart Notes</h1>
            <p>Jot it down. The model files it.</p>
          </div>
        </div>
        <div className="top-actions">
          <button className="btn secondary" onClick={loadStarter}>Load starter notes</button>
          <button className="btn icon" onClick={() => setTheme(nextTheme)} aria-label={`Theme: ${theme}. Switch to ${nextTheme}`} title={`Theme: ${theme}`}>
            {themeIcon}
          </button>
        </div>
      </header>

      <main>
        <Composer onSave={saveNote} />

        <section className="toolbar">
          <div className="search">
            <span aria-hidden="true">🔎</span>
            <input
              type="search"
              value={query}
              placeholder="Search notes, e.g. hotel or “what do I need to buy?”"
              aria-label="Search notes"
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <FilterBar active={filter} counts={counts} onChange={setFilter} />
        </section>

        {searching && search?.query === debouncedQuery.trim() && (
          <p className="search-summary" aria-live="polite">
            {search.results.length
              ? `${search.results.length} result${search.results.length > 1 ? "s" : ""} for “${search.query}”, best match first`
              : search.message}
          </p>
        )}

        {!loaded ? (
          <p className="state">Loading notes…</p>
        ) : shown === null ? (
          <p className="state">Searching…</p>
        ) : shown.length === 0 ? (
          <EmptyState searching={searching} hasNotes={notes.length > 0} filter={filter} onLoad={loadStarter} />
        ) : (
          <div className="grid">
            {shown.map(({ note, result }) => (
              <NoteCard key={note.id} note={note} result={result} onDelete={removeNote} />
            ))}
          </div>
        )}
      </main>

      <Toasts toasts={toasts} />
    </div>
  );
}

function EmptyState({ searching, hasNotes, filter, onLoad }) {
  if (searching) {
    return <p className="state">{filter === "All" ? "Nothing to show for that search." : `No ${filter} notes match that search.`}</p>;
  }
  if (!hasNotes) {
    return (
      <div className="state empty-big">
        <div className="empty-art" aria-hidden="true">🗒️</div>
        <h2>No notes yet</h2>
        <p>Write one above, or load the 12 starter notes to explore search and filters.</p>
        <button className="btn primary" onClick={onLoad}>Load starter notes</button>
      </div>
    );
  }
  return <p className="state">No {filter} notes yet.</p>;
}
