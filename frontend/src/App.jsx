import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api/client.js";
import { needsReview } from "./components/Badges.jsx";
import Composer from "./components/Composer.jsx";
import FilterBar from "./components/FilterBar.jsx";
import GraphView from "./components/GraphView.jsx";
import NoteCard from "./components/NoteCard.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Toasts from "./components/Toasts.jsx";
import { useDebounced } from "./hooks/useDebounced.js";

const VIEWS = [
  { key: "cards", icon: "▦", name: "Cards" },
  { key: "list", icon: "☰", name: "List" },
  { key: "map", icon: "◎", name: "Map" },
];

function matchesFilter(note, filter) {
  if (filter === "All") return true;
  if (filter === "Uncertain") return needsReview(note);
  return note.label === filter; // the label shown on the card; core labels are the four categories
}

// "#map" or "#list" in the URL picks a view; "#light" / "#dark" pick a theme (handy for screenshots).
function fromHash(options, fallback) {
  const hit = window.location.hash.replace("#", "").split("&").find((t) => options.includes(t));
  return hit || fallback;
}

export default function App() {
  const [notes, setNotes] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [filter, setFilter] = useState("All");
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState(null); // { query, results, message } | null
  const [toasts, setToasts] = useState([]);
  const [focusId, setFocusId] = useState(null);
  const [view, setView] = useState(() => fromHash(["cards", "list", "map"], localStorage.getItem("view") || "cards"));
  const [theme, setTheme] = useState(() => fromHash(["light", "dark"], localStorage.getItem("theme") || "auto"));
  const searchRef = useRef(null);
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

  useEffect(() => { localStorage.setItem("view", view); }, [view]);

  // Ctrl/⌘+K or "/" jumps to search; Escape clears it.
  useEffect(() => {
    function onKey(e) {
      const typing = ["INPUT", "TEXTAREA"].includes(document.activeElement?.tagName);
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === "Escape" && document.activeElement === searchRef.current) {
        setQuery("");
        searchRef.current.blur();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
    setFocusId(note.id);
    setTimeout(() => setFocusId(null), 2200);
    const review = needsReview(note);
    toast(review ? `Saved as ${note.id}, marked uncertain (best guess: ${note.label})` : `Saved as ${note.id} · ${note.label}`,
      review ? "warn" : "ok");
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

  function openFromMap(id) {
    setFilter("All");
    setView("cards");
    setFocusId(id);
    setTimeout(() => setFocusId(null), 2200);
  }

  const counts = useMemo(() => {
    const c = { All: notes.length, Uncertain: notes.filter(needsReview).length };
    for (const n of notes) c[n.label] = (c[n.label] || 0) + 1;
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

  const title = { All: "All notes", Uncertain: "Needs a look" }[filter] || filter;
  const effectiveView = searching && view === "map" ? "cards" : view; // the map can't show ranked results

  return (
    <div className="shell">
      <Sidebar active={filter} counts={counts} onChange={setFilter} onLoadStarter={loadStarter} theme={theme} onTheme={setTheme} />

      <main className="main">
        <div className="page">
          <header className="page-head">
            <div>
              <h1>{searching ? "Search" : title}</h1>
              <p className="page-sub">
                {searching ? "Ranked by the hybrid search" : `${shown?.length ?? 0} ${(shown?.length ?? 0) === 1 ? "note" : "notes"}`}
                {filter !== "All" && !searching && <button className="link" onClick={() => setFilter("All")}>· show all</button>}
              </p>
            </div>
            <div className="seg" role="group" aria-label="View">
              {VIEWS.map((v) => (
                <button key={v.key} className={view === v.key ? "on" : ""} aria-pressed={view === v.key} onClick={() => setView(v.key)} title={v.name}>
                  <span aria-hidden="true">{v.icon}</span> {v.name}
                </button>
              ))}
            </div>
          </header>

          <Composer onSave={saveNote} />

          <div className="searchbar">
            <span aria-hidden="true">⌕</span>
            <input
              ref={searchRef}
              type="search"
              value={query}
              placeholder="Search your notes, e.g. hotel or “what do I need to buy?”"
              aria-label="Search notes"
              onChange={(e) => setQuery(e.target.value)}
            />
            <kbd>Ctrl K</kbd>
          </div>
          <div className="mobile-filters"><FilterBar active={filter} counts={counts} onChange={setFilter} /></div>

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
          ) : effectiveView === "map" ? (
            <GraphView notes={shown.map((s) => s.note)} onOpen={openFromMap} />
          ) : (
            <div className={effectiveView === "list" ? "list" : "grid"}>
              {shown.map(({ note, result }) => (
                <NoteCard key={note.id} note={note} result={result} onDelete={removeNote}
                  variant={effectiveView === "list" ? "row" : "card"} focused={focusId === note.id} />
              ))}
            </div>
          )}
        </div>
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
        <h2>Your notes will live here</h2>
        <p>Write one above, or load the 12 starter notes to explore search, labels and the map.</p>
        <button className="btn primary" onClick={onLoad}>Load starter notes</button>
      </div>
    );
  }
  return <p className="state">No {filter === "Uncertain" ? "notes that need a look" : `${filter} notes`} yet.</p>;
}
