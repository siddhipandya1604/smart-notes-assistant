import { CategoryChip, ConfidenceMeter, UncertainBadge } from "./Badges.jsx";

const MATCH_LABEL = { keyword: "Keyword match", both: "Keyword + topic match", topic: "Related topic" };

export default function NoteCard({ note, result, onDelete }) {
  const relevance = result ? Math.min(100, Math.round(result.score * 100)) : null;
  return (
    <article className={`card note cat-border-${note.category} ${note.uncertain ? "is-uncertain" : ""}`}>
      <header className="note-head">
        <span className="note-id">{note.id}</span>
        <div className="note-tags">
          <CategoryChip category={note.category} />
          {note.uncertain && <UncertainBadge confidence={note.confidence} />}
        </div>
      </header>
      <p className="note-text">{note.text}</p>
      {result && (
        <div className="relevance">
          <span className={`match match-${result.match}`}>{MATCH_LABEL[result.match]}</span>
          <div className="bar" aria-label={`Relevance ${relevance}%`}>
            <div className="bar-fill" style={{ width: `${relevance}%` }} />
          </div>
        </div>
      )}
      <footer className="note-foot">
        <ConfidenceMeter value={note.confidence} uncertain={note.uncertain} />
        <button className="btn ghost danger" onClick={() => onDelete(note.id)} aria-label={`Delete note ${note.id}`}>
          Delete
        </button>
      </footer>
    </article>
  );
}
