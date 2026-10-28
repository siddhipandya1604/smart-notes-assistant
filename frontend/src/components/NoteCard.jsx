import { useEffect, useRef } from "react";
import { timeAgo } from "../lib/time.js";
import { CategoryChip, ConfidenceMeter, UncertainBadge, needsReview } from "./Badges.jsx";

const MATCH_LABEL = { keyword: "Keyword match", both: "Keyword + topic", topic: "Related topic" };

export default function NoteCard({ note, result, onDelete, variant = "card", focused = false }) {
  const ref = useRef(null);
  const review = needsReview(note);
  const relevance = result ? Math.min(100, Math.round(result.score * 100)) : null;

  useEffect(() => {
    if (focused) ref.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [focused]);

  return (
    <article ref={ref} className={`note ${variant} l-${note.label} ${review ? "review" : ""} ${focused ? "pulse" : ""}`}>
      <div className="note-main">
        <p className="note-text">{note.text}</p>
        <div className="note-tags">
          <CategoryChip category={note.label} />
          {review && <UncertainBadge confidence={note.label_confidence} />}
          {result && <span className={`match match-${result.match}`}>{MATCH_LABEL[result.match]}</span>}
        </div>
        {result && (
          <div className="bar" aria-label={`Relevance ${relevance}%`}>
            <div className="bar-fill" style={{ width: `${relevance}%` }} />
          </div>
        )}
      </div>
      <div className="note-meta">
        <span className="note-id">{note.id}</span>
        <span className="note-time">{timeAgo(note.created_at)}</span>
        <ConfidenceMeter value={note.label_confidence} uncertain={review} />
        <button className="icon-btn" onClick={() => onDelete(note.id)} aria-label={`Delete note ${note.id}`} title="Delete">
          ✕
        </button>
      </div>
    </article>
  );
}
