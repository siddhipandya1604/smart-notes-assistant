import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import { useDebounced } from "../hooks/useDebounced.js";
import { CategoryChip, ConfidenceMeter, UncertainBadge, needsReview } from "./Badges.jsx";

const MAX = 1000;

export default function Composer({ onSave }) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [preview, setPreview] = useState(null);
  const debounced = useDebounced(text, 350);

  // Live preview: ask the same model that will label the saved note.
  useEffect(() => {
    let current = true;
    if (!debounced.trim()) {
      setPreview(null);
      return;
    }
    api.classify(debounced).then((p) => current && setPreview(p)).catch(() => current && setPreview(null));
    return () => {
      current = false;
    };
  }, [debounced]);

  async function submit(event) {
    event.preventDefault();
    if (!text.trim()) {
      setError("Write something first. An empty note can't be saved.");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await onSave(text);
      setText("");
      setPreview(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  const stale = preview && text.trim() !== debounced.trim();

  return (
    <form className="card composer" onSubmit={submit} noValidate>
      <label htmlFor="note-text" className="composer-title">New note</label>
      <textarea
        id="note-text"
        value={text}
        maxLength={MAX}
        rows={3}
        placeholder="e.g. Practice machine learning for tomorrow's quiz"
        aria-invalid={Boolean(error)}
        aria-describedby="note-error"
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError("");
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
        }}
      />
      <div id="note-error" className="field-error" role="alert">{error}</div>

      <div className="composer-row">
        <div className={`preview ${preview ? "" : "empty"} ${stale ? "stale" : ""}`} aria-live="polite">
          {preview ? (
            <>
              <span className="preview-label">Model thinks</span>
              <CategoryChip category={preview.label} />
              {needsReview(preview) && <UncertainBadge confidence={preview.label_confidence} />}
              <ConfidenceMeter value={preview.label_confidence} uncertain={needsReview(preview)} />
            </>
          ) : (
            <span className="preview-label">Start typing to see the predicted category</span>
          )}
        </div>
        <div className="composer-actions">
          <span className="counter">{text.length}/{MAX}</span>
          <button className="btn primary" type="submit" disabled={saving}>
            {saving ? "Saving…" : "Save note"}
          </button>
        </div>
      </div>
    </form>
  );
}
