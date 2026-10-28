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

  // Live preview: ask the same models that will label the saved note.
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
    event?.preventDefault();
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
    <form className={`capture ${error ? "has-error" : ""}`} onSubmit={submit} noValidate>
      <textarea
        id="note-text"
        aria-label="New note"
        value={text}
        maxLength={MAX}
        rows={2}
        placeholder="Write a note… the model files it as you type"
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
      <div className="capture-row">
        <div className={`preview ${stale ? "stale" : ""}`} aria-live="polite">
          {preview ? (
            <>
              <span className="preview-label">Files under</span>
              <CategoryChip category={preview.label} />
              {needsReview(preview) && <UncertainBadge confidence={preview.label_confidence} />}
              <ConfidenceMeter value={preview.label_confidence} uncertain={needsReview(preview)} />
            </>
          ) : (
            <span className="preview-label">Label and confidence appear here</span>
          )}
        </div>
        <div className="capture-actions">
          {text.length > 0 && <span className="counter">{text.length}/{MAX}</span>}
          <kbd title="Press Ctrl or ⌘ + Enter to save">⌘↵</kbd>
          <button className="btn primary" type="submit" disabled={saving}>
            {saving ? "Saving…" : "Add note"}
          </button>
        </div>
      </div>
    </form>
  );
}
