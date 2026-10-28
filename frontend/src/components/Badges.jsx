export const CORE = ["Study", "Money", "Travel", "Shopping"];
export const EXTRA = ["Health", "Family", "Home", "Work"];
export const ICONS = {
  Study: "📚", Money: "💰", Travel: "✈️", Shopping: "🛍️",
  Health: "💪", Family: "👨‍👩‍👧", Home: "🏠", Work: "💼", Other: "🗂️",
};

// The API's `uncertain` means "fits none of the four core categories, or the model isn't sure".
// A note the model confidently recognises as Health/Family/Home/Work is not a mystery to the user,
// so the amber badge is reserved for core labels the model doubts and for notes it can't place.
export function needsReview(item) {
  return Boolean(item.uncertain) && (CORE.includes(item.label) || item.label === "Other");
}

export function CategoryChip({ category }) {
  const extra = EXTRA.includes(category);
  return (
    <span className={`chip cat-${category}`} title={extra ? "Outside the four core categories" : undefined}>
      <span aria-hidden="true">{ICONS[category] || "🏷️"}</span> {category}
    </span>
  );
}

export function UncertainBadge({ confidence }) {
  return (
    <span className="badge-uncertain" title={`The model is only ${Math.round(confidence * 100)}% sure. This is its best guess.`}>
      ⚠ Uncertain
    </span>
  );
}

export function ConfidenceMeter({ value, uncertain }) {
  const pct = Math.round(value * 100);
  return (
    <div className="meter" role="img" aria-label={`Confidence ${pct}%`}>
      <div className={`meter-fill ${uncertain ? "low" : ""}`} style={{ width: `${pct}%` }} />
      <span className="meter-label">{pct}%</span>
    </div>
  );
}
