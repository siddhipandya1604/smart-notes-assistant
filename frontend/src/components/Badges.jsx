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

export function LabelDot({ label }) {
  return <span className={`dot l-${label}`} aria-hidden="true" />;
}

export function CategoryChip({ category }) {
  const extra = EXTRA.includes(category);
  return (
    <span className={`chip l-${category}`} title={extra ? "Outside the four core categories" : undefined}>
      <LabelDot label={category} />
      {category}
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
    <span className="meter" role="img" aria-label={`Confidence ${pct}%`} title={`Confidence ${pct}%`}>
      <span className="meter-track">
        <span className={`meter-fill ${uncertain ? "low" : ""}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="meter-label">{pct}%</span>
    </span>
  );
}
