export const ICONS = { Study: "📚", Money: "💰", Travel: "✈️", Shopping: "🛍️" };

export function CategoryChip({ category }) {
  return (
    <span className={`chip cat-${category}`}>
      <span aria-hidden="true">{ICONS[category]}</span> {category}
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
