import { CATEGORIES } from "../api/client.js";
import { ICONS } from "./Badges.jsx";

export default function FilterBar({ active, counts, onChange }) {
  const filters = [
    { key: "All", label: "All", count: counts.All },
    ...CATEGORIES.map((c) => ({ key: c, label: `${ICONS[c]} ${c}`, count: counts[c] })),
    { key: "Uncertain", label: "⚠ Uncertain", count: counts.Uncertain },
  ];
  return (
    <div className="filters" role="group" aria-label="Filter notes by category">
      {filters.map((f) => (
        <button
          key={f.key}
          className={`filter ${active === f.key ? "active" : ""} ${f.key !== "All" ? `f-${f.key}` : ""}`}
          aria-pressed={active === f.key}
          onClick={() => onChange(f.key)}
        >
          {f.label} <span className="count">{f.count}</span>
        </button>
      ))}
    </div>
  );
}
