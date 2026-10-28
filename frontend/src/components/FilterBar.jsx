import { CORE, EXTRA, ICONS } from "./Badges.jsx";

// The four core filters are always there. Health/Family/Home/Work/Other appear once a note has them.
export default function FilterBar({ active, counts, onChange }) {
  const shown = [...CORE, ...[...EXTRA, "Other"].filter((label) => counts[label] > 0 || label === active)];
  const filters = [
    { key: "All", label: "All", count: counts.All },
    ...shown.map((label) => ({ key: label, label: `${ICONS[label]} ${label}`, count: counts[label] || 0 })),
    { key: "Uncertain", label: "⚠ Uncertain", count: counts.Uncertain },
  ];
  return (
    <div className="filters" role="group" aria-label="Filter notes by label">
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
