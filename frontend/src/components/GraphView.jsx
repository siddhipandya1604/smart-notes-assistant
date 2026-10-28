import { useMemo, useState } from "react";
import { ICONS, needsReview } from "./Badges.jsx";

const W = 900;
const H = 560;

// Obsidian-style map: one hub per label on an ellipse, each note orbiting its hub.
// Layout is deterministic (no physics), so it never jitters between renders.
function layout(notes) {
  const labels = [...new Set(notes.map((n) => n.label))];
  const hubs = {};
  labels.forEach((label, i) => {
    const angle = (2 * Math.PI * i) / labels.length - Math.PI / 2;
    hubs[label] = { x: W / 2 + Math.cos(angle) * (W * 0.32), y: H / 2 + Math.sin(angle) * (H * 0.32) };
  });
  const seen = {};
  const nodes = notes.map((note) => {
    const k = (seen[note.label] = (seen[note.label] ?? -1) + 1);
    const ring = 1 + Math.floor(k / 7);
    const angle = (k % 7) * ((2 * Math.PI) / 7) + ring * 0.7;
    const hub = hubs[note.label];
    return { note, x: hub.x + Math.cos(angle) * (34 + ring * 20), y: hub.y + Math.sin(angle) * (34 + ring * 20), hub };
  });
  return { hubs, nodes };
}

export default function GraphView({ notes, onOpen }) {
  const [hover, setHover] = useState(null);
  const { hubs, nodes } = useMemo(() => layout(notes), [notes]);
  const counts = useMemo(() => notes.reduce((c, n) => ({ ...c, [n.label]: (c[n.label] || 0) + 1 }), {}), [notes]);

  return (
    <div className="graph-wrap">
      <svg className="graph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Map of notes grouped by label">
        {nodes.map(({ note, x, y, hub }) => (
          <line key={`e-${note.id}`} x1={hub.x} y1={hub.y} x2={x} y2={y} className="edge" />
        ))}
        {Object.entries(hubs).map(([label, hub]) => (
          <g key={label} className={`hub l-${label}`}>
            <circle cx={hub.x} cy={hub.y} r={22 + Math.min(counts[label], 8) * 1.5} className="hub-circle" />
            <text x={hub.x} y={hub.y + 5} textAnchor="middle" className="hub-icon">{ICONS[label]}</text>
          </g>
        ))}
        {nodes.map(({ note, x, y }) => (
          <g
            key={note.id}
            className={`gnode l-${note.label} ${needsReview(note) ? "review" : ""} ${hover?.id === note.id ? "hot" : ""}`}
            transform={`translate(${x} ${y})`}
            tabIndex={0}
            role="button"
            aria-label={`${note.id}: ${note.text}`}
            onMouseEnter={() => setHover(note)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(note)}
            onBlur={() => setHover(null)}
            onClick={() => onOpen(note.id)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onOpen(note.id)}
          >
            <circle r="9" className="gnode-ring" />
            <circle r="5.5" className="gnode-core" />
          </g>
        ))}
        {/* labels last, with an outline in the page colour, so no orbiting dot can hide them */}
        {Object.entries(hubs).map(([label, hub]) => (
          <text key={`t-${label}`} x={hub.x} y={hub.y + 52 + Math.min(counts[label], 8) * 1.5} textAnchor="middle" className="hub-label">
            {label} · {counts[label]}
          </text>
        ))}
      </svg>
      <div className="graph-caption" aria-live="polite">
        {hover ? (
          <><strong>{hover.id}</strong> · {hover.label}{needsReview(hover) ? " · uncertain" : ""}<span>{hover.text}</span></>
        ) : (
          <span>Hover a dot to read the note, click to open it. Dashed rings are notes the model isn't sure about.</span>
        )}
      </div>
    </div>
  );
}
