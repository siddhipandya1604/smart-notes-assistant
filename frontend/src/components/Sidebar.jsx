import { CORE, EXTRA, ICONS, LabelDot } from "./Badges.jsx";

const THEMES = { auto: "◐ Auto", light: "☀ Light", dark: "☾ Dark" };

export default function Sidebar({ active, counts, onChange, onLoadStarter, theme, onTheme }) {
  const labels = [...CORE, ...[...EXTRA, "Other"].filter((l) => counts[l] > 0 || l === active)];
  const item = (key, content, count) => (
    <button key={key} className={`nav-item ${active === key ? "active" : ""}`} aria-pressed={active === key} onClick={() => onChange(key)}>
      {content}
      <span className="nav-count">{count}</span>
    </button>
  );
  return (
    <aside className="sidebar">
      <div className="workspace">
        <span className="ws-logo" aria-hidden="true">📝</span>
        <div>
          <div className="ws-name">Smart Notes</div>
          <div className="ws-sub">Filed by a model you trained</div>
        </div>
      </div>

      <nav className="nav" aria-label="Notes">
        {item("All", <><span className="nav-icon">🗒️</span>All notes</>, counts.All)}
        {item("Uncertain", <><span className="nav-icon">⚠️</span>Needs a look</>, counts.Uncertain)}
        <div className="nav-heading">Labels</div>
        {labels.map((l) => item(l, <><LabelDot label={l} /><span className="nav-label">{l}</span><span className="nav-emoji" aria-hidden="true">{ICONS[l]}</span></>, counts[l] || 0))}
      </nav>

      <div className="sidebar-foot">
        <button className="nav-item plain" onClick={onLoadStarter}>
          <span className="nav-icon">⬇️</span>Load starter notes
        </button>
        <button className="nav-item plain" onClick={() => onTheme({ auto: "light", light: "dark", dark: "auto" }[theme])}>
          {THEMES[theme]}<span className="nav-count">theme</span>
        </button>
      </div>
    </aside>
  );
}
