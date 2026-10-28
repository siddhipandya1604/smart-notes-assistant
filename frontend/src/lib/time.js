// "just now", "5 min ago", "3 h ago", "yesterday", "12 Oct". Falls back to "" for bad input.
export function timeAgo(iso) {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const seconds = Math.max(0, (Date.now() - then.getTime()) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  if (seconds < 172800) return "yesterday";
  return then.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
