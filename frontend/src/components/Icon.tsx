import type { CSSProperties } from "react";

export type IconName = "grid" | "shield" | "chart" | "image" | "users" | "text" | "menu" | "close" | "arrow" | "check" | "alert" | "database" | "refresh" | "upload" | "layers";
const paths: Record<IconName, string> = {
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  shield: "M12 3 3 7v5c0 5 9 9 9 9s9-4 9-9V7l-9-4Z M12 8v5 M12 16h.01",
  chart: "M4 3v17h17 M7 14l4-5 4 3 5-7",
  image: "M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z M3 16l6-6 5 5 3-3 4 4 M15 7h.01",
  users: "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2 M16 4a4 4 0 0 1 0 8 M22 21v-2a4 4 0 0 0-3-3.87 M9 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8",
  text: "M4 3h16v14H8l-4 4V3Z M8 7h8 M8 11h6",
  menu: "M4 6h16 M4 12h16 M4 18h16",
  close: "m6 6 12 12 M6 18 18 6",
  arrow: "M5 12h14 m-5-5 5 5-5 5",
  check: "m5 12 4 4L19 6",
  alert: "m12 3 10 18H2L12 3Z M12 9v5 M12 17h.01",
  database: "M20 6c0 2-3.6 3-8 3S4 8 4 6s3.6-3 8-3 8 1 8 3Z M4 6v12c0 2 3.6 3 8 3s8-1 8-3V6 M4 12c0 2 3.6 3 8 3s8-1 8-3",
  refresh: "M20 7v5h-5 M4 17v-5h5 M6 7a7 7 0 0 1 11-2l3 3 M4 16l3 3a7 7 0 0 0 11-2",
  upload: "M12 16V3 m-5 5 5-5 5 5 M4 16v5h16v-5",
  layers: "m12 3 10 5-10 5L2 8l10-5Z M2 12l10 5 10-5 M2 16l10 5 10-5",
};

export default function Icon({ name, className = "h-5 w-5", style }: { name: IconName; className?: string; style?: CSSProperties }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={`shrink-0 ${className}`} style={style}><path d={paths[name]} /></svg>;
}
