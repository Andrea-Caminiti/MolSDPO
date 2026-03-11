import { useState, useCallback, useRef } from "react";
import Papa from "papaparse";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";

// ── Palette ──────────────────────────────────────────────────────────────
const C = {
  bg:         "#05080d",
  surface:    "#0a0f18",
  card:       "#0d1520",
  border:     "#152030",
  gridLine:   "#0f1e2e",
  muted:      "#3a5068",
  text:       "#6a8aaa",
  bright:     "#c0d4e8",
  green:      "#00e887",
  blue:       "#29b6f6",
  indigo:     "#7986cb",
  yellow:     "#ffd740",
  orange:     "#ff9d57",
  red:        "#ef5350",
  pink:       "#f06292",
  cyan:       "#26c6da",
  lime:       "#aeea00",
  teal:       "#4db6ac",
  purple:     "#ba68c8",
  amber:      "#ffca28",
};

// ── Smoothing (EMA) ───────────────────────────────────────────────────────
function ema(data, alpha = 0.08) {
  let s = null;
  return data.map((v) => {
    if (v == null || isNaN(v)) return null;
    s = s === null ? v : alpha * v + (1 - alpha) * s;
    return s;
  });
}

// ── Build column arrays from CSV rows ────────────────────────────────────
function buildColumns(rows) {
  if (!rows || rows.length === 0) return { steps: [], columns: {} };
  const actualKeys = Object.keys(rows[0]);
  const stepKey = actualKeys.find((k) => k.toLowerCase() === "step") || "step";
  const allKeys = actualKeys.filter((k) => k !== stepKey && k !== "epoch");
  const steps = [...new Set(rows.map((r) => r[stepKey]).filter((s) => s != null))].sort((a, b) => a - b);
  const stepIndex = Object.fromEntries(steps.map((s, i) => [s, i]));
  const columns = {};
  for (const key of allKeys) columns[key] = new Array(steps.length).fill(null);
  for (const row of rows) {
    const idx = stepIndex[row[stepKey]];
    if (idx == null) continue;
    for (const key of allKeys) {
      const v = row[key];
      if (v != null && !isNaN(v) && v !== "") columns[key][idx] = Number(v);
    }
  }
  // Forward-fill
  for (const key of allKeys) {
    let last = null;
    for (let i = 0; i < columns[key].length; i++) {
      if (columns[key][i] != null) last = columns[key][i];
      else if (last != null) columns[key][i] = last;
    }
  }
  return { steps, columns };
}

// ── Shared chart style helpers ────────────────────────────────────────────
const axisStyle  = { fontSize: 9, fill: C.muted, fontFamily: "monospace" };
const tooltipStyle = { background: "#0a0f18", border: `1px solid ${C.border}`, fontSize: 10, fontFamily: "monospace", color: C.bright };

function fmtNum(v) { return typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toExponential(2) : v.toFixed(4)) : "—"; }

// ── Status badge ──────────────────────────────────────────────────────────
function Badge({ value, warn, healthy }) {
  if (value == null || healthy == null) return null;
  const bad = warn?.(value);
  return (
    <span style={{
      fontSize: 9, fontFamily: "monospace", padding: "1px 6px", borderRadius: 3,
      background: bad ? "#ef535018" : "#00e88718",
      color: bad ? C.red : C.green,
      border: `1px solid ${bad ? "#ef535040" : "#00e88740"}`,
    }}>{healthy}</span>
  );
}

// ── Single-line MetricChart ───────────────────────────────────────────────
function SingleChart({ title, metricKey, color, healthy, warn, steps, columns, height = 140 }) {
  const raw    = steps.map((_, i) => columns[metricKey]?.[i] ?? null);
  const smooth = ema(raw);
  const chartData = steps.map((s, i) => ({ step: s, raw: raw[i], smooth: smooth[i] }));
  const last = raw.filter((v) => v != null).at(-1);
  const isBad = last != null && warn?.(last);

  return (
    <div style={{
      background: C.card,
      border: `1px solid ${isBad ? "#ef535050" : C.border}`,
      borderRadius: 6, padding: "14px 16px 10px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ color: C.text, fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase" }}>{title}</span>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Badge value={last} warn={warn} healthy={healthy} />
          <span style={{ fontFamily: "monospace", fontSize: 12, color: isBad ? C.red : color, fontWeight: 700 }}>
            {fmtNum(last)}
          </span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={C.gridLine} strokeDasharray="3 3" vertical={true} />
          <XAxis dataKey="step" tick={axisStyle} tickLine={false} axisLine={{ stroke: C.border }} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} minTickGap={40} />
          <YAxis tick={axisStyle} tickLine={false} axisLine={{ stroke: C.border }} tickFormatter={v => Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(3)} width={52} domain={["auto","auto"]} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtNum(v)} labelFormatter={(l) => `step ${l}`} />
          <Line type="monotone" dataKey="raw"    dot={false} stroke={color} strokeWidth={1} strokeOpacity={0.25} isAnimationActive={false} name="raw" />
          <Line type="monotone" dataKey="smooth" dot={false} stroke={color} strokeWidth={2}                      isAnimationActive={false} name="smooth" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Multi-line MetricChart (merged) ───────────────────────────────────────
function MultiChart({ title, metrics, steps, columns, height = 160, note }) {
  const chartData = steps.map((s, i) => {
    const point = { step: s };
    metrics.forEach(({ key, label }) => {
      const v = columns[key]?.[i] ?? null;
      point[label] = v;
    });
    return point;
  });
  // Smooth data (show raw only, smoothed as main line)
  const smoothed = {};
  metrics.forEach(({ key, label }) => {
    const raw = steps.map((_, i) => columns[key]?.[i] ?? null);
    smoothed[label] = ema(raw);
  });
  const chartDataS = steps.map((s, i) => {
    const point = { step: s };
    metrics.forEach(({ label }) => { point[`${label}_s`] = smoothed[label][i]; });
    metrics.forEach(({ key, label }) => { point[label] = columns[key]?.[i] ?? null; });
    return point;
  });

  return (
    <div style={{
      background: C.card,
      border: `1px solid ${C.border}`,
      borderRadius: 6, padding: "14px 16px 10px",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ color: C.text, fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase" }}>{title}</span>
        {note && <span style={{ fontSize: 9, color: C.muted }}>{note}</span>}
        <div style={{ display: "flex", gap: 10 }}>
          {metrics.map(({ label, color, key }) => {
            const vals = steps.map((_, i) => columns[key]?.[i] ?? null).filter(v => v != null);
            const last = vals.at(-1);
            return (
              <span key={label} style={{ fontFamily: "monospace", fontSize: 11, color }}>
                <span style={{ color: C.muted, marginRight: 3 }}>{label}</span>{fmtNum(last)}
              </span>
            );
          })}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartDataS} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={C.gridLine} strokeDasharray="3 3" />
          <XAxis dataKey="step" tick={axisStyle} tickLine={false} axisLine={{ stroke: C.border }} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} minTickGap={40} />
          <YAxis tick={axisStyle} tickLine={false} axisLine={{ stroke: C.border }} tickFormatter={v => Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(3)} width={52} domain={["auto","auto"]} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtNum(v)} labelFormatter={(l) => `step ${l}`} />
          <Legend wrapperStyle={{ fontSize: 9, color: C.text, fontFamily: "monospace", paddingTop: 4 }} />
          {metrics.map(({ label, color }) => (
            <Line key={label+"_raw"} type="monotone" dataKey={label} dot={false} stroke={color} strokeWidth={1} strokeOpacity={0.2} isAnimationActive={false} legendType="none" />
          ))}
          {metrics.map(({ label, color }) => (
            <Line key={label+"_s"} type="monotone" dataKey={`${label}_s`} name={label} dot={false} stroke={color} strokeWidth={2} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Group definitions ─────────────────────────────────────────────────────
// Each chart is either type "single" or "multi"
// Columns prefixed with __ are debug/internal — shown in a dedicated section
function buildGroups() {
  return [
    // ── Rewards ────────────────────────────────────────────────────────────
    {
      label: "Rewards",
      charts: [
        { type: "multi", title: "Reward Levels", span: 2, height: 180,
          metrics: [
            { key: "reward/last",   label: "Last",   color: C.green  },
            { key: "reward/anchor", label: "Anchor", color: C.blue   },
            { key: "reward/start",  label: "Start",  color: C.indigo },
          ]
        },
        { type: "single", title: "Reward Progression", metricKey: "reward/progression", color: C.yellow, healthy: "> 0", warn: v => v <= 0 },
        { type: "single", title: "Reward Last σ",       metricKey: "reward/last_std",    color: C.orange },
        { type: "single", title: "Reward0 Mean",        metricKey: "Reward0_mean",        color: C.pink   },
        { type: "single", title: "Novelty Mean",        metricKey: "reward/novelty_mean", color: C.teal   },
      ],
    },
    // ── Reward sub-scores ─────────────────────────────────────────────────
    {
      label: "Reward Sub-scores",
      charts: [
        { type: "multi", title: "Sub-score Components", span: 2, height: 180,
          metrics: [
            { key: "reward_sub/validity",  label: "Validity",  color: C.green  },
            { key: "reward_sub/chemical",  label: "Chemical",  color: C.cyan   },
            { key: "reward_sub/geometric", label: "Geometric", color: C.yellow },
            { key: "reward_sub/bond_dev",  label: "Bond Dev",  color: C.orange },
          ]
        },
        { type: "multi", title: "Sigma Scores",
          metrics: [
            { key: "reward_sigma/validity",  label: "Validity",  color: C.green  },
            { key: "reward_sigma/geometric", label: "Geometric", color: C.yellow },
            { key: "reward_sigma/bond_dev",  label: "Bond Dev",  color: C.orange },
          ]
        },
        { type: "multi", title: "Reward Weights",
          metrics: [
            { key: "reward_weight/validity",  label: "Validity",  color: C.green  },
            { key: "reward_weight/chemical",  label: "Chemical",  color: C.cyan   },
            { key: "reward_weight/geometric", label: "Geometric", color: C.yellow },
          ]
        },
      ],
    },
    // ── Winrate ───────────────────────────────────────────────────────────
    {
      label: "Winrate",
      charts: [
        { type: "multi", title: "Winrate Mean & Conf", span: 2, height: 180,
          metrics: [
            { key: "winrate/mean",      label: "Mean",      color: C.green },
            { key: "winrate/conf_mean", label: "Conf Mean", color: C.cyan  },
          ]
        },
        { type: "single", title: "Winrate σ", metricKey: "winrate/std", color: C.orange },
      ],
    },
    // ── Policy ────────────────────────────────────────────────────────────
    {
      label: "Policy",
      charts: [
        { type: "single", title: "Approx KL",     metricKey: "__policy/approx_kl",    color: C.red,  healthy: "0.01–0.05", warn: v => v > 0.1 || v < 0.001 },
        { type: "multi",  title: "Log-Ratio Mean & Std", height: 160,
          metrics: [
            { key: "__policy/log_ratio_mean", label: "Mean", color: C.cyan },
            { key: "policy/log_ratio_std",    label: "Std",  color: C.pink },
          ]
        },
      ],
    },
    // ── Alignment & Loss ──────────────────────────────────────────────────
    {
      label: "Alignment & Loss",
      charts: [
        { type: "multi", title: "Correlation — all windows", span: 2, height: 180,
          metrics: [
            { key: "alignment/corr",       label: "Overall", color: C.green  },
            { key: "alignment/corr_early", label: "Early",   color: C.blue   },
            { key: "alignment/corr_mid",   label: "Mid",     color: C.yellow },
            { key: "alignment/corr_late",  label: "Late",    color: C.orange },
          ]
        },
        { type: "single", title: "SDPO Loss", metricKey: "loss/sdpo", color: C.red, healthy: "↓", warn: v => isNaN(v) },
      ],
    },
    // ── Advantage ─────────────────────────────────────────────────────────
    {
      label: "Advantage",
      charts: [
        { type: "multi", title: "Advantage Mean & Std",
          metrics: [
            { key: "advantage/mean", label: "Mean", color: C.teal   },
            { key: "advantage/std",  label: "Std",  color: C.purple },
          ]
        },
        { type: "single", title: "Advantage Abs Max", metricKey: "advantage/abs_max", color: C.amber },
      ],
    },
    // ── Collapse detection ────────────────────────────────────────────────
    {
      label: "Collapse",
      charts: [
        { type: "single", title: "Detection",           metricKey: "collapse/detection",           color: C.red,    warn: v => v > 0 },
        { type: "single", title: "Win-rate Entropy",    metricKey: "collapse/win_rate_entropy",    color: C.orange, warn: v => v < 0.5 },
        { type: "single", title: "Trajectory Variance", metricKey: "collapse/trajectory_variance", color: C.yellow },
      ],
    },
    // ── Gradients & Entropy ───────────────────────────────────────────────
    {
      label: "Gradients & Entropy",
      charts: [
        { type: "single", title: "Grad Norm", metricKey: "__grad/norm", color: C.orange },
        { type: "multi",  title: "Entropy",
          metrics: [
            { key: "entropy/coord", label: "Coord", color: C.lime   },
            { key: "entropy/types", label: "Types", color: C.purple },
          ]
        },
      ],
    },
    // ── Validation ────────────────────────────────────────────────────────
    {
      label: "Validation — Diversity & Quality",
      charts: [
        { type: "multi", title: "Core Validity Metrics", span: 2, height: 180,
          metrics: [
            { key: "val/validity",   label: "Validity",   color: C.green  },
            { key: "val/uniqueness", label: "Uniqueness", color: C.blue   },
            { key: "val/diversity",  label: "Diversity",  color: C.orange },
            { key: "val/quality",    label: "Quality",    color: C.purple },
          ]
        },
        { type: "multi", title: "Chem Scores",
          metrics: [
            { key: "val/qed",     label: "QED",     color: C.teal },
            { key: "val/sa_score",label: "SA Score",color: C.pink },
            { key: "val/sa_norm", label: "SA Norm", color: C.cyan },
          ]
        },
        { type: "single", title: "Stopping Score", metricKey: "val/stopping_score", color: C.lime  },
        { type: "single", title: "Mol Weight",     metricKey: "val/mol_weight",     color: C.amber },
      ],
    },
    // ── Debug (__ prefix) ─────────────────────────────────────────────────
    {
      label: "Debug",
      charts: [
        { type: "single", title: "Diversity Mean",    metricKey: "__reward/diversity_mean", color: C.muted },
        { type: "single", title: "Novelty Rate",      metricKey: "__reward/novelty_rate",   color: C.muted },
        { type: "single", title: "Corr Late (dbg)",   metricKey: "__alignment/corr_late",   color: C.muted },
        { type: "single", title: "Sigma Chemical",    metricKey: "__reward_sigma/chemical", color: C.muted },
        { type: "single", title: "Weight Bond Dev",   metricKey: "__reward_weight/bond_dev",color: C.muted },
        { type: "single", title: "Val Novelty (dbg)", metricKey: "__val/novelty",            color: C.muted },
        { type: "single", title: "Val Gate (dbg)",    metricKey: "__val/gate",               color: C.muted },
      ],
    },
  ];
}

// ── Summary bar ───────────────────────────────────────────────────────────
function SummaryBar({ columns }) {
  const get = (k) => columns[k] ?? [];
  const checks = [
    { label: "No NaN in SDPO loss",  ok: !get("loss/sdpo").some(isNaN) },
    { label: "Reward ↑",             ok: (get("reward/last").at(-1) ?? 0) >= (get("reward/last")[0] ?? 0) },
    { label: "KL < 0.1",            ok: (get("__policy/approx_kl").at(-1) ?? 0) < 0.1 },
    { label: "Winrate > 0.5",       ok: (get("winrate/mean").at(-1) ?? 0) > 0.5 },
    { label: "Corr → 1",           ok: (get("alignment/corr").at(-1) ?? 0) > 0.3 },
    { label: "No Collapse",         ok: (get("collapse/detection").at(-1) ?? 0) === 0 },
    { label: "Validity > 0.5",      ok: (get("val/validity").at(-1) ?? 0) > 0.5 },
  ];

  return (
    <div style={{ display: "flex", gap: 8, marginBottom: 28, flexWrap: "wrap" }}>
      {checks.map((c) => (
        <span key={c.label} style={{
          fontSize: 10, padding: "4px 12px", borderRadius: 4, fontFamily: "monospace",
          background: c.ok ? "#00e88710" : "#ef535010",
          color: c.ok ? C.green : C.red,
          border: `1px solid ${c.ok ? "#00e88730" : "#ef535030"}`,
        }}>
          {c.ok ? "✓" : "✗"} {c.label}
        </span>
      ))}
    </div>
  );
}

// ── Section heading ───────────────────────────────────────────────────────
function SectionHeading({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, marginTop: 28 }}>
      <span style={{ fontSize: 11, color: C.muted, letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "monospace" }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 1, background: C.border }} />
    </div>
  );
}

// ── Download helper ───────────────────────────────────────────────────────
async function downloadDashboard(ref) {
  const node = ref.current;
  if (!node) return;

  if (!window.html2canvas) {
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js";
      s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  const canvas = await window.html2canvas(node, { backgroundColor: C.bg, scale: 2, useCORS: true });
  const a = document.createElement("a");
  a.download = `training-dashboard-${Date.now()}.png`;
  a.href = canvas.toDataURL("image/png");
  a.click();
}

// ── Drop zone / file input ────────────────────────────────────────────────
function DropZone({ onFile }) {
  const [dragging, setDragging] = useState(false);
  const handleDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) readFile(file, onFile);
  };
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${dragging ? C.green : C.border}`,
        borderRadius: 10, padding: "80px 40px",
        textAlign: "center", transition: "border-color 0.2s",
        background: dragging ? "#00e88705" : "transparent",
      }}
    >
      <div style={{ fontSize: 36, marginBottom: 16 }}>📈</div>
      <div style={{ color: C.bright, fontSize: 14, marginBottom: 8, fontFamily: "monospace" }}>
        Drop a training CSV here
      </div>
      <div style={{ color: C.muted, fontSize: 11, marginBottom: 20 }}>or click to browse</div>
      <label style={{
        padding: "8px 20px", background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: 5, color: C.text, fontSize: 11, cursor: "pointer", fontFamily: "monospace",
      }}>
        Choose File
        <input type="file" accept=".csv" style={{ display: "none" }} onChange={(e) => {
          const file = e.target.files[0];
          if (file) readFile(file, onFile);
        }} />
      </label>
    </div>
  );
}

function readFile(file, cb) {
  const reader = new FileReader();
  reader.onload = (ev) => cb(ev.target.result, file.name);
  reader.readAsText(file);
}

// ── Main App ──────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData]         = useState(null);
  const [filename, setFilename] = useState("");
  const [downloading, setDownloading] = useState(false);
  const dashRef = useRef(null);

  const onFile = useCallback((text, name) => {
    const { data: rows } = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
    setData(buildColumns(rows));
    setFilename(name);
  }, []);

  const handleDownload = async () => {
    setDownloading(true);
    try { await downloadDashboard(dashRef); }
    catch (e) { console.error(e); }
    finally { setDownloading(false); }
  };

  const groups = buildGroups();

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.bright, fontFamily: "monospace" }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${C.border}`, padding: "14px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: C.surface, position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: C.green, fontSize: 14, fontWeight: 700, letterSpacing: "0.1em" }}>⬡ TRAIN MONITOR</span>
          {filename && <span style={{ color: C.muted, fontSize: 10 }}>{filename} · {data?.steps?.length ?? 0} steps</span>}
        </div>
        {data && (
          <div style={{ display: "flex", gap: 10 }}>
            <button onClick={handleDownload} disabled={downloading} style={{
              padding: "6px 16px", background: downloading ? C.surface : "#00e88715",
              border: `1px solid ${downloading ? C.border : "#00e88740"}`,
              borderRadius: 5, color: downloading ? C.muted : C.green,
              fontSize: 10, cursor: downloading ? "default" : "pointer", fontFamily: "monospace",
              letterSpacing: "0.05em",
            }}>
              {downloading ? "⏳ Rendering…" : "⬇ Download PNG"}
            </button>
            <button onClick={() => { setData(null); setFilename(""); }} style={{
              padding: "6px 14px", background: "none",
              border: `1px solid ${C.border}`,
              borderRadius: 5, color: C.muted,
              fontSize: 10, cursor: "pointer", fontFamily: "monospace",
            }}>
              ✕ Clear
            </button>
          </div>
        )}
      </div>

      {/* Body */}
      <div style={{ padding: "28px 32px" }}>
        {!data ? (
          <DropZone onFile={onFile} />
        ) : (
          <div ref={dashRef} style={{ background: C.bg, padding: "24px" }}>
            <SummaryBar columns={data.columns} />

            {groups.map((group) => (
              <div key={group.label}>
                <SectionHeading label={group.label} />
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 1fr)",
                  gap: 12,
                }}>
                  {group.charts.map((chart, ci) => {
                    const spanStyle = chart.span === 2
                      ? { gridColumn: "span 2" }
                      : chart.span === 3
                      ? { gridColumn: "span 3" }
                      : {};

                    const hasData = chart.type === "multi"
                      ? chart.metrics.some(m => (data.columns[m.key] ?? []).some(v => v != null))
                      : (data.columns[chart.metricKey] ?? []).some(v => v != null);

                    if (!hasData) return null;

                    return (
                      <div key={ci} style={spanStyle}>
                        {chart.type === "multi" ? (
                          <MultiChart
                            title={chart.title}
                            metrics={chart.metrics}
                            steps={data.steps}
                            columns={data.columns}
                            height={chart.height ?? 150}
                          />
                        ) : (
                          <SingleChart
                            title={chart.title}
                            metricKey={chart.metricKey}
                            color={chart.color}
                            healthy={chart.healthy}
                            warn={chart.warn}
                            steps={data.steps}
                            columns={data.columns}
                            height={chart.height ?? 140}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
