// SPDX-FileCopyrightText: Copyright (c) 2026 Shreeyut Maheshwari. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Flow-map run dashboard: vanilla SVG charts, no external dependencies.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const MAX_SERIES = 3; // the first three categorical slots validate all-pairs (small multiples) in both modes
const REFRESH_MS = 15000;
const NFES_CONSISTENCY = [1, 2, 4, 8];

const LOSS_CHARTS = [
  { key: "loss/r_eq_t", title: "r = t · flow matching", sub: "u(x_t, t, t) against the data velocity" },
  { key: "loss/r_eq_0", title: "r = 0 · consistency", sub: "one jump from t straight to clean data" },
  { key: "loss/r_lt_t", title: "0 < r < t · flow map", sub: "jumps over a general interval [r, t]" },
  { key: "loss/total", title: "Total objective", sub: "the optimized loss, after all weights" },
];

const state = {
  runs: [],
  smooth: 0.9,
  yscale: "log",
  showRaw: true,
  sliderIndex: null,
  followLatest: true,
  sampleSignature: "",
};

// ---------- small DOM helpers (labels are untrusted: textContent only) ----------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}
function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  return node;
}
function lineKey(slot) {
  return el("span", { class: `key s${slot}`, "aria-hidden": "true" });
}

// ---------- number formatting ----------
function fmt(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return Math.round(v).toLocaleString();
  if (a >= 0.001) return Number(v.toPrecision(3)).toString();
  return v.toExponential(2);
}
function fmtIter(n) {
  if (n >= 10000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}K`;
  return Math.round(n).toLocaleString();
}
function fmtDuration(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "–";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.round(sec)}s`;
}

// ---------- scales & ticks ----------
function niceStep(range, count) {
  const raw = range / Math.max(count, 1);
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / pow;
  return (r >= 5 ? 10 : r >= 2 ? 5 : r >= 1 ? 2 : 1) * pow;
}
function linearTicks(lo, hi, count) {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, count);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}
function logTicks(lo, hi) {
  const ticks = [];
  const d0 = Math.floor(Math.log10(lo));
  const d1 = Math.ceil(Math.log10(hi));
  const decades = d1 - d0;
  const mult = decades <= 1 ? [1, 2, 5] : decades <= 3 ? [1, 3] : [1];
  for (let d = d0; d <= d1; d++) {
    for (const m of mult) {
      const v = m * Math.pow(10, d);
      if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
    }
  }
  return ticks;
}
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const i = Math.floor(pos);
  const f = pos - i;
  return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
}

// TensorBoard-style debiased exponential moving average; nulls stay null and do not reset the average.
function smoothSeries(ys, w) {
  if (!(w > 0)) return ys.slice();
  let last = 0;
  let weight = 0;
  return ys.map((y) => {
    if (y === null || !Number.isFinite(y)) return null;
    last = last * w + (1 - w) * y;
    weight = weight * w + (1 - w);
    return last / weight;
  });
}

function bisectNearest(xs, x) {
  let lo = 0;
  let hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(xs[hi] - x) < Math.abs(xs[lo] - x) ? hi : lo;
}
function indexOfX(xs, x) {
  let lo = 0;
  let hi = xs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] === x) return mid;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;
}

// ---------- line chart ----------
let chartUid = 0;
class LineChart {
  constructor(parent, spec) {
    this.spec = spec;
    this.uid = ++chartUid;
    this.view = "chart";
    this.series = [];
    this.opts = {};
    this.hoverX = null;

    this.toggleBtn = el("button", { class: "btn", type: "button", "aria-pressed": "false", text: "Table" });
    this.toggleBtn.addEventListener("click", () => {
      this.view = this.view === "chart" ? "table" : "chart";
      this.toggleBtn.setAttribute("aria-pressed", String(this.view === "table"));
      this.render();
    });
    const head = el(
      "div",
      { class: "card-head" },
      el("div", {}, el("div", { class: "card-title", text: spec.title }), el("div", { class: "card-sub", text: spec.sub || "" })),
      this.toggleBtn,
    );
    this.body = el("div", { class: "card-body" });
    this.tooltip = el("div", { class: "tooltip", hidden: true, role: "status" });
    this.fig = el("figure", { class: "card" }, head, this.body, this.tooltip);
    parent.append(this.fig);
    new ResizeObserver(() => this.render()).observe(this.body);
  }

  update(series, opts) {
    this.series = series;
    this.opts = opts;
    this.render();
  }

  visibleValues() {
    const vals = [];
    for (const s of this.series) {
      for (const v of s.smooth) if (v !== null && Number.isFinite(v)) vals.push(v);
      if (this.opts.showRaw) for (const v of s.ys) if (v !== null && Number.isFinite(v)) vals.push(v);
    }
    return vals;
  }

  render() {
    this.body.replaceChildren();
    this.tooltip.hidden = true;
    const hasData = this.series.some((s) => s.smooth.some((v) => v !== null));
    if (!hasData) {
      this.body.append(el("div", { class: "empty", text: this.opts.emptyText || "Waiting for data…" }));
      return;
    }
    if (this.view === "table") return this.renderTable();
    this.renderChart();
  }

  renderTable() {
    const xsAll = [...new Set(this.series.flatMap((s) => s.xs))].sort((a, b) => a - b);
    const maxRows = 150;
    const stride = Math.max(1, Math.ceil(xsAll.length / maxRows));
    const rows = xsAll.filter((_, i) => i % stride === 0 || i === xsAll.length - 1);
    const headRow = el("tr", {}, el("th", { text: "Iteration" }));
    for (const s of this.series) {
      headRow.append(el("th", { text: this.opts.smoothed ? `${s.label} (smoothed)` : s.label }));
      if (this.opts.smoothed) headRow.append(el("th", { text: `${s.label} (raw)` }));
    }
    const tbody = el("tbody");
    for (const x of rows.slice().reverse()) {
      const tr = el("tr", {}, el("td", { text: fmtIter(x) }));
      for (const s of this.series) {
        const i = indexOfX(s.xs, x);
        tr.append(el("td", { text: i >= 0 ? fmt(s.smooth[i]) : "–" }));
        if (this.opts.smoothed) tr.append(el("td", { text: i >= 0 ? fmt(s.ys[i]) : "–" }));
      }
      tbody.append(tr);
    }
    this.body.append(el("div", { class: "table-wrap" }, el("table", { class: "data" }, el("thead", {}, headRow), tbody)));
  }

  renderChart() {
    const W = Math.max(this.body.clientWidth, 260);
    const H = 230; // includes the x-axis band, so the card never scrolls
    const m = { l: 52, r: 64, t: 10, b: 26 };
    const pw = W - m.l - m.r;
    const ph = H - m.t - m.b;
    const log = this.opts.yscale === "log";

    let vals = this.visibleValues();
    if (log) vals = vals.filter((v) => v > 0);
    if (!vals.length) {
      this.body.append(el("div", { class: "empty", text: "No positive values for a log scale" }));
      return;
    }
    vals.sort((a, b) => a - b);
    // Robust top: early warm-up spikes would otherwise flatten the rest of the curve (lines clip at the edge).
    const smoothVals = this.series.flatMap((s) => s.smooth).filter((v) => v !== null && Number.isFinite(v) && (!log || v > 0));
    smoothVals.sort((a, b) => a - b);
    let lo = vals[0];
    let hi = Math.max(quantile(vals, 0.985), smoothVals.length ? smoothVals[smoothVals.length - 1] : vals[vals.length - 1]);
    if (!(hi > lo)) hi = lo === 0 ? 1 : lo * 1.5;

    let yTicks;
    let yMap;
    if (log) {
      lo = lo / 1.15;
      hi = hi * 1.15;
      yTicks = logTicks(lo, hi);
      const l0 = Math.log10(lo);
      const l1 = Math.log10(hi);
      yMap = (v) => m.t + ph - ((Math.log10(Math.max(v, lo * 1e-6)) - l0) / (l1 - l0)) * ph;
    } else {
      const pad = (hi - lo) * 0.06;
      lo = lo - pad;
      hi = hi + pad;
      yTicks = linearTicks(lo, hi, 4);
      yMap = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
    }

    const xsAll = [...new Set(this.series.flatMap((s) => s.xs))].sort((a, b) => a - b);
    this.xsAll = xsAll;
    let x0 = xsAll[0];
    let x1 = xsAll[xsAll.length - 1];
    if (x1 === x0) {
      // A single point: pad the domain, but never below iteration 0.
      x0 = Math.max(0, x0 - 1);
      x1 += 1;
    }
    const xMap = (x) => m.l + ((x - x0) / (x1 - x0)) * pw;
    this.geom = { m, pw, ph, xMap, yMap, x0, x1, W, H };

    const svg = svgEl("svg", {
      class: "plot",
      viewBox: `0 0 ${W} ${H}`,
      height: H,
      tabindex: 0,
      role: "img",
      "aria-label": `${this.spec.title}. Use the arrow keys to read values; the Table button lists them.`,
    });
    const clipId = `clip-${this.uid}`;
    const defs = svgEl("defs");
    const clip = svgEl("clipPath", { id: clipId });
    clip.append(svgEl("rect", { x: m.l, y: m.t - 2, width: pw, height: ph + 4 }));
    defs.append(clip);
    svg.append(defs);

    // Recessive hairline grid + tick labels.
    for (const v of yTicks) {
      const y = Math.round(yMap(v)) + 0.5;
      if (y < m.t - 1 || y > m.t + ph + 1) continue;
      svg.append(svgEl("line", { class: "gridline", x1: m.l, x2: m.l + pw, y1: y, y2: y }));
      const t = svgEl("text", { class: "tick", x: m.l - 6, y: y + 3.5, "text-anchor": "end" });
      t.textContent = fmt(v);
      svg.append(t);
    }
    const baseY = m.t + ph + 0.5;
    svg.append(svgEl("line", { class: "axisline", x1: m.l, x2: m.l + pw, y1: baseY, y2: baseY }));
    for (const v of linearTicks(x0, x1, Math.max(3, Math.floor(pw / 70)))) {
      if (v < x0 || v > x1) continue;
      const t = svgEl("text", { class: "tick", x: xMap(v), y: m.t + ph + 17, "text-anchor": "middle" });
      t.textContent = fmtIter(v);
      svg.append(t);
    }

    const plotG = svgEl("g", { "clip-path": `url(#${clipId})` });
    const pathFor = (xs, ys) => {
      let d = "";
      let pen = false;
      for (let i = 0; i < xs.length; i++) {
        const y = ys[i];
        if (y === null || !Number.isFinite(y) || (log && y <= 0)) {
          pen = false;
          continue;
        }
        d += `${pen ? "L" : "M"}${xMap(xs[i]).toFixed(1)},${yMap(y).toFixed(1)}`;
        pen = true;
      }
      return d;
    };
    if (this.opts.showRaw) {
      for (const s of this.series) plotG.append(svgEl("path", { class: `raw s${s.slot}`, d: pathFor(s.xs, s.ys) }));
    }
    for (const s of this.series) plotG.append(svgEl("path", { class: `line s${s.slot}`, d: pathFor(s.xs, s.smooth) }));
    if (this.opts.markers) {
      for (const s of this.series) {
        s.xs.forEach((x, i) => {
          const y = s.smooth[i];
          if (y === null || !Number.isFinite(y) || (log && y <= 0)) return;
          plotG.append(svgEl("circle", { class: `dot s${s.slot}`, cx: xMap(x), cy: yMap(y), r: 4 }));
        });
      }
    }
    svg.append(plotG);

    // End dots + direct value labels; drop the labels (not the dots) when they would collide.
    const ends = [];
    for (const s of this.series) {
      for (let i = s.xs.length - 1; i >= 0; i--) {
        const y = s.smooth[i];
        if (y !== null && Number.isFinite(y) && (!log || y > 0)) {
          ends.push({ s, x: xMap(s.xs[i]), y: Math.min(Math.max(yMap(y), m.t), m.t + ph), v: y });
          break;
        }
      }
    }
    const collide = ends.some((a, i) => ends.some((b, j) => j > i && Math.abs(a.y - b.y) < 13));
    for (const e of ends) {
      if (!this.opts.markers) svg.append(svgEl("circle", { class: `dot s${e.s.slot}`, cx: e.x, cy: e.y, r: 4 }));
      if (!collide) {
        const t = svgEl("text", { class: "endlabel", x: m.l + pw + 8, y: e.y + 3.5 });
        t.textContent = fmt(e.v);
        svg.append(t);
      }
    }

    // Crosshair + hover layer (the whole plot is the hit target; the crosshair snaps to the nearest iteration).
    this.cross = svgEl("line", { class: "crosshair", x1: 0, x2: 0, y1: m.t, y2: m.t + ph, visibility: "hidden" });
    svg.append(this.cross);
    const hit = svgEl("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
    svg.append(hit);
    hit.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = ((ev.clientX - rect.left) / rect.width) * W;
      const xv = x0 + ((px - m.l) / pw) * (x1 - x0);
      this.showAt(xsAll[bisectNearest(xsAll, xv)]);
    });
    hit.addEventListener("pointerleave", () => this.hide());
    svg.addEventListener("keydown", (ev) => {
      if (!xsAll.length) return;
      let i = this.hoverX === null ? xsAll.length - 1 : indexOfX(xsAll, this.hoverX);
      if (ev.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (ev.key === "ArrowRight") i = Math.min(xsAll.length - 1, i + 1);
      else if (ev.key === "Home") i = 0;
      else if (ev.key === "End") i = xsAll.length - 1;
      else if (ev.key === "Escape") return this.hide();
      else return;
      ev.preventDefault();
      this.showAt(xsAll[i]);
    });
    svg.addEventListener("focus", () => this.showAt(this.hoverX ?? xsAll[xsAll.length - 1]));
    svg.addEventListener("blur", () => this.hide());
    this.body.append(svg);
  }

  hide() {
    this.hoverX = null;
    this.tooltip.hidden = true;
    if (this.cross) this.cross.setAttribute("visibility", "hidden");
  }

  showAt(x) {
    if (x === undefined || x === null || !this.geom) return;
    this.hoverX = x;
    const { xMap, W } = this.geom;
    const px = xMap(x);
    this.cross.setAttribute("x1", px);
    this.cross.setAttribute("x2", px);
    this.cross.setAttribute("visibility", "visible");

    const tt = this.tooltip;
    tt.replaceChildren(el("div", { class: "tt-head", text: `${this.opts.xLabel || "Iteration"} ${fmtIter(x)}` }));
    for (const s of this.series) {
      const i = indexOfX(s.xs, x);
      const row = el("div", { class: "tt-row" }, lineKey(s.slot), el("span", { class: "tt-val", text: i >= 0 ? fmt(s.smooth[i]) : "–" }));
      row.append(el("span", { class: "tt-name", text: s.label }));
      if (this.opts.smoothed && this.opts.showRaw && i >= 0) row.append(el("span", { class: "tt-raw", text: `raw ${fmt(s.ys[i])}` }));
      tt.append(row);
    }
    tt.hidden = false;
    // The tooltip is positioned in the card (figure) frame; the plot body sits at its own offset inside it.
    const bodyWidth = this.body.getBoundingClientRect().width;
    const pxInBody = px * (bodyWidth / W);
    const width = tt.offsetWidth;
    const right = pxInBody + 12;
    const leftOfCross = pxInBody - 12 - width;
    const ttX = right + width <= bodyWidth ? right : Math.max(0, leftOfCross);
    tt.style.left = `${this.body.offsetLeft + ttX}px`;
    tt.style.top = `${this.body.offsetTop + 8}px`;
  }
}

// ---------- series assembly ----------
function runSeries(run, slot, key, smoothing) {
  const it = run.metrics.iteration || [];
  const col = run.metrics[key] || [];
  const xs = [];
  const ys = [];
  // Only the iterations that logged this metric: alternating updates (e.g. DMD2's student and fake-score
  // steps) log different metrics, and gaps would otherwise break the line at every point.
  for (let i = 0; i < it.length; i++) {
    const v = col[i];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    xs.push(it[i]);
    ys.push(v);
  }
  return { label: run.info.label, slot, xs, ys, smooth: smoothSeries(ys, smoothing) };
}
function evalSeries(run, slot, nfe) {
  const xs = [];
  const ys = [];
  for (const e of run.evals) {
    const v = e.consistency_mse ? e.consistency_mse[String(nfe)] : undefined;
    if (v === undefined) continue;
    xs.push(e.iteration);
    ys.push(v === null || !Number.isFinite(v) ? null : v);
  }
  return { label: run.info.label, slot, xs, ys, smooth: ys.slice() };
}

// ---------- page sections ----------
let lossCharts = [];
let lossSpecs = [];
let lossSpecKey = "";
function currentLossSpecs() {
  const withCharts = state.runs.find((run) => Array.isArray(run.info.charts) && run.info.charts.length);
  return withCharts ? withCharts.info.charts : LOSS_CHARTS;
}
function ensureLossCharts() {
  const specs = currentLossSpecs();
  const key = JSON.stringify(specs);
  if (key === lossSpecKey) return;
  lossSpecKey = key;
  lossSpecs = specs;
  const container = document.getElementById("loss-charts");
  container.replaceChildren();
  lossCharts = specs.map((spec) => new LineChart(container, spec));
  const custom = specs !== LOSS_CHARTS;
  document.getElementById("loss-h").textContent = custom ? "Training losses" : "Training losses by (t, r) bucket";
  document.getElementById("loss-desc").hidden = custom;
}
const consCharts = NFES_CONSISTENCY.map(
  (k) => new LineChart(document.getElementById("cons-charts"), { title: `NFE ${k} vs 16`, sub: "latent MSE to the 16-step sample" }),
);

function renderLegends() {
  for (const id of ["legend-loss", "legend-cons", "legend-samples"]) {
    const node = document.getElementById(id);
    node.replaceChildren(...state.runs.map((r, i) => el("span", {}, lineKey(i + 1), document.createTextNode(r.info.label))));
  }
}

function renderTiles() {
  const node = document.getElementById("run-tiles");
  const now = Date.now() / 1000;
  node.replaceChildren(
    ...state.runs.map((run, i) => {
      const it = run.metrics.iteration || [];
      const last = it.length ? it[it.length - 1] : null;
      const spi = (run.metrics.sec_per_iter || []).filter((v) => v !== null && Number.isFinite(v)).slice(-20);
      const spiMed = spi.length ? spi.slice().sort((a, b) => a - b)[Math.floor(spi.length / 2)] : null;
      const maxIter = run.info.max_iter;
      const wall = (run.metrics.wall_time || []).filter((v) => v !== null);
      const lastWall = wall.length ? wall[wall.length - 1] : null;
      const evalsDone = run.evals.length;
      const tile = el("div", { class: "tile" });
      tile.append(el("div", { class: "tile-head" }, lineKey(i + 1), el("span", { text: run.info.label })));
      const stats = el("div", { class: "tile-stats" });
      const stat = (label, value) => el("div", {}, el("div", { class: "stat-label", text: label }), el("div", { class: "stat-value", text: value }));
      stats.append(stat("Iteration", last === null ? "–" : maxIter ? `${fmtIter(last)} / ${fmtIter(maxIter)}` : fmtIter(last)));
      stats.append(stat("Seconds / iter", spiMed === null ? "–" : spiMed.toFixed(1)));
      stats.append(stat("ETA", spiMed !== null && maxIter && last !== null ? fmtDuration((maxIter - last) * spiMed) : "–"));
      stats.append(stat("Evals", String(evalsDone)));
      tile.append(stats);
      if (run.info.description) tile.append(el("div", { class: "tile-note", text: run.info.description }));
      if (lastWall !== null) {
        const age = now - lastWall;
        const done = maxIter && last !== null && last >= maxIter;
        const cls = !done && age > 900 ? "tile-note status-warn" : "tile-note";
        const msg = done ? "Finished" : age > 900 ? `No new iteration for ${fmtDuration(age)}` : `Last iteration ${fmtDuration(age)} ago`;
        tile.append(el("div", { class: cls, text: msg }));
      } else {
        tile.append(el("div", { class: "tile-note", text: "Waiting for the first logged iteration" }));
      }
      return tile;
    }),
  );
}

function renderCharts() {
  const opts = { yscale: state.yscale, showRaw: state.showRaw, smoothed: true, emptyText: "Waiting for metrics…" };
  ensureLossCharts();
  lossSpecs.forEach((spec, ci) => {
    lossCharts[ci].update(
      state.runs.map((run, i) => runSeries(run, i + 1, spec.key, state.smooth)),
      opts,
    );
  });
  NFES_CONSISTENCY.forEach((k, ci) => {
    consCharts[ci].update(
      state.runs.map((run, i) => evalSeries(run, i + 1, k)),
      { yscale: state.yscale, showRaw: false, smoothed: false, markers: true, xLabel: "Iteration", emptyText: "Waiting for evals…" },
    );
  });
}

// ---------- eval samples ----------
function evalIterations() {
  return [...new Set(state.runs.flatMap((r) => r.evals.map((e) => e.iteration)))].sort((a, b) => a - b);
}
function evalAtOrBefore(run, iteration) {
  let best = null;
  for (const e of run.evals) if (e.iteration <= iteration && (!best || e.iteration > best.iteration)) best = e;
  return best;
}

const lightbox = document.getElementById("lightbox");
function openLightbox(src, caption) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-img").alt = caption;
  document.getElementById("lightbox-cap").textContent = caption;
  lightbox.hidden = false;
}
lightbox.addEventListener("click", () => (lightbox.hidden = true));
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") lightbox.hidden = true;
});

function renderSamples() {
  const iters = evalIterations();
  const slider = document.getElementById("iter-slider");
  const label = document.getElementById("iter-label");
  const out = document.getElementById("samples");
  if (!iters.length) {
    slider.max = 0;
    label.textContent = "No evals yet";
    if (state.sampleSignature !== "none") out.replaceChildren(el("div", { class: "empty", text: "The first eval runs before training starts." }));
    state.sampleSignature = "none";
    return;
  }
  slider.max = iters.length - 1;
  if (state.followLatest || state.sliderIndex === null || state.sliderIndex > iters.length - 1) state.sliderIndex = iters.length - 1;
  slider.value = state.sliderIndex;
  const iteration = iters[state.sliderIndex];
  label.textContent = `Iteration ${fmtIter(iteration)} (${state.sliderIndex + 1} of ${iters.length})`;

  const chosen = state.runs.map((run) => evalAtOrBefore(run, iteration));
  const signature = JSON.stringify([iteration, chosen.map((e) => (e ? e.dir : null)), state.runs.map((r) => r.info.label)]);
  if (signature === state.sampleSignature) return;
  state.sampleSignature = signature;

  const ref = chosen.find((e) => e) || null;
  const prompts = ref ? ref.prompts || [] : [];
  const tags = ref ? ref.prompt_tags || [] : [];
  const nfes = ref ? ref.nfes || [1, 2, 4, 8, 16] : [1, 2, 4, 8, 16];
  const blocks = prompts.map((prompt, j) => {
    const grid = el("div", { class: "sample-grid", style: `--ncols:${nfes.length}` });
    grid.append(el("div"));
    for (const k of nfes) grid.append(el("div", { class: "col-head", text: `NFE ${k}` }));
    state.runs.forEach((run, i) => {
      const e = chosen[i];
      const head = el("div", { class: "row-head" }, el("span", { class: "run-name" }, lineKey(i + 1), document.createTextNode(run.info.label)));
      const weights = e && e.weights === "ema" ? " · EMA" : "";
      head.append(el("span", { text: e ? (e.iteration === iteration ? `iteration ${fmtIter(e.iteration)}` : `latest: ${fmtIter(e.iteration)}`) + weights : "no eval yet" }));
      grid.append(head);
      for (const k of nfes) {
        if (!e) {
          grid.append(el("div", { class: "missing", text: "–" }));
          continue;
        }
        const src = `/media/${encodeURIComponent(run.key)}/${encodeURIComponent(e.dir)}/p${j}_nfe${k}.jpg`;
        const caption = `${run.info.label} · NFE ${k} · iteration ${fmtIter(e.iteration)}${e.weights === "ema" ? " (EMA)" : ""} · ${prompt}`;
        const btn = el("button", { class: "thumb", type: "button", "aria-label": `Enlarge: ${caption}` }, el("img", { src, alt: caption, loading: "lazy" }));
        btn.addEventListener("click", () => openLightbox(src, caption));
        grid.append(btn);
      }
    });
    const text = el("div", { class: "prompt-text" }, document.createTextNode(prompt));
    if (tags[j]) text.append(el("span", { class: "prompt-tag", text: `(${tags[j]})` }));
    return el("div", { class: "prompt-block" }, text, grid);
  });
  out.replaceChildren(...blocks);
}

// ---------- controls ----------
document.getElementById("smooth").addEventListener("input", (ev) => {
  state.smooth = Number(ev.target.value);
  document.getElementById("smooth-val").textContent = state.smooth.toFixed(2);
  renderCharts();
});
for (const btn of document.querySelectorAll("[data-yscale]")) {
  btn.addEventListener("click", () => {
    state.yscale = btn.dataset.yscale;
    for (const b of document.querySelectorAll("[data-yscale]")) b.setAttribute("aria-pressed", String(b === btn));
    renderCharts();
  });
}
document.getElementById("show-raw").addEventListener("change", (ev) => {
  state.showRaw = ev.target.checked;
  renderCharts();
});
document.getElementById("iter-slider").addEventListener("input", (ev) => {
  state.sliderIndex = Number(ev.target.value);
  state.followLatest = state.sliderIndex === evalIterations().length - 1;
  document.getElementById("follow-latest").checked = state.followLatest;
  renderSamples();
});
document.getElementById("iter-prev").addEventListener("click", () => {
  if (state.sliderIndex > 0) state.sliderIndex -= 1;
  state.followLatest = false;
  document.getElementById("follow-latest").checked = false;
  renderSamples();
});
document.getElementById("iter-next").addEventListener("click", () => {
  const n = evalIterations().length;
  if (state.sliderIndex < n - 1) state.sliderIndex += 1;
  state.followLatest = state.sliderIndex === n - 1;
  document.getElementById("follow-latest").checked = state.followLatest;
  renderSamples();
});
document.getElementById("follow-latest").addEventListener("change", (ev) => {
  state.followLatest = ev.target.checked;
  renderSamples();
});

const THEMES = ["auto", "light", "dark"];
function applyTheme(theme) {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  document.getElementById("theme-btn").textContent = `Theme: ${theme}`;
  try {
    localStorage.setItem("flowmap-dashboard-theme", theme);
  } catch (_) {
    /* storage may be unavailable */
  }
}
document.getElementById("theme-btn").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || "auto";
  applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
});
try {
  applyTheme(localStorage.getItem("flowmap-dashboard-theme") || "auto");
} catch (_) {
  applyTheme("auto");
}

// ---------- polling (refetch keeps the previous render; no skeleton flash) ----------
async function refresh() {
  const updated = document.getElementById("updated");
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.runs.length > MAX_SERIES) {
      updated.textContent = `Showing the first ${MAX_SERIES} of ${data.runs.length} runs`;
    }
    state.runs = data.runs.slice(0, MAX_SERIES);
    for (const node of document.querySelectorAll(".card-body")) node.classList.remove("stale");
    renderLegends();
    renderTiles();
    renderCharts();
    renderSamples();
    if (data.runs.length <= MAX_SERIES) updated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    for (const node of document.querySelectorAll(".card-body")) node.classList.add("stale");
    updated.textContent = `Server unreachable (${err.message}); retrying`;
  }
}
refresh();
setInterval(refresh, REFRESH_MS);
