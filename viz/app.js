/* ============================================================================
   Cell Plotter — interactive SVG renderer for the eyeball-probe geometry.

   Coordinates are LOCAL cell units (~0..256), not geographic — so we draw with a
   plain custom SVG renderer (no map library). SVG y grows downward while the data's
   y grows upward, so every point is plotted at (x, -y) and the viewBox lives in that
   flipped "svg space".
   ========================================================================== */
(function () {
  "use strict";

  const D = window.PROBE_DATA;
  const SVGNS = "http://www.w3.org/2000/svg";

  // class -> display label + colour (kept in sync with styles.css)
  const CLASS = {
    building_sealed:   { label: "building · sealed",      color: "#ff7a5c" },
    building_unsealed: { label: "building · near-closed", color: "#ff7a5c" },
    road:              { label: "road",                   color: "#38c4d6" },
    road_node:         { label: "road node",              color: "#ffc857" },
  };
  const CLASS_ORDER = ["building_sealed", "building_unsealed", "road", "road_node"];

  const state = {
    mode: "inspect",
    ctx: D.contexts[0],
    cell: 0,
    compareIndex: 2,
    layers: new Set(CLASS_ORDER), // visible classes
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
  const svgEl = (tag) => document.createElementNS(SVGNS, tag);

  const cellsOf = (ctx) => D.cells.filter((c) => c.context === ctx);
  const getCell = (ctx, idx) => cellsOf(ctx).find((c) => c.cell_index === idx) || cellsOf(ctx)[0];

  // ── geometry → svg ────────────────────────────────────────────────────────
  function pathD(feat) {
    if (feat.type === "Polygon") {
      const ring = feat.coords[0];
      return "M" + ring.map((p) => `${p[0]} ${-p[1]}`).join("L") + "Z";
    }
    // LineString / MultiLineString flattened to one polyline
    const pts = feat.type === "MultiLineString" ? feat.coords.flat() : feat.coords;
    return "M" + pts.map((p) => `${p[0]} ${-p[1]}`).join("L");
  }

  function unionBBox(cells) {
    let [a, b, c, d] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const cell of cells) {
      a = Math.min(a, cell.bbox[0]); b = Math.min(b, cell.bbox[1]);
      c = Math.max(c, cell.bbox[2]); d = Math.max(d, cell.bbox[3]);
    }
    return [a, b, c, d];
  }

  // bbox (data space) -> viewBox object (flipped svg space), with padding + square fit
  function fitViewBox(bbox, square) {
    let [minx, miny, maxx, maxy] = bbox;
    let w = Math.max(maxx - minx, 1), h = Math.max(maxy - miny, 1);
    if (square) {
      const s = Math.max(w, h);
      minx -= (s - w) / 2; miny -= (s - h) / 2; w = s; h = s;
    }
    const pad = Math.max(w, h) * 0.08;
    return { x: minx - pad, y: -(maxy + pad), w: w + 2 * pad, h: h + 2 * pad };
  }
  const applyVB = (svg, vb) => svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.w} ${vb.h}`);

  // render the features of a cell into an svg group (returns the <g>)
  function renderCell(svg, cell, { animate } = {}) {
    svg.querySelectorAll("g.geoms").forEach((g) => g.remove());
    const g = svgEl("g");
    g.setAttribute("class", "geoms");
    const refDiag = Math.hypot(cell.bbox[2] - cell.bbox[0], cell.bbox[3] - cell.bbox[1]) || 1;
    // draw order: roads first, then buildings, then nodes on top
    const draw = (clsFilter) =>
      cell.features.forEach((f, i) => {
        if (!clsFilter(f.cls)) return;
        const visible = state.layers.has(f.cls);
        let node;
        if (f.cls === "road_node") {
          node = svgEl("circle");
          node.setAttribute("cx", f.coords[0]);
          node.setAttribute("cy", -f.coords[1]);
          node.setAttribute("r", refDiag * 0.012);
        } else {
          node = svgEl("path");
          node.setAttribute("d", pathD(f));
          node.setAttribute("vector-effect", "non-scaling-stroke");
        }
        node.setAttribute("class", `geom ${f.cls}${visible ? "" : " layer-hidden"}`);
        node.dataset.cls = f.cls;
        node.dataset.info = f.cls === "building_unsealed" && f.gap != null
          ? `closure gap ${(f.gap * 100).toFixed(1)}% of bbox`
          : "";
        g.appendChild(node);
        if (animate && visible) {
          node.style.animationDelay = Math.min(i * 14, 600) + "ms";
          if (f.cls === "road" || f.cls === "building_sealed") {
            // stroke-draw: dasharray = path length, offset animates length -> 0
            node.style.setProperty("--len", node.getTotalLength());
            node.classList.add("draw-stroke");
          } else if (f.cls === "road_node") {
            node.classList.add("draw-pop");
          } else {
            node.classList.add("draw-fade"); // unsealed: fade in, keep the 4 3 dash
          }
        }
      });
    draw((c) => c === "road");
    draw((c) => c === "building_unsealed" || c === "building_sealed");
    draw((c) => c === "road_node");
    svg.appendChild(g);
    return g;
  }

  // ── zoom / pan on the inspect plot ────────────────────────────────────────
  function attachPanZoom(svg, getVB, setVB) {
    let dragging = false, last = null;
    const toSvg = (cx, cy) => {
      const r = svg.getBoundingClientRect(); const vb = getVB();
      return { x: vb.x + ((cx - r.left) / r.width) * vb.w, y: vb.y + ((cy - r.top) / r.height) * vb.h };
    };
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const vb = { ...getVB() };
      const f = e.deltaY > 0 ? 1.12 : 1 / 1.12;
      const p = toSvg(e.clientX, e.clientY);
      vb.x = p.x - (p.x - vb.x) * f; vb.y = p.y - (p.y - vb.y) * f;
      vb.w *= f; vb.h *= f;
      setVB(vb); applyVB(svg, vb);
      fadeHint();
    }, { passive: false });
    svg.addEventListener("pointerdown", (e) => { dragging = true; last = { x: e.clientX, y: e.clientY }; svg.setPointerCapture(e.pointerId); });
    svg.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const vb = { ...getVB() }; const r = svg.getBoundingClientRect();
      vb.x -= ((e.clientX - last.x) / r.width) * vb.w;
      vb.y -= ((e.clientY - last.y) / r.height) * vb.h;
      last = { x: e.clientX, y: e.clientY };
      setVB(vb); applyVB(svg, vb);
    });
    const stop = () => { dragging = false; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
  }
  let hintFaded = false;
  function fadeHint() { if (hintFaded) return; const h = $("#plot-hint"); if (h) h.style.opacity = 0; hintFaded = true; }

  // ── hover tooltip / highlight (event delegation) ──────────────────────────
  const tip = $("#tooltip");
  function wireHover(svg) {
    svg.addEventListener("pointermove", (e) => {
      const t = e.target.closest(".geom");
      if (!t || t.classList.contains("layer-hidden")) { clearHot(svg); return; }
      svg.querySelectorAll(".geom").forEach((n) => n.classList.toggle("dim", n !== t));
      t.classList.add("hot");
      const info = t.dataset.info;
      tip.innerHTML = `<span>${CLASS[t.dataset.cls].label}</span>${info ? `<br><span class="tc">${info}</span>` : ""}`;
      tip.hidden = false; tip.style.left = e.clientX + "px"; tip.style.top = e.clientY + "px";
    });
    svg.addEventListener("pointerleave", () => clearHot(svg));
  }
  function clearHot(svg) {
    svg.querySelectorAll(".geom").forEach((n) => n.classList.remove("dim", "hot"));
    tip.hidden = true;
  }

  // ── inspect view ──────────────────────────────────────────────────────────
  const plot = $("#plot");
  let inspectVB = null;
  attachPanZoom(plot, () => inspectVB, (vb) => { inspectVB = vb; });
  wireHover(plot);
  $("#zoom-in").onclick = () => { const v = { ...inspectVB }; v.x += v.w * 0.06; v.y += v.h * 0.06; v.w *= 0.88; v.h *= 0.88; inspectVB = v; applyVB(plot, v); };
  $("#zoom-out").onclick = () => { const v = { ...inspectVB }; v.x -= v.w * 0.07; v.y -= v.h * 0.07; v.w *= 1.14; v.h *= 1.14; inspectVB = v; applyVB(plot, v); };
  $("#zoom-reset").onclick = () => { inspectVB = fitViewBox(getCell(state.ctx, state.cell).bbox, true); applyVB(plot, inspectVB); };

  function fmt(n) {
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k";
    return Math.round(n).toString();
  }

  function renderInspect(animate) {
    const cell = getCell(state.ctx, state.cell);
    $("#insp-ctx-pill").textContent = (D.context_labels[state.ctx] || state.ctx).toLowerCase();
    $("#insp-cell-id").textContent = `cell ${cell.cell_index}  ·  seed ${cell.gen_seed}`;
    inspectVB = fitViewBox(cell.bbox, true);
    applyVB(plot, inspectVB);
    renderCell(plot, cell, { animate });
    renderConditioning(cell);
    renderChar(cell);
    renderOutput(cell);
    renderLegend(cell);
  }

  function renderConditioning(cell) {
    const grid = $("#cond-grid"); grid.innerHTML = "";
    const dims = [
      ["zoning", cell.stratum[0]],
      ["road_skeleton", cell.stratum[1]],
      ["density", cell.stratum[2]],
      ["coastal", cell.stratum[3]],
    ];
    for (const [key, val] of dims) {
      const dim = D.dimensions[key];
      const c = el("div", "cond-cell");
      let display, gloss;
      if (dim.kind === "ordinal") {
        const b = dim.buckets[val] || {};
        display = `${val}`; gloss = `${b.gloss || ""}<br><span style="color:var(--text-faint)">${b.range || ""}</span>`;
      } else {
        display = dim.values[val] ?? val; gloss = dim.note ? "" : "";
      }
      c.innerHTML = `<div class="k">${dim.label.split("(")[0].trim()}</div><div class="v">${display}</div><div class="g">${gloss}</div>`;
      if (dim.kind === "ordinal") {
        const m = el("div", "meter"); const s = el("span"); s.style.width = (val / 3) * 100 + "%"; m.appendChild(s); c.appendChild(m);
      }
      grid.appendChild(c);
    }
  }

  function renderChar(cell) {
    const list = $("#char-list"); list.innerHTML = "";
    cell.char_decoded.forEach((ch) => {
      const row = el("div", "char-row" + (ch.unit === "flag" ? " flag" : ""));
      let v;
      if (ch.unit === "flag") v = ch.value ? "yes" : "no";
      else if (ch.unit === "") v = `${fmt(ch.value)}`;
      else v = `${ch.value < 100 ? ch.value : fmt(ch.value)}<small>${ch.unit}</small>`;
      row.innerHTML = `<span class="cn">${ch.name}</span><span class="cv">${v}</span>`;
      list.appendChild(row);
    });
  }

  function renderOutput(cell) {
    const term = cell.self_terminated;
    $("#stat-row").innerHTML = `
      <div class="stat"><div class="sv">${fmt(cell.n_tokens)}</div><div class="sk">tokens</div></div>
      <div class="stat"><div class="sv ${term ? "ok" : ""}">${term ? "✓" : "cap"}</div><div class="sk">self-term</div></div>
      <div class="stat"><div class="sv">${Math.round(cell.decodability * 100)}%</div><div class="sk">decodable</div></div>`;
    const counts = $("#counts"); counts.innerHTML = "";
    const max = Math.max(1, ...CLASS_ORDER.map((k) => cell.counts[k] || 0));
    CLASS_ORDER.forEach((k) => {
      const n = cell.counts[k] || 0;
      const bar = el("div", "count-bar");
      bar.innerHTML = `<span class="dot" style="background:${CLASS[k].color};${k === "building_unsealed" ? "outline:1.5px dashed " + CLASS[k].color + ";outline-offset:-2px;background:transparent;" : ""}"></span>
        <span class="track"><i style="width:${(n / max) * 100}%;background:${CLASS[k].color};${k === "building_unsealed" ? "opacity:.55" : ""}"></i></span>
        <span class="n">${n}</span>`;
      counts.appendChild(bar);
    });
  }

  function renderLegend(cell) {
    const lg = $("#legend"); lg.innerHTML = "";
    CLASS_ORDER.forEach((k) => {
      const li = el("li"); if (!state.layers.has(k)) li.classList.add("off");
      const sw = el("span", "swatch");
      if (k === "building_unsealed") { sw.style.background = "transparent"; sw.style.border = `1.5px dashed ${CLASS[k].color}`; }
      else sw.style.background = CLASS[k].color;
      li.appendChild(sw);
      li.appendChild(document.createTextNode(CLASS[k].label));
      const cnt = el("span", "count"); cnt.textContent = `${cell.counts[k] || 0}`; li.appendChild(cnt);
      li.onclick = () => { state.layers.has(k) ? state.layers.delete(k) : state.layers.add(k); renderInspect(false); if (state.mode === "compare") renderCompare(); };
      lg.appendChild(li);
    });
  }

  function buildInspectPickers() {
    const seg = $("#ctx-seg"); seg.innerHTML = "";
    D.contexts.forEach((ctx) => {
      const b = el("button"); b.textContent = (D.context_labels[ctx] || ctx);
      if (ctx === state.ctx) b.classList.add("on");
      b.onclick = () => { state.ctx = ctx; state.cell = cellsOf(ctx)[0].cell_index; buildInspectPickers(); renderInspect(true); };
      seg.appendChild(b);
    });
    const cs = $("#cell-seg"); cs.innerHTML = "";
    cellsOf(state.ctx).forEach((c) => {
      const b = el("button"); b.textContent = c.cell_index;
      if (c.cell_index === state.cell) b.classList.add("on");
      b.onclick = () => { state.cell = c.cell_index; buildInspectPickers(); renderInspect(true); };
      cs.appendChild(b);
    });
  }

  // ── compare view ──────────────────────────────────────────────────────────
  function renderCompare() {
    const idx = state.compareIndex;
    const shown = D.contexts.map((ctx) => getCell(ctx, idx));
    const shared = fitViewBox(unionBBox(shown), true);
    const grid = $("#compare-grid"); grid.innerHTML = "";
    shown.forEach((cell) => {
      const col = el("div", "compare-col");
      const nbuild = cell.counts.building_sealed + cell.counts.building_unsealed;
      col.innerHTML = `<header><span class="nm">${D.context_labels[cell.context]}</span>
          <span class="tag">density ${cell.stratum[2]} · skel ${cell.stratum[1]}</span></header>`;
      const svg = svgEl("svg"); svg.setAttribute("class", "compare-svg");
      col.appendChild(svg);
      const readout = el("div", "readout");
      readout.innerHTML = `
        <div><div class="rv">${fmt(cell.n_tokens)}</div><div class="rk">tokens</div></div>
        <div><div class="rv">${nbuild}</div><div class="rk">buildings</div></div>
        <div><div class="rv">${cell.counts.road}</div><div class="rk">roads</div></div>`;
      col.appendChild(readout);
      grid.appendChild(col);
      applyVB(svg, shared);
      renderCell(svg, cell, { animate: true });
      wireHover(svg);
    });
  }

  function buildComparePicker() {
    const wrap = $("#compare-cellpick"); wrap.innerHTML = "";
    const lbl = el("span", "lbl"); lbl.textContent = "cell index"; wrap.appendChild(lbl);
    const n = Math.max(...D.contexts.map((c) => cellsOf(c).length));
    for (let i = 0; i < n; i++) {
      const b = el("button"); b.textContent = i; if (i === state.compareIndex) b.classList.add("on");
      b.onclick = () => { state.compareIndex = i; buildComparePicker(); renderCompare(); };
      wrap.appendChild(b);
    }
  }

  function renderSummary() {
    const tb = $("#summary-table tbody"); tb.innerHTML = "";
    D.summary.forEach((s) => {
      const tr = el("tr");
      tr.innerHTML = `<td class="ctx">${D.context_labels[s.context] || s.context}</td>
        <td>${s.n_cells}</td><td>${s.med_tokens}</td><td>${s.med_buildings}</td><td>${s.med_roads}</td>`;
      tb.appendChild(tr);
    });
  }

  // ── mode switching ────────────────────────────────────────────────────────
  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("is-active", b.dataset.mode === mode));
    $("#view-inspect").classList.toggle("is-active", mode === "inspect");
    $("#view-compare").classList.toggle("is-active", mode === "compare");
    if (mode === "inspect") renderInspect(true);
    else { buildComparePicker(); renderCompare(); renderSummary(); }
  }
  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));

  // ── provenance footer ─────────────────────────────────────────────────────
  $("#provenance").innerHTML = `
    <b>Source</b> reports/_eyeball_probe/ · 21 pre-generated cells (PI-approved eyeball probe, NON-scored).
    <b>Model</b> ${D.meta.backbone} · d${D.meta.d_model}/${D.meta.n_layers}L · ~53M params · step ${fmt(D.meta.global_step)} · train=${D.meta.train_set} · conditioning=${D.meta.conditioning_scheme} (${D.meta.conditioning_ablation}).
    <b>Honesty</b> coordinates are local cell units, not geographic; building footprints that don't seal to exact float-equality are shown as <em>near-closed</em> (a known over-strict closure check, deferred). Character-stats moments are fed to the model — the conditioning response is partly echo. This validates the methodology; it is not a realism or architecture claim.`;

  // ── boot ──────────────────────────────────────────────────────────────────
  buildInspectPickers();
  renderInspect(true);
})();
