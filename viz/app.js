/* ============================================================================
   Cell Plotter — interactive SVG renderer for the eyeball-probe geometry.

   Coordinates are LOCAL cell units (~0..256), not geographic — so we draw with a
   plain custom SVG renderer (no map library). SVG y grows downward while the data's
   y grows upward, so every point is plotted at (x, -y) and the viewBox lives in that
   flipped "svg space".

   Two backbones (transformer, mamba) are loaded as separate globals; they share the
   SAME conditioning / contexts / seeds, so only the generated geometry differs.
   ========================================================================== */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";

  // backbone registry — mamba is optional (viz still works transformer-only)
  const DATA = { transformer: window.PROBE_DATA, mamba: window.PROBE_DATA_MAMBA || null };
  const BACKBONES = ["transformer", "mamba"].filter((k) => DATA[k]);
  const BB_LABEL = { transformer: "Transformer", mamba: "Mamba" };
  const META = DATA.transformer; // labels/dimensions/contexts are backbone-independent

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
    backbone: "transformer",
    ctx: META.contexts[0],
    cell: 0,
    compareIndex: 2,
    layers: new Set(CLASS_ORDER),
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
  const svgEl = (tag) => document.createElementNS(SVGNS, tag);

  const cellsOf = (ctx, bb) => DATA[bb].cells.filter((c) => c.context === ctx);
  const getCell = (ctx, idx, bb) => cellsOf(ctx, bb).find((c) => c.cell_index === idx) || cellsOf(ctx, bb)[0];
  const nbuild = (c) => c.counts.building_sealed + c.counts.building_unsealed;

  // ── geometry → svg ────────────────────────────────────────────────────────
  function pathD(feat) {
    if (feat.type === "Polygon") {
      const ring = feat.coords[0];
      return "M" + ring.map((p) => `${p[0]} ${-p[1]}`).join("L") + "Z";
    }
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

  function renderCell(svg, cell, { animate } = {}) {
    svg.querySelectorAll("g.geoms").forEach((g) => g.remove());
    const g = svgEl("g");
    g.setAttribute("class", "geoms");
    const refDiag = Math.hypot(cell.bbox[2] - cell.bbox[0], cell.bbox[3] - cell.bbox[1]) || 1;
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
            node.style.setProperty("--len", node.getTotalLength());
            node.classList.add("draw-stroke");
          } else if (f.cls === "road_node") {
            node.classList.add("draw-pop");
          } else {
            node.classList.add("draw-fade");
          }
        }
      });
    draw((c) => c === "road");
    draw((c) => c === "building_unsealed" || c === "building_sealed");
    draw((c) => c === "road_node");
    svg.appendChild(g);
    return g;
  }

  // ── zoom / pan ────────────────────────────────────────────────────────────
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

  // ── hover tooltip / highlight ─────────────────────────────────────────────
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

  function fmt(n) {
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1) + "k";
    return Math.round(n).toString();
  }

  // ── inspect view ──────────────────────────────────────────────────────────
  const plot = $("#plot");
  let inspectVB = null;
  attachPanZoom(plot, () => inspectVB, (vb) => { inspectVB = vb; });
  wireHover(plot);
  const fitInspect = () => { inspectVB = fitViewBox(getCell(state.ctx, state.cell, state.backbone).bbox, true); applyVB(plot, inspectVB); };
  $("#zoom-in").onclick = () => { const v = { ...inspectVB }; v.x += v.w * 0.06; v.y += v.h * 0.06; v.w *= 0.88; v.h *= 0.88; inspectVB = v; applyVB(plot, v); };
  $("#zoom-out").onclick = () => { const v = { ...inspectVB }; v.x -= v.w * 0.07; v.y -= v.h * 0.07; v.w *= 1.14; v.h *= 1.14; inspectVB = v; applyVB(plot, v); };
  $("#zoom-reset").onclick = fitInspect;

  function renderInspect(animate) {
    const cell = getCell(state.ctx, state.cell, state.backbone);
    $("#insp-ctx-pill").textContent = (META.context_labels[state.ctx] || state.ctx).toLowerCase();
    $("#insp-cell-id").textContent = `${BB_LABEL[state.backbone]} · cell ${cell.cell_index} · seed ${cell.gen_seed}`;
    fitInspect();
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
      const dim = META.dimensions[key];
      const c = el("div", "cond-cell");
      let display, gloss = "";
      if (dim.kind === "ordinal") {
        const b = dim.buckets[val] || {};
        display = `${val}`; gloss = `${b.gloss || ""}<br><span style="color:var(--text-faint)">${b.range || ""}</span>`;
      } else {
        display = dim.values[val] ?? val;
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
      const dotStyle = k === "building_unsealed"
        ? `outline:1.5px dashed ${CLASS[k].color};outline-offset:-2px;background:transparent;`
        : `background:${CLASS[k].color};`;
      bar.innerHTML = `<span class="dot" style="${dotStyle}"></span>
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
      li.onclick = () => { state.layers.has(k) ? state.layers.delete(k) : state.layers.add(k); renderInspect(false); };
      lg.appendChild(li);
    });
  }

  function buildInspectPickers() {
    const bb = $("#bb-seg"); bb.innerHTML = "";
    bb.style.display = BACKBONES.length > 1 ? "" : "none";
    bb.style.gridTemplateColumns = `repeat(${BACKBONES.length}, 1fr)`;
    BACKBONES.forEach((k) => {
      const b = el("button"); b.textContent = BB_LABEL[k];
      if (k === state.backbone) b.classList.add("on");
      b.onclick = () => { state.backbone = k; buildInspectPickers(); renderInspect(true); };
      bb.appendChild(b);
    });
    const seg = $("#ctx-seg"); seg.innerHTML = "";
    META.contexts.forEach((ctx) => {
      const b = el("button"); b.textContent = META.context_labels[ctx] || ctx;
      if (ctx === state.ctx) b.classList.add("on");
      b.onclick = () => { state.ctx = ctx; state.cell = cellsOf(ctx, state.backbone)[0].cell_index; buildInspectPickers(); renderInspect(true); };
      seg.appendChild(b);
    });
    const cs = $("#cell-seg"); cs.innerHTML = "";
    cellsOf(state.ctx, state.backbone).forEach((c) => {
      const b = el("button"); b.textContent = c.cell_index;
      if (c.cell_index === state.cell) b.classList.add("on");
      b.onclick = () => { state.cell = c.cell_index; buildInspectPickers(); renderInspect(true); };
      cs.appendChild(b);
    });
  }

  // ── compare view (transformer vs mamba, same ctx + cell + seed) ────────────
  function renderCompare() {
    const idx = state.compareIndex, ctx = state.ctx;
    const shown = BACKBONES.map((bb) => ({ bb, cell: getCell(ctx, idx, bb) }));
    const shared = fitViewBox(unionBBox(shown.map((s) => s.cell)), true);
    const grid = $("#compare-grid"); grid.innerHTML = "";
    grid.style.gridTemplateColumns = `repeat(${shown.length}, 1fr)`;
    shown.forEach(({ bb, cell }) => {
      const col = el("div", "compare-col");
      const term = cell.self_terminated;
      col.innerHTML = `<header><span class="nm">${BB_LABEL[bb]}</span>
          <span class="tag">cell ${cell.cell_index} · seed ${cell.gen_seed}</span></header>`;
      const svg = svgEl("svg"); svg.setAttribute("class", "compare-svg");
      col.appendChild(svg);
      const readout = el("div", "readout four");
      readout.innerHTML = `
        <div><div class="rv">${fmt(cell.n_tokens)}</div><div class="rk">tokens</div></div>
        <div><div class="rv">${nbuild(cell)}</div><div class="rk">buildings</div></div>
        <div><div class="rv">${cell.counts.road}</div><div class="rk">roads</div></div>
        <div><div class="rv ${term ? "ok" : ""}">${term ? "✓" : "cap"}</div><div class="rk">self-term</div></div>`;
      col.appendChild(readout);
      grid.appendChild(col);
      applyVB(svg, shared);
      renderCell(svg, cell, { animate: true });
      wireHover(svg);
    });
  }

  function buildComparePicker() {
    const seg = $("#cmp-ctx-seg"); seg.innerHTML = "";
    META.contexts.forEach((ctx) => {
      const b = el("button"); b.textContent = META.context_labels[ctx] || ctx;
      if (ctx === state.ctx) b.classList.add("on");
      b.onclick = () => { state.ctx = ctx; buildComparePicker(); renderCompare(); };
      seg.appendChild(b);
    });
    const wrap = $("#compare-cellpick"); wrap.innerHTML = "";
    const lbl = el("span", "lbl"); lbl.textContent = "cell"; wrap.appendChild(lbl);
    const n = Math.max(...META.contexts.map((c) => cellsOf(c, "transformer").length));
    for (let i = 0; i < n; i++) {
      const b = el("button"); b.textContent = i; if (i === state.compareIndex) b.classList.add("on");
      b.onclick = () => { state.compareIndex = i; buildComparePicker(); renderCompare(); };
      wrap.appendChild(b);
    }
  }

  function renderSummary() {
    const tb = $("#summary-table tbody"); tb.innerHTML = "";
    const byCtx = (bb) => Object.fromEntries(DATA[bb].summary.map((s) => [s.context, s]));
    const T = byCtx("transformer"), M = DATA.mamba ? byCtx("mamba") : null;
    META.contexts.forEach((ctx) => {
      const t = T[ctx], m = M ? M[ctx] : null;
      const cell = (tv, mv) => `<td>${tv}</td><td class="m">${m ? mv : "—"}</td>`;
      const tr = el("tr");
      tr.innerHTML = `<td class="ctx">${META.context_labels[ctx] || ctx}</td>` +
        cell(t.med_tokens, m && m.med_tokens) +
        cell(t.med_buildings, m && m.med_buildings) +
        cell(t.med_roads, m && m.med_roads);
      tb.appendChild(tr);
    });
  }

  // ── mode switching ────────────────────────────────────────────────────────
  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("is-active", b.dataset.mode === mode));
    $("#view-inspect").classList.toggle("is-active", mode === "inspect");
    $("#view-compare").classList.toggle("is-active", mode === "compare");
    if (mode === "inspect") { buildInspectPickers(); renderInspect(true); }
    else { buildComparePicker(); renderCompare(); renderSummary(); }
  }
  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));

  // ── provenance footer ─────────────────────────────────────────────────────
  const bbMeta = (k) => { const m = DATA[k].meta; return `${BB_LABEL[k]} ${m.backbone} d${m.d_model}/${m.n_layers}L step ${fmt(m.global_step)}`; };
  $("#provenance").innerHTML = `
    <b>Source</b> reports/_eyeball_probe{,_mamba}/ · 21 pre-generated cells per backbone (PI-approved eyeball probe, NON-scored).
    <b>Models</b> ${BACKBONES.map(bbMeta).join(" · ")} — both ~53M, train=${META.meta.train_set}, conditioning=${META.meta.conditioning_scheme} (${META.meta.conditioning_ablation}), same krakow-seed7 checkpoints, identical conditioning + gen seeds.
    <b>Honesty</b> coordinates are local cell units, not geographic; building footprints that don't seal to exact float-equality are shown as <em>near-closed</em> (a known over-strict closure check, deferred). Character-stats moments are fed to the model — the conditioning response is partly echo. This validates the methodology and is an eyeball backbone comparison; it is NOT a verdict, crown, or realism claim.`;

  // ── boot ──────────────────────────────────────────────────────────────────
  buildInspectPickers();
  renderInspect(true);
})();
