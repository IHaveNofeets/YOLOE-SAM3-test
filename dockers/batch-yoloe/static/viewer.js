"use strict";

// Result viewer: shows a test's results.json (from output/YOLOE/<test>/) with the detections, and
// optionally the ground truth, drawn on the dataset's images (served by /api/image).
//
// Everything is scored per "unit" = one class of one model. A single-class model is one unit
// (shown just by the model name); a multi-class model gets one unit per class ("model / class"),
// grouped under the model in the filter, the summary and the per-image bars.
const Viewer = (() => {
  const PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000", "#000075"];
  const CARD_BARS = 6; // bars shown per grid card; the large view shows all
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  let items = [];      // [{key, url, entry}]
  let groups = [];     // [{model, multi, units: [unit]}]
  let unitById = {};   // id -> {id, model, cls, name, label, color, multi}
  let soloStats = {};   // model -> {n, median_ms, ...}: the model alone, from the results' timing sample
  let allAccurate = false; // every results file was timed one model at a time
  let timingNote = "";  // how the model times in the loaded results were measured
  let timingWarn = false; // true when they are not comparable between models
  let shown = [];      // filtered + sorted items currently in the grid
  let current = -1;    // index into shown for the large view
  const imgCache = new Map();
  let observer = null;

  const unitId = (model, cls) => `${model}#${cls}`;
  let agnostic = new Set(); // models scored class-agnostically: all their detections are one unit

  function buildUnits(sources) {
    const classes = new Map(); // model -> Map(cls -> name)
    const add = (model, cls, name) => {
      if (!classes.has(model)) classes.set(model, new Map());
      const m = classes.get(model);
      if (!m.has(cls)) m.set(cls, name ?? `object${cls}`);
    };
    agnostic = new Set(sources.flatMap((s) => (s.result.models || []).filter((m) => m.class_agnostic).map((m) => m.name)));
    for (const s of sources) {
      for (const m of s.result.models || []) {
        // class-agnostic (prompt-free, older models, external detectors): one entry, "any object"
        if (agnostic.has(m.name)) { add(m.name, 0, "any object"); continue; }
        // no class_names = a huge vocabulary (prompt-free): only classes actually detected get listed
        if (m.class_names) m.class_names.forEach((n, i) => add(m.name, i, n));
        else if (!m.num_classes) add(m.name, 0, "object0");
      }
      for (const entry of s.result.images || []) {
        for (const [model, dets] of Object.entries(entry.results || {})) {
          if (agnostic.has(model)) continue;
          if (!classes.has(model)) add(model, 0, "object0");
          for (const d of dets) add(model, d.cls, d.name);
        }
      }
    }
    groups = [];
    unitById = {};
    let c = 0;
    for (const model of [...classes.keys()].sort()) {
      const clsMap = classes.get(model);
      // show classes when there are several, or when the one class has a real name (a text prompt,
      // a prompt-free class) rather than the visual-prompt default "object0"
      const multi = clsMap.size > 1 || [...clsMap.values()].some((n) => !/^object\d+$/.test(n));
      const units = [...clsMap.entries()].sort((a, b) => a[0] - b[0]).map(([cls, name]) => {
        const u = { id: unitId(model, cls), model, cls, name, multi, color: PALETTE[c++ % PALETTE.length],
          label: multi ? `${model} / ${name}` : model };
        unitById[u.id] = u;
        return u;
      });
      groups.push({ model, multi, units });
    }
  }

  // result: a test's results.json; conf: the confidence to start at (the page's slider)
  function show(result, label, conf) {
    const sources = [{ result }];
    items = (result.images || []).map((entry) => ({
      key: entry.path.replace(/^testImages\//, "").replace(/\/org\.png$/, ""),
      url: "/api/image?path=" + encodeURIComponent(entry.path), entry }));
    buildUnits(sources);
    const modes = new Set(sources.map((s) => s.result.timing?.mode || "unknown"));
    const t0 = sources.map((s) => s.result.timing).find(Boolean);
    soloStats = Object.assign({}, ...sources.map((s) => s.result.timing?.solo || {}));
    allAccurate = modes.size === 1 && modes.has("accurate");
    const nSample = Math.max(0, ...Object.values(soloStats).map((s) => s.n));
    timingWarn = modes.has("unknown") || (modes.has("parallel") && !Object.keys(soloStats).length);
    timingNote = "<b>Alone ms</b>: " + (Object.keys(soloStats).length
        ? `each model running by itself, one image at a time, median over ${nSample} sample image(s) timed before the run`
        : allAccurate ? "each model ran by itself, one image at a time, median over the shown images"
        : "not measured in these results (they were run in parallel without a timing sample)")
      + ". <b>In run ms</b>: median of the times measured during the run"
      + (modes.has("parallel") || modes.has("hybrid") ? ", which include the slowdown from the other models running at the same time" : "")
      + (t0 ? `. ${esc(t0.device)}${t0.cpu_threads ? `, ${t0.cpu_threads} threads` : ""}.` : ".")
      + " A model's time covers its own preprocessing, inference and postprocessing, not the upload or the image loading.";
    imgCache.clear();

    $("vConf").value = (conf ?? 0.25).toFixed(2);
    renderFilter();
    const nClasses = Object.keys(unitById).length;
    $("viewerInfo").textContent = `${label}: ${items.length} test image(s), ${groups.length} model(s), ${nClasses} class(es)`;
    $("viewerBody").hidden = false;
    render();
  }

  // ---------- model / class filter ----------
  const swatch = (u) => `<span class="swatch" style="background:${u.color}"></span>`;

  function renderFilter() {
    $("vModels").innerHTML = groups.map((g) => g.multi
      ? `<span class="vgroup"><label><input type="checkbox" data-model="${esc(g.model)}" checked><b>${esc(g.model)}</b></label>
          <span class="vclasses">${g.units.map((u) =>
            `<label><input type="checkbox" data-unit="${esc(u.id)}" checked>${swatch(u)}${esc(u.name)}</label>`).join("")}</span></span>`
      : `<span class="vgroup"><label><input type="checkbox" data-unit="${esc(g.units[0].id)}" checked>${swatch(g.units[0])}${esc(g.model)}</label></span>`
    ).join("");
  }

  function onFilterChange(e) {
    const box = e.target;
    const group = box.closest(".vgroup");
    if (box.dataset.model !== undefined) {
      // model checkbox of a multi-class model: toggle all its classes
      group.querySelectorAll("input[data-unit]").forEach((i) => { i.checked = box.checked; });
    } else if (group) {
      const modelBox = group.querySelector("input[data-model]");
      if (modelBox) {
        const all = [...group.querySelectorAll("input[data-unit]")];
        const on = all.filter((i) => i.checked).length;
        modelBox.checked = on > 0;
        modelBox.indeterminate = on > 0 && on < all.length;
      }
    }
    render();
  }

  function visibleUnits() {
    return new Set([...$("vModels").querySelectorAll("input[data-unit]:checked")].map((i) => i.dataset.unit));
  }

  // ---------- scoring ----------
  function detsFor(item, vis, minConf) {
    const out = [];
    for (const [model, dets] of Object.entries(item.entry.results || {})) {
      for (const d of dets) {
        const u = unitById[unitId(model, agnostic.has(model) ? 0 : d.cls)];
        if (u && vis.has(u.id) && d.conf >= minConf) out.push({ ...d, model, unit: u });
      }
    }
    return out.sort((a, b) => b.conf - a.conf);
  }

  // per visible unit: best confidence (0 = nothing above minConf) and number of detections
  function scores(item, vis, minConf) {
    const s = new Map();
    for (const g of groups) for (const u of g.units) if (vis.has(u.id)) s.set(u.id, { unit: u, best: 0, count: 0 });
    for (const d of detsFor(item, vis, minConf)) {
      const e = s.get(d.unit.id);
      e.best = Math.max(e.best, d.conf);
      e.count++;
    }
    return [...s.values()].sort((a, b) => b.best - a.best || a.unit.label.localeCompare(b.unit.label));
  }

  const bestOf = (it) => (it.scores[0] && it.scores[0].best > 0 ? it.scores[0] : null);
  const totalOf = (it) => it.scores.reduce((n, s) => n + s.count, 0);

  function render() {
    const vis = visibleUnits();
    const minConf = parseFloat($("vConf").value) || 0;
    const text = $("vFilter").value.trim().toLowerCase();
    const onlyHits = $("vOnlyHits").checked;
    const sort = $("vSort").value;
    const byPath = (a, b) => a.key.localeCompare(b.key, undefined, { numeric: true });

    shown = items
      .map((it) => ({ ...it, scores: scores(it, vis, minConf) }))
      .filter((it) => !text || it.key.toLowerCase().includes(text))
      .filter((it) => !onlyHits || bestOf(it));
    if (sort === "conf") shown.sort((a, b) => (bestOf(b)?.best || 0) - (bestOf(a)?.best || 0) || byPath(a, b));
    else if (sort === "model") shown.sort((a, b) => (bestOf(a)?.unit.label ?? "~").localeCompare(bestOf(b)?.unit.label ?? "~") || byPath(a, b));
    else if (sort === "count") shown.sort((a, b) => totalOf(b) - totalOf(a) || byPath(a, b));
    else shown.sort(byPath);

    renderSummary(vis, minConf);
    renderGrid();
  }

  function renderSummary(vis, minConf) {
    const n = shown.length || 1;
    const pct = (v) => `${v} <span class="muted">(${Math.round(100 * v / n)}%)</span>`;
    const stats = (match) => {
      // match(scoreEntry) selects which units of an image count toward this row
      let hits = 0, best = 0, sum = 0, dets = 0, maxPer = 0;
      for (const it of shown) {
        const mine = it.scores.filter(match);
        const top = Math.max(0, ...mine.map((s) => s.best));
        const count = mine.reduce((k, s) => k + s.count, 0);
        if (top > 0) { hits++; sum += top; }
        const b = bestOf(it);
        if (b && match(b)) best++;
        dets += count;
        maxPer = Math.max(maxPer, count);
      }
      return `<td>${pct(hits)}</td><td>${pct(best)}</td><td>${hits ? (sum / hits).toFixed(2) : "–"}</td>`
        + `<td>${dets}</td><td>${dets ? maxPer : "–"}</td>`;
    };

    // time per image for a whole model (median and mean over the shown images that have a time)
    const timeCells = (model) => {
      const ms = shown.map((it) => it.entry.model_ms?.[model]).filter((v) => typeof v === "number").sort((a, b) => a - b);
      const med = ms.length ? (ms.length % 2 ? ms[(ms.length - 1) / 2] : (ms[ms.length / 2 - 1] + ms[ms.length / 2]) / 2) : null;
      const alone = soloStats[model] ? soloStats[model].median_ms : allAccurate ? med : null;
      return `<td>${alone == null ? "–" : `<b>${alone.toFixed(0)}</b>`}</td><td>${med == null ? "–" : med.toFixed(0)}</td>`;
    };

    const rows = [];
    for (const g of groups) {
      const units = g.units.filter((u) => vis.has(u.id));
      if (!units.length) continue;
      if (g.multi) {
        rows.push(`<tr class="vgroup-row"><td><b>${esc(g.model)}</b> <span class="muted">any class</span></td>`
          + stats((s) => s.unit.model === g.model) + timeCells(g.model) + "</tr>");
        for (const u of units) {
          rows.push(`<tr><td class="vindent">${swatch(u)}${esc(u.name)}</td>` + stats((s) => s.unit.id === u.id) + "<td></td><td></td></tr>");
        }
      } else {
        rows.push(`<tr><td>${swatch(units[0])}${esc(g.model)}</td>` + stats((s) => s.unit.id === units[0].id) + timeCells(g.model) + "</tr>");
      }
    }
    const none = shown.filter((it) => !bestOf(it)).length;
    $("vSummary").innerHTML = `<table>
      <thead><tr><th>Model / class</th><th>Images with a detection</th><th>Best match for</th><th>Mean best conf</th>
        <th>Detections</th><th>Max per image</th><th title="The model running by itself (median)">Alone ms</th><th title="Median of the times measured during the run">In run ms</th></tr></thead>
      <tbody>${rows.join("")}
      <tr><td class="muted">no match</td><td></td><td>${pct(none)}</td><td></td><td></td><td></td><td></td><td></td></tr>
      </tbody></table>
      <div class="muted">${shown.length} of ${items.length} image(s) shown, conf ≥ ${minConf}</div>
      <div class="${timingWarn ? "warn" : "muted"} small">${timingNote}</div>`;
  }

  // ---------- drawing ----------
  function loadImage(item) {
    if (!imgCache.has(item.key)) {
      imgCache.set(item.key, new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = item.url;
      }));
    }
    return imgCache.get(item.key);
  }

  function draw(canvas, img, item, maxW) {
    const vis = visibleUnits();
    const minConf = parseFloat($("vConf").value) || 0;
    const w = img ? img.naturalWidth : item.entry.width || 300;
    const h = img ? img.naturalHeight : item.entry.height || 200;
    const scale = maxW ? Math.min(1, maxW / w) : 1;
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    const ctx = canvas.getContext("2d");
    if (img) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    else { ctx.fillStyle = "#888"; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    const lw = Math.max(2, Math.round(canvas.width / 250));
    const fs = Math.max(maxW ? 16 : 13, Math.round(canvas.width / (maxW ? 18 : 35)));
    ctx.lineWidth = lw;
    ctx.font = `${fs}px sans-serif`;
    const dets = detsFor(item, vis, minConf); // best first
    // place labels best-first at the box's top-left corner, pushing each one down until it no
    // longer overlaps a label already placed (several full-frame matches, or nearby boxes)
    const lh = fs + 4;
    const placed = [];
    const overlaps = (a) => placed.some((b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + lh && b.y < a.y + lh);
    const labels = dets.map((d) => {
      const [x1, y1, x2, y2] = d.bbox.map((v) => v * scale);
      const cname = agnostic.has(d.model) ? d.name : d.unit.name; // class-agnostic: show the model's own word
      const text = `${d.unit.multi ? `${d.model}/${cname}` : d.model} ${d.conf.toFixed(2)}`;
      const w = ctx.measureText(text).width + 6;
      const r = { x: Math.max(0, Math.min(x1, canvas.width - w)), y: Math.max(0, y1), w };
      while (overlaps(r) && r.y + lh < canvas.height) r.y += lh;
      placed.push(r);
      return { d, x1, y1, x2, y2, text, r };
    });
    const tpfp = $("vTpFp").checked;
    const colorOf = (d) => (tpfp && d.tp !== undefined ? (d.tp ? "#2ecc40" : "#ff4136") : d.unit.color);
    if ($("vTruth").checked) drawTruth(ctx, item, scale, lw, fs);
    for (const { d, x1, y1, x2, y2 } of [...labels].reverse()) { // best box drawn last, on top
      ctx.strokeStyle = colorOf(d);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    for (const { d, text, r } of labels) {
      ctx.fillStyle = colorOf(d);
      ctx.fillRect(r.x, r.y, r.w, lh);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, r.x + 3, r.y + fs);
    }
  }

  // ground truth: dashed white boxes (dark outline so they show on light images); U = uncompleted,
  // C = covered, the objects that are left out of training and baking unless ticked
  function drawTruth(ctx, item, scale, lw, fs) {
    ctx.save();
    ctx.setLineDash([lw * 3, lw * 2]);
    ctx.font = `${Math.round(fs * 0.85)}px sans-serif`;
    for (const t of item.entry.truth || []) {
      const [x1, y1, x2, y2] = t.bbox.map((v) => v * scale);
      ctx.lineWidth = lw + 2; ctx.strokeStyle = "rgba(0,0,0,.7)"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.lineWidth = lw; ctx.strokeStyle = "#fff"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const tag = (t.uncompleted ? "U" : "") + (t.covered ? "C" : "");
      if (tag) {
        ctx.fillStyle = "rgba(0,0,0,.7)"; ctx.fillRect(x2 - fs * 1.4, y2 - fs - 2, fs * 1.4, fs + 2);
        ctx.fillStyle = "#fff"; ctx.fillText(tag, x2 - fs * 1.3, y2 - 4);
      }
    }
    ctx.restore();
  }

  function bar(s, isBest) {
    return `<div class="bar${isBest ? " best" : ""}">
      <span class="bname" title="${esc(s.unit.label)}">${esc(s.unit.label)}</span>
      <span class="btrack"><span style="width:${Math.round(s.best * 100)}%;background:${s.unit.color}"></span></span>
      <span class="bval">${s.best > 0 ? s.best.toFixed(2) : "–"}</span>
      <span class="bcount">${s.count > 1 ? "×" + s.count : ""}</span></div>`;
  }

  // limit: max bars (cards); the rest are summarised in one line
  function scoreBars(it, limit = Infinity) {
    const best = bestOf(it);
    const head = it.scores.slice(0, limit).map((s) => bar(s, s === best)).join("");
    const rest = it.scores.slice(limit);
    if (!rest.length) return head;
    const hits = rest.filter((s) => s.best > 0).length;
    return head + `<div class="muted bmore">+${rest.length} more${hits ? ` (${hits} with detections)` : ""}</div>`;
  }

  // large view: bars grouped by model, classes of a multi-class model under their model
  function groupedBars(it) {
    const best = bestOf(it);
    return groups.map((g) => {
      const mine = it.scores.filter((s) => s.unit.model === g.model).sort((a, b) => a.unit.cls - b.unit.cls);
      if (!mine.length) return "";
      if (!g.multi) return bar(mine[0], mine[0] === best);
      return `<div class="bgroup"><b>${esc(g.model)}</b>${mine.map((s) =>
        bar({ ...s, unit: { ...s.unit, label: s.unit.name } }, s === best)).join("")}</div>`;
    }).join("");
  }

  function renderGrid() {
    const grid = $("vGrid");
    grid.innerHTML = "";
    if (observer) observer.disconnect();
    observer = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        observer.unobserve(e.target);
        const it = shown[+e.target.dataset.i];
        loadImage(it).then((img) => draw(e.target, img, it, 320));
      }
    }, { rootMargin: "300px" });

    shown.forEach((it, i) => {
      const el = document.createElement("div");
      el.className = "vcard";
      const err = it.entry.error ? `<div class="err">${esc(it.entry.error)}</div>` : "";
      const total = totalOf(it);
      el.innerHTML = `<canvas data-i="${i}" width="320" height="200"></canvas>
        <div class="vpath" title="${esc(it.key)}">${esc(it.key)}</div>
        <div class="muted vcount">${total} detection(s)</div>${err}<div class="bars">${scoreBars(it, CARD_BARS)}</div>`;
      el.onclick = () => openLarge(i);
      grid.appendChild(el);
      observer.observe(el.querySelector("canvas"));
    });
  }

  // ---------- large view ----------
  async function openLarge(i) {
    if (i < 0 || i >= shown.length) return;
    current = i;
    const it = shown[i];
    $("vModal").hidden = false;
    $("vModalTitle").textContent = `${it.key}  (${i + 1}/${shown.length})`;
    const vis = visibleUnits();
    const minConf = parseFloat($("vConf").value) || 0;
    const dets = detsFor(it, vis, minConf)
      .sort((a, b) => a.model.localeCompare(b.model) || a.cls - b.cls || b.conf - a.conf);
    const times = Object.entries(it.entry.model_ms || {}).sort((a, b) => a[1] - b[1]);
    const aloneMs = it.entry.model_ms_solo; // present on the timing-sample images
    const split = it.entry.model_split_solo_ms || it.entry.model_split_ms || {};
    const timeTable = times.length ? `<table><thead><tr><th>Model</th><th>In run ms</th>${aloneMs ? "<th>Alone ms</th>" : ""}<th>Preprocess</th><th>Inference</th><th>Postprocess</th></tr></thead><tbody>
      ${times.map(([m, ms]) => `<tr><td>${esc(m)}</td><td>${ms.toFixed(1)}</td>${aloneMs ? `<td>${aloneMs[m]?.toFixed(1) ?? "–"}</td>` : ""}<td>${split[m]?.preprocess ?? "–"}</td><td>${split[m]?.inference ?? "–"}</td><td>${split[m]?.postprocess ?? "–"}</td></tr>`).join("")}
      </tbody></table>${aloneMs ? `<div class="muted small">The preprocess / inference / postprocess split is from the run alone.</div>` : ""}` : "";
    const truth = it.entry.truth || [];
    const truthLine = `<div class="small muted">Ground truth: ${truth.length} object(s)`
      + `${truth.filter((t) => t.uncompleted).length ? `, ${truth.filter((t) => t.uncompleted).length} uncompleted` : ""}`
      + `${truth.filter((t) => t.covered).length ? `, ${truth.filter((t) => t.covered).length} covered` : ""}`
      + (truth.length ? ": " + [...new Set(truth.map((t) => t.name))].map(esc).join(", ") : "") + "</div>";
    $("vModalDets").innerHTML = `${truthLine}<div class="bars">${groupedBars(it)}</div>${timeTable}
      <table><thead><tr><th>Model</th><th>Class</th><th>Conf</th><th>Match</th><th>Box (x1, y1, x2, y2)</th></tr></thead><tbody>
      ${dets.map((d) => `<tr><td>${swatch(d.unit)}${esc(d.model)}</td>
        <td>${esc(d.name)}${d.prompt && d.prompt !== d.name ? ` <span class="muted">(${esc(d.prompt)})</span>` : ""}</td><td>${d.conf.toFixed(3)}</td>
        <td>${d.tp === undefined ? "–" : d.tp ? '<span class="ok">TP</span>' : '<span class="err">FP</span>'}</td><td>${d.bbox.map((v) => Math.round(v)).join(", ")}</td></tr>`).join("")
        || `<tr><td colspan="5" class="muted">No detections at conf ≥ ${minConf}</td></tr>`}
      </tbody></table>${it.entry.error ? `<div class="err">${esc(it.entry.error)}</div>` : ""}`;
    const img = await loadImage(it);
    if (current === i) draw($("vModalCanvas"), img, it, 0);
  }

  function closeLarge() {
    $("vModal").hidden = true;
    current = -1;
  }

  // ---------- wiring ----------
  for (const id of ["vConf", "vFilter", "vOnlyHits", "vSort", "vTruth", "vTpFp"]) $(id).addEventListener("input", render);
  $("vModels").addEventListener("change", onFilterChange);
  $("vClose").onclick = closeLarge;
  $("vPrev").onclick = () => openLarge(current - 1);
  $("vNext").onclick = () => openLarge(current + 1);
  $("vModal").addEventListener("click", (e) => { if (e.target.id === "vModal") closeLarge(); });
  document.addEventListener("keydown", (e) => {
    if (current < 0) return;
    if (e.key === "Escape") closeLarge();
    else if (e.key === "ArrowLeft") openLarge(current - 1);
    else if (e.key === "ArrowRight") openLarge(current + 1);
  });
  // the page's confidence slider drives the viewer too
  function setConf(t) {
    $("vConf").value = Number(t).toFixed(2);
    if (items.length) render();
  }

  return { show, setConf };
})();
