"use strict";

// Dataset workflow for SAM 3: build models from the mounted dataset (refImages / trainImages), test them on
// testImages, and look at the scores with a confidence chosen afterwards. viewer.js draws the images.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const KIND_LABELS = { text: "Text prompt", exemplar: "Exemplar prompt", visual: "Visual prompt", prompt_tune: "Prompt-tuned" };
const f3 = (v) => (v == null || Number.isNaN(v) ? "–" : v.toFixed(3));
const pct = (v) => (v == null ? "–" : (100 * v).toFixed(1) + "%");

let DS = null;        // /api/dataset
let models = [];      // /api/models
let test = null;      // the opened test: {id, metrics, matches, results, prep}

async function api(url, opts) {
  const res = await fetch(url, opts);
  let body;
  try { body = await res.json(); } catch { body = { message: res.statusText }; }
  if (!res.ok) throw new Error(body.message || `HTTP ${res.status}`);
  return body;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// ============================================================ dataset
const classNames = () => Object.fromEntries(Object.entries(DS.classes).map(([k, v]) => [+k, v]));
const isOther = (id) => DS.other_ids.includes(+id);
const shortDs = (n) => n.replace(/^dataset(\d+)_.*/, "d$1");
// a text model's prompts as one list per class (older models store a flat list, one prompt per class)
const promptGroups = (m) => (Array.isArray(m.prompts) ? m.prompts.map((g) => (Array.isArray(g) ? g : [g])) : []);

async function loadDataset() {
  try {
    DS = await api("/api/dataset");
  } catch (e) {
    $("dsInfo").innerHTML = `<span class="err">${esc(e.message)}</span>`;
    return;
  }
  const names = classNames();
  const ids = Object.keys(names).map(Number).sort((a, b) => a - b);
  $("dsInfo").innerHTML = `Root <code>${esc(DS.root)}</code> · ${DS.datasets.length} dataset(s). Cells: objects in train / test, `
    + `with <b>U</b> uncompleted and <b>C</b> covered in train. "Other" is never used.`;
  const cell = (sp, id) => {
    const tr = sp.train.classes[id], te = sp.test.classes[id];
    if (!tr && !te) return `<td class="muted">–</td>`;
    return `<td>${tr?.objects ?? 0} / ${te?.objects ?? 0}${tr ? `<div class="muted small">U ${tr.uncompleted} · C ${tr.covered}</div>` : ""}</td>`;
  };
  $("dsTable").innerHTML = `<table class="dtable"><thead><tr><th>Dataset</th><th>Images train / test / ref</th>
      ${ids.map((i) => `<th class="${isOther(i) ? "muted" : ""}">${esc(names[i])}</th>`).join("")}</tr></thead><tbody>
    ${DS.datasets.map((d) => `<tr><td>${esc(d.name)}</td><td>${d.splits.train.images} / ${d.splits.test.images} / ${d.splits.ref.images}</td>
      ${ids.map((i) => cell(d.splits, String(i))).join("")}</tr>`).join("")}</tbody></table>`;

  $("bDataset").innerHTML = DS.datasets.map((d) => `<option value="${esc(d.name)}">${esc(d.name)}</option>`).join("");
  onDatasetChange();
}

function onDatasetChange() {
  if (!DS) return;
  const d = DS.datasets.find((x) => x.name === $("bDataset").value);
  const names = classNames();
  const count = (i) => (d.splits.train.classes[i]?.objects || 0) + (d.splits.ref.classes[i]?.objects || 0);
  const ids = Object.keys(names).map(Number).filter((i) => !isOther(i)).sort((a, b) => count(b) - count(a));
  $("bClass").innerHTML = ids.map((i) => `<option value="${i}">${esc(names[i])} (${d.splits.train.classes[i]?.objects || 0} train, `
    + `${d.splits.ref.classes[i]?.objects || 0} ref)</option>`).join("");
  onFormChange();
}

// ============================================================ build form
function scopeFromForm() {
  return document.querySelector('input[name="bScope"]:checked').value === "all"
    ? { type: "all" } : { type: "single", dataset: $("bDataset").value, class_id: +$("bClass").value };
}

function scopeClassIds(scope) {
  if (scope.type === "single") return [scope.class_id];
  return Object.keys(DS.classes).map(Number).filter((i) => !isOther(i)).sort((a, b) => a - b);
}

function suggestName() {
  const s = scopeFromForm(), k = $("bKind").value;
  const kind = { text: "text", exemplar: "ex", visual: "vis", prompt_tune: "pt" }[k];
  const part = s.type === "all" ? "all" : `${shortDs(s.dataset)}_${classNames()[s.class_id].replace(/[^A-Za-z0-9]+/g, "")}`;
  const flags = ($("bIncU").checked ? "U" : "") + ($("bIncC").checked ? "C" : "");
  let name = `${kind}_${part}${flags ? "_" + flags : ""}`;
  const taken = new Set([...models.map((m) => m.name), ...queuedTargets]);
  if (taken.has(name)) { let i = 2; while (taken.has(`${name}_${i}`)) i++; name = `${name}_${i}`; }
  return name;
}

let nameTouched = false;
function onFormChange() {
  if (!DS) return;
  const kind = $("bKind").value;
  $("bDataset").disabled = $("bClass").disabled = scopeFromForm().type === "all";
  const boxes = kind === "exemplar" || kind === "visual";
  $("optText").hidden = kind !== "text";
  $("optExemplar").hidden = kind !== "exemplar";
  $("optVisual").hidden = kind !== "visual";
  $("optBoxText").hidden = !boxes;
  $("optTune").hidden = kind !== "prompt_tune";
  $("bFlagsRow").hidden = kind === "text"; // no dataset objects are used to build a text model
  if (!nameTouched) $("bName").value = suggestName();
  const scope = scopeFromForm();
  if (kind === "text") renderPromptEditor(scope);
  if (kind === "prompt_tune") renderStartEditor(scope);
  if (boxes || kind === "prompt_tune") renderBuildInfo(scope, kind);
}

// prompt tuning: the frozen text each class's learned tokens follow (one line per class)
function renderStartEditor(scope) {
  const names = classNames();
  const ids = scopeClassIds(scope);
  const key = JSON.stringify(ids);
  const box = $("bStart");
  if (box.dataset.key === key) return; // keep what the user typed
  box.dataset.key = key;
  box.innerHTML = ids.map((i) => `<div class="prow"><b>${esc(names[i])}</b>
    <input type="text" data-cls="${i}" size="28" value="${esc(names[i])}"></div>`).join("");
}

function renderBuildInfo(scope, kind) {
  const ids = new Set(scopeClassIds(scope));
  const dss = scope.type === "all" ? DS.datasets : DS.datasets.filter((d) => d.name === scope.dataset);
  const source = kind === "prompt_tune" ? $("bSource").value : "ref";
  const fromRef = source === "ref";
  $("bCountRow").hidden = source !== "train";
  $("bSeedRow").hidden = fromRef;
  $("bSeedHint").textContent = source === "train" ? "the images are picked at random with the seed; the same number and seed always give the same images"
    : source === "all" ? "the seed picks which 10% are held out for validation" : "";
  const split = fromRef ? "ref" : "train";
  let objs = 0, u = 0, c = 0, imgs = 0;
  for (const d of dss) {
    imgs += d.splits[split].images;
    for (const [i, v] of Object.entries(d.splits[split].classes)) if (ids.has(+i)) { objs += v.objects; u += v.uncompleted; c += v.covered; }
  }
  const used = objs - (!$("bIncU").checked ? u : 0) - (!$("bIncC").checked ? c : 0); // approximate: an object can be both
  const txt = `${imgs} ${split === "ref" ? "reference" : "training"} image(s), ${objs} in-scope object(s), of which ${u} uncompleted and ${c} covered `
    + `→ about ${Math.max(0, used)} used.`;
  if (kind !== "prompt_tune") { $("bRefInfo").textContent = txt; return; }
  if (fromRef) {
    $("bTrainInfo").textContent = txt + " With so few images nothing is held out: validation uses the same reference images. "
      + "Test images are never used. Left-out flagged objects are greyed out.";
    return;
  }
  const n = source === "train" ? +$("bMaxImages").value || 0 : 0;
  const pick = n > 0 && n < imgs ? ` Using ${n} of them (about ${Math.round(used * n / imgs)} objects), picked with seed ${+$("bSeed").value || 0}.` : "";
  $("bTrainInfo").textContent = txt + pick + " 10% of the training images used are held out for validation; test images are never used. "
    + "Left-out flagged objects are greyed out in the training images.";
}

function renderPromptEditor(scope) {
  const names = classNames();
  const ids = scopeClassIds(scope);
  const key = JSON.stringify(ids);
  const box = $("bPrompts");
  if (box.dataset.key === key) return; // keep what the user typed
  box.dataset.key = key;
  box.innerHTML = ids.map((i) => `<div class="prow"><b>${esc(names[i])}</b>
    <textarea data-cls="${i}" rows="3">${esc(names[i])}</textarea></div>`).join("");
  const textModels = models.filter((m) => m.type === "text" && JSON.stringify(m.class_ids) === key);
  $("bCopyFrom").innerHTML = `<option value="">–</option>` + textModels.map((m) => `<option>${esc(m.name)}</option>`).join("");
}

function copyPrompts() {
  const m = models.find((x) => x.name === $("bCopyFrom").value);
  const groups = m ? promptGroups(m) : [];
  [...$("bPrompts").querySelectorAll("textarea")].forEach((ta, i) => { if (groups[i]) ta.value = groups[i].join("\n"); });
}

async function build() {
  const kind = $("bKind").value;
  const body = {
    name: $("bName").value.trim(), kind, scope: scopeFromForm(),
    include_uncompleted: $("bIncU").checked, include_covered: $("bIncC").checked, options: {},
  };
  if (kind === "text") body.options.prompts = [...$("bPrompts").querySelectorAll("textarea")]
    .map((ta) => ta.value.split("\n").map((s) => s.trim()).filter(Boolean));
  if (kind === "exemplar" || kind === "visual") body.options.with_text = $("bWithText").checked;
  if (kind === "exemplar") Object.assign(body.options, { max_exemplars: +$("bMaxEx").value || 4, negatives: $("bNeg").checked });
  if (kind === "prompt_tune") Object.assign(body.options, { epochs: +$("bEpochs").value, tokens: +$("bTokens").value,
    lr: +$("bLr").value, source: $("bSource").value, seed: +$("bSeed").value || 0,
    max_images: $("bSource").value === "train" ? +$("bMaxImages").value || 0 : 0,
    start_text: [...$("bStart").querySelectorAll("input")].map((i) => i.value.trim()) });
  $("bMsg").textContent = "";
  try {
    await post("/api/models", body);
    $("bMsg").innerHTML = `<span class="ok">added "${esc(body.name)}" to the queue</span>`;
    nameTouched = false;
    await pollJob();
    $("bName").value = suggestName(); // the next build gets a fresh name
  } catch (e) {
    $("bMsg").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  }
}

// ============================================================ jobs (one queue, run one after the other)
let pollTimer = null;
let seenFinished = null; // ids of finished jobs already handled (null until the first poll)
let queuedTargets = new Set(); // model names that queued / running builds will create
let currentJobId = null;

async function pollJob() {
  clearTimeout(pollTimer);
  let q = null;
  try { q = await api("/api/jobs"); } catch {}
  if (q) {
    try { await onJobs(q); } catch (e) { console.error(e); } // never stop polling on a failure
  }
  const busy = q && (q.running || q.queued.length);
  pollTimer = setTimeout(pollJob, busy ? 1500 : 6000);
}

async function onJobs(q) {
  const shown = q.running || q.finished[0] || null; // the bar shows the running job, else the last one
  currentJobId = shown?.id ?? null;
  queuedTargets = new Set([q.running, ...q.queued].filter((j) => j && j.kind === "build").map((j) => j.target));
  const pending = [...queuedTargets].filter((n) => !models.some((m) => m.name === n)).join(",");
  if (pending !== ($("tModels").dataset.pending || "")) renderTestModels();
  const busy = !!(q.running || q.queued.length);
  $("bBuild").textContent = busy ? "Add to queue" : "Build";
  $("tRun").textContent = busy ? "Add to queue" : "Run test";

  const bar = $("jobBar");
  const sig = `${q.running?.id}|${q.queued.map((j) => j.id).join(",")}|${q.finished[0]?.id}`;
  if (shown) {
    bar.hidden = bar.dataset.hidden === sig;
    $("jobTitle").textContent = shown.title;
    $("jobProgress").value = shown.progress;
    const eta = shown.eta_s ? ` · about ${fmtSecs(shown.eta_s)} left` : "";
    const cls = shown.status === "error" ? "err" : shown.status === "done" ? "ok" : "";
    $("jobMsg").innerHTML = `<span class="${cls}">${esc(shown.status)}</span> · ${esc(shown.message)} · ${fmtSecs(shown.elapsed_s)}${eta}`
      + (q.queued.length ? ` · <b>${q.queued.length} waiting</b>` : "");
    $("jobCancel").disabled = shown.status !== "running";
    $("jobLog").textContent = shown.log.join("\n");
    if (!$("jobLog").hidden) $("jobLog").scrollTop = $("jobLog").scrollHeight;
    $("jobQueueToggle").textContent = `Queue (${q.queued.length})`;
  } else {
    bar.hidden = true;
  }
  bar.dataset.sig = sig;
  renderQueue(q);

  // jobs that finished since the last poll: refresh what they changed
  const fresh = seenFinished ? q.finished.filter((j) => !seenFinished.has(j.id)) : [];
  seenFinished = new Set(q.finished.map((j) => j.id));
  if (fresh.some((j) => j.kind === "build")) await loadModels();
  const test = fresh.find((j) => j.kind === "test" && j.result?.id); // newest first
  if (fresh.some((j) => j.kind === "test")) await loadTests();
  if (test) await openTest(test.result.id);
}

function renderQueue(q) {
  const row = (j, extra) => `<tr><td class="small">${esc(j.status)}</td><td>${esc(j.title)}</td><td class="small muted">${extra}</td>
    <td>${j.status === "queued" || j.status === "running" ? `<button data-cancel="${esc(j.id)}" title="${j.status === "queued" ? "Remove from the queue" : "Cancel"}">✕</button>` : ""}</td></tr>`;
  const rows = [];
  if (q.running) rows.push(row(q.running, `${Math.round(100 * q.running.progress)}% · ${esc(q.running.message)}`));
  q.queued.forEach((j, i) => rows.push(row(j, `#${i + 1} in the queue`)));
  q.finished.slice(0, 12).forEach((j) => rows.push(row(j, `${fmtSecs(j.elapsed_s)}${j.status === "error" ? " · " + esc(j.message) : ""}`)));
  $("jobQueue").innerHTML = rows.length
    ? `<table class="qtable"><tbody>${rows.join("")}</tbody></table>
       ${q.finished.length ? `<button id="jobClear" class="small">Clear finished</button>` : ""}`
    : `<span class="muted small">Nothing queued</span>`;
}

function fmtSecs(s) {
  s = Math.round(s);
  return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ============================================================ models
// what a prompt tuning was trained on, from its training data stats
function trainDataLabel(td) {
  if (!td?.available_images) return "";
  if (td.source === "ref") return `trained on the ${td.train_images} reference image(s)`;
  const used = td.used_images ?? td.train_images + (td.val_is_train ? 0 : td.val_images);
  return used >= td.available_images ? `all ${used} train images, seed ${td.seed ?? 0}`
    : `${used} of ${td.available_images} train images, seed ${td.seed ?? 0}`;
}

async function loadModels() {
  const r = await api("/api/models");
  models = r.models;
  $("serverInfo").innerHTML = `${models.length} model(s) · ${esc(r.device)}, ${r.cpu_threads} threads`
    + (r.weights ? "" : ` · <span class="err">SAM 3 weights missing: copy sam3.pt to dockers/batch-sam3/modelData/sam3/</span>`);
  $("modelList").innerHTML = models.length ? "" : `<span class="muted">No models yet</span>`;
  for (const m of models) {
    const el = document.createElement("div");
    el.className = "model";
    const flags = m.type !== "text"
      ? `<div class="small">uncompleted: ${m.include_uncompleted ? "used" : "left out"} · covered: ${m.include_covered ? "used" : "left out"}</div>` : "";
    const o = m.options || {};
    const boxInfo = m.type === "exemplar" || m.type === "visual"
      ? `<div class="small muted">${m.type === "exemplar" ? `${o.max_exemplars || 4} exemplar(s) per class${o.negatives ? " + negatives" : ""} · ` : ""}`
        + `text: ${o.with_text ? "class name" : "“visual”"}</div>` : "";
    const tr = m.training;
    const td = tr?.data;
    const trainInfo = tr ? `<div class="small muted">${tr.epochs_done}/${tr.epochs} epochs · ${tr.tokens} token(s) per class`
      + `${tr.best_val_ap50 != null ? ` · best val AP50 ${f3(tr.best_val_ap50)} (epoch ${tr.best_epoch})` : ""}</div>`
      + `<div class="small muted">${trainDataLabel(td)}</div>`
      + `<div class="small muted">start: ${(o.start_text || m.class_names || []).map((s) => `“${esc(s)}”`).join(", ")}</div>` : "";
    const prompts = m.type === "text" && promptGroups(m).length
      ? `<div class="mpreview">${promptGroups(m).map((g, i) => g.map((p) => `<span class="chip" title="${esc(m.class_names?.[i] ?? "")}">${esc(p)}</span>`).join("")).join("")}</div>` : "";
    const preview = m.type === "exemplar" || m.type === "visual" ? `<img src="/api/models/${m.name}/preview?t=${encodeURIComponent(m.created)}" alt="">`
      : prompts || `<div class="mpreview mpf">${esc(KIND_LABELS[m.type] || m.type)}</div>`;
    el.innerHTML = `${preview}
      <b>${esc(m.name)}</b><span class="badge badge-${m.type}">${esc(KIND_LABELS[m.type] || m.type)}</span>
      <div class="small">${esc(m.scope_label || "")}</div>
      <div class="small muted">${m.class_names && m.num_classes <= 8 ? m.class_names.map(esc).join(", ") : `${m.num_classes} class(es)`}</div>
      ${flags}${boxInfo}${trainInfo}
      ${m.build_seconds ? `<div class="small muted">built in ${fmtSecs(m.build_seconds)}</div>` : ""}
      <a class="btn" href="/api/models/${m.name}/download" download>Download</a><button>Delete</button>`;
    const del = el.querySelector("button");
    if (del) del.onclick = async () => {
      if (!confirm(`Delete model "${m.name}"?`)) return;
      await api(`/api/models/${m.name}`, { method: "DELETE" });
      loadModels();
    };
    $("modelList").appendChild(el);
  }
  renderTestModels();
  if (!nameTouched && DS) $("bName").value = suggestName();
  $("bPrompts").dataset.key = ""; // refresh "copy prompts from"
  if (DS && $("bKind").value === "text") renderPromptEditor(scopeFromForm());
}

// the test's model checkboxes: built models, plus models still waiting in the queue to be built
// (a test queued after their builds runs on them once they exist)
function renderTestModels() {
  const prev = new Set([...$("tModels").querySelectorAll("input:checked")].map((i) => i.value));
  const pending = [...queuedTargets].filter((n) => !models.some((m) => m.name === n));
  $("tModels").innerHTML = models.map((m) => `<label title="${esc(m.scope_label || "")}"><input type="checkbox" value="${esc(m.name)}"
    ${prev.has(m.name) ? "checked" : ""}> ${esc(m.name)}</label>`).join("")
    + pending.map((n) => `<label class="muted" title="waiting in the queue to be built"><input type="checkbox" value="${esc(n)}"
    ${prev.has(n) ? "checked" : ""}> ${esc(n)} <i>(queued)</i></label>`).join("");
  $("tModels").dataset.pending = pending.join(",");
}

// ============================================================ tests
async function runTest() {
  const picked = [...$("tModels").querySelectorAll("input:checked")].map((i) => i.value);
  $("tMsg").textContent = "";
  try {
    await post("/api/tests", { name: $("tName").value.trim(), models: picked, max_images: +$("tMax").value || 0 });
    $("tMsg").innerHTML = `<span class="ok">test added to the queue</span>`;
    pollJob();
  } catch (e) {
    $("tMsg").innerHTML = `<span class="err">${esc(e.message)}</span>`;
  }
}

async function loadTests() {
  const r = await api("/api/tests");
  const cur = $("tPick").value;
  $("tPick").innerHTML = r.tests.length
    ? r.tests.map((t) => `<option value="${esc(t.id)}">${esc(t.id)} (${t.models.length} model(s))</option>`).join("")
    : `<option value="">no tests yet</option>`;
  if (cur && r.tests.some((t) => t.id === cur)) $("tPick").value = cur;
  return r.tests;
}

async function openTest(id) {
  if (!id) return;
  $("tPick").value = id;
  $("tInfo").textContent = "Loading…";
  const [metrics, matches, results] = await Promise.all(["metrics", "matches", "results"].map((p) => api(`/api/tests/${id}/${p}`)));
  test = { id, metrics, matches, results, prep: prepMatches(matches), sort: { key: "f1", dir: -1 }, open: new Set() };
  $("tCsv").href = `/api/tests/${id}/metrics.csv`;
  $("tBody").hidden = false;
  const best = metrics.models.map((m) => m.best?.conf).filter((v) => v != null);
  const start = best.length ? best.reduce((a, b) => a + b, 0) / best.length : 0.25;
  setConf(Math.round(start * 100) / 100, false);
  Viewer.show(results, id, +$("tConf").value);
  $("tInfo").innerHTML = `Test <code>${esc(id)}</code> · ${esc(metrics.timing?.device || "")}, ${metrics.timing?.cpu_threads || "?"} threads · `
    + `times are one model at a time (preprocess + inference + postprocess per image). AP50 and AP50-95 do not depend on the confidence. `
    + `Matching: IoU ≥ 0.5, greedy by confidence.`;
}

// per model / class: sorted confidences with running sums, so P/R/F1 at any confidence is a binary search
function prepMatches(matches) {
  const out = {};
  for (const [model, m] of Object.entries(matches)) {
    out[model] = m.classes.map((c) => {
      const n = c.d.length;
      const conf = new Float64Array(n), tp = new Int32Array(n + 1), u = new Int32Array(n + 1), cv = new Int32Array(n + 1);
      c.d.forEach(([s, t, fl], i) => {
        conf[i] = s;
        tp[i + 1] = tp[i] + t;
        u[i + 1] = u[i] + (t && fl >= 0 && fl & 1 ? 1 : 0);
        cv[i + 1] = cv[i] + (t && fl >= 0 && fl & 2 ? 1 : 0);
      });
      return { name: c.name, n_gt: c.n_gt, n_u: c.n_gt_uncompleted, n_c: c.n_gt_covered, conf, tp, u, cv };
    });
  }
  return out;
}

function countAt(c, t) {
  let lo = 0, hi = c.conf.length; // number of detections with conf >= t (sorted descending)
  while (lo < hi) { const mid = (lo + hi) >> 1; if (c.conf[mid] >= t) lo = mid + 1; else hi = mid; }
  return { k: lo, tp: c.tp[lo], u: c.u[lo], cv: c.cv[lo] };
}

function prf(k, tp, nGt) {
  const p = k ? tp / k : 0, r = nGt ? tp / nGt : 0;
  return { p, r, f1: p + r ? 2 * p * r / (p + r) : 0, tp, fp: k - tp, fn: nGt - tp };
}

function statsAt(model, t) {
  const cls = test.prep[model].map((c) => {
    const a = countAt(c, t);
    return { name: c.name, n_gt: c.n_gt, ...prf(a.k, a.tp, c.n_gt), ru: c.n_u ? a.u / c.n_u : null, rc: c.n_c ? a.cv / c.n_c : null };
  });
  const sum = (f) => cls.reduce((s, c) => s + c[f], 0);
  const nGt = sum("n_gt"), k = sum("tp") + sum("fp");
  const nu = test.prep[model].reduce((s, c) => s + c.n_u, 0), nc = test.prep[model].reduce((s, c) => s + c.n_c, 0);
  const hu = test.prep[model].reduce((s, c) => s + countAt(c, t).u, 0), hc = test.prep[model].reduce((s, c) => s + countAt(c, t).cv, 0);
  return { all: { n_gt: nGt, ...prf(k, sum("tp"), nGt), ru: nu ? hu / nu : null, rc: nc ? hc / nc : null }, cls };
}

function setConf(t, fromSlider = true) {
  t = Math.min(0.99, Math.max(0.01, t));
  $("tConf").value = t;
  $("tConfVal").textContent = t.toFixed(2);
  renderTestTable();
  renderCharts();
  if (fromSlider) { clearTimeout(setConf.timer); setConf.timer = setTimeout(() => Viewer.setConf(t), 250); }
}

const COLS = [
  ["model", "Model"], ["type", "Kind"], ["scope", "Scope"], ["n_images", "Images"], ["ap50", "AP50"], ["ap50_95", "AP50-95"],
  ["p", "P"], ["r", "R"], ["f1", "F1"], ["tp", "TP"], ["fp", "FP"], ["fn", "FN"], ["ru", "Recall uncompl."], ["rc", "Recall covered"],
  ["best", "Best F1 @ conf"], ["ms", "Median ms"], ["msmean", "Mean ms"],
];

function renderTestTable() {
  if (!test) return;
  const t = +$("tConf").value;
  const rows = test.metrics.models.map((m) => {
    const s = statsAt(m.name, t);
    return { m, s, v: { model: m.name, type: m.type, scope: m.scope_label, n_images: m.n_images, ap50: m.ap50, ap50_95: m.ap50_95,
      ...s.all, best: m.best?.f1, ms: m.ms_median, msmean: m.ms_mean } };
  });
  const { key, dir } = test.sort;
  rows.sort((a, b) => {
    const x = a.v[key], y = b.v[key];
    if (typeof x === "string" || typeof y === "string") return dir * String(x ?? "").localeCompare(String(y ?? ""));
    return dir * ((x ?? -Infinity) - (y ?? -Infinity));
  });
  const head = COLS.map(([k, label]) => `<th data-sort="${k}" class="sortable">${label}${key === k ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("");
  const cells = (v, m, isClass) => `
    <td>${isClass ? "" : esc(v.type ? KIND_LABELS[v.type] || v.type : "")}</td><td class="small">${isClass ? "" : esc(v.scope || "")}</td>
    <td>${isClass ? `${v.n_gt} gt` : v.n_images}</td>
    <td>${f3(v.ap50)}</td><td>${f3(v.ap50_95)}</td><td>${f3(v.p)}</td><td>${f3(v.r)}</td><td><b>${f3(v.f1)}</b></td>
    <td>${v.tp}</td><td>${v.fp}</td><td>${v.fn}</td><td>${pct(v.ru)}</td><td>${pct(v.rc)}</td>
    <td>${v.bestObj ? `${f3(v.bestObj.f1)} @ <a href="#" data-conf="${v.bestObj.conf}">${v.bestObj.conf.toFixed(2)}</a>` : "–"}</td>
    <td>${isClass ? "" : v.ms ?? "–"}</td><td>${isClass ? "" : v.msmean ?? "–"}</td>`;
  const body = rows.map(({ m, s, v }) => {
    const open = test.open.has(m.name);
    const main = `<tr class="mrow"><td><a href="#" data-open="${esc(m.name)}">${open ? "▾" : "▸"}</a> <b>${esc(m.name)}</b></td>${cells({ ...v, bestObj: m.best }, m, false)}</tr>`;
    if (!open) return main;
    return main + m.classes.map((c, i) => {
      const cs = s.cls[i];
      return `<tr class="crow"><td class="vindent">${esc(c.name)}</td>${cells({ ...cs, ap50: c.ap50, ap50_95: c.ap50_95, bestObj: c.best }, m, true)}</tr>`;
    }).join("");
  }).join("");
  $("tTable").innerHTML = `<table class="mtable"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// P / R / F1 over confidence for each model, with the slider position marked
function renderCharts() {
  if (!test) return;
  const t = +$("tConf").value;
  const W = 260, H = 120, pad = 22;
  const x = (c) => pad + c * (W - pad - 6), y = (v) => H - pad - v * (H - pad - 8);
  const line = (curve, i, color) => `<polyline fill="none" stroke="${color}" stroke-width="1.6" points="${curve.map((c) => `${x(c[0]).toFixed(1)},${y(c[i]).toFixed(1)}`).join(" ")}"/>`;
  $("tCharts").innerHTML = test.metrics.models.map((m) => {
    const curve = m.curve || [];
    const best = m.best;
    return `<figure class="chart"><figcaption><b>${esc(m.name)}</b> <span class="muted small">best F1 ${f3(best?.f1)} @ ${best?.conf ?? "–"}</span></figcaption>
      <svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="precision, recall and F1 against confidence for ${esc(m.name)}">
        <rect x="${pad}" y="8" width="${W - pad - 6}" height="${H - pad - 8}" class="cbg"/>
        ${[0, 0.5, 1].map((v) => `<text x="${pad - 4}" y="${y(v) + 3}" class="ctick" text-anchor="end">${v}</text>`).join("")}
        ${[0, 0.5, 1].map((v) => `<text x="${x(v)}" y="${H - 6}" class="ctick" text-anchor="middle">${v}</text>`).join("")}
        ${curve.length ? line(curve, 1, "var(--c-p)") + line(curve, 2, "var(--c-r)") + line(curve, 3, "var(--c-f1)") : ""}
        <line x1="${x(t)}" x2="${x(t)}" y1="8" y2="${H - pad}" class="cmark"/>
      </svg></figure>`;
  }).join("") + `<div class="legend small"><span class="lp">P</span> precision · <span class="lr">R</span> recall · <span class="lf">F1</span> · x = confidence</div>`;
}

async function deleteTest() {
  const id = $("tPick").value;
  if (!id || !confirm(`Delete test ${id} and its files in output/SAM3?`)) return;
  await api(`/api/tests/${id}`, { method: "DELETE" });
  test = null;
  $("tBody").hidden = true;
  const tests = await loadTests();
  if (tests.length) openTest(tests[0].id);
}

// ============================================================ wiring
$("bDataset").onchange = onDatasetChange;
$("bClass").onchange = onFormChange;
$("bKind").onchange = onFormChange;
for (const el of document.querySelectorAll('input[name="bScope"]')) el.onchange = onFormChange;
$("bIncU").onchange = onFormChange;
$("bIncC").onchange = onFormChange;
$("bMaxImages").oninput = $("bSeed").oninput = $("bSource").onchange = onFormChange;
$("bName").oninput = () => { nameTouched = true; };
$("bCopyFrom").onchange = copyPrompts;
$("bBuild").onclick = build;
$("tRun").onclick = runTest;
$("tPick").onchange = () => openTest($("tPick").value);
$("tDelete").onclick = deleteTest;
$("tConf").oninput = () => setConf(+$("tConf").value);
$("tTable").onclick = (e) => {
  const s = e.target.closest("[data-sort]"), o = e.target.closest("[data-open]"), c = e.target.closest("[data-conf]");
  if (s) { const k = s.dataset.sort; test.sort = { key: k, dir: test.sort.key === k ? -test.sort.dir : -1 }; renderTestTable(); }
  if (o) { e.preventDefault(); const n = o.dataset.open; test.open.has(n) ? test.open.delete(n) : test.open.add(n); renderTestTable(); }
  if (c) { e.preventDefault(); setConf(+c.dataset.conf); }
};
$("jobCancel").onclick = () => currentJobId && post(`/api/jobs/${currentJobId}/cancel`, {}).then(pollJob);
$("jobQueueToggle").onclick = () => { $("jobQueue").hidden = !$("jobQueue").hidden; };
$("jobQueue").onclick = (e) => {
  const c = e.target.closest("[data-cancel]");
  if (c) post(`/api/jobs/${c.dataset.cancel}/cancel`, {}).then(pollJob);
  if (e.target.id === "jobClear") post("/api/jobs/clear", {}).then(pollJob);
};
$("jobLogToggle").onclick = () => { $("jobLog").hidden = !$("jobLog").hidden; };
$("jobHide").onclick = () => { $("jobBar").dataset.hidden = $("jobBar").dataset.sig; $("jobBar").hidden = true; };

(async () => {
  await loadDataset();
  await loadModels().catch((e) => { $("serverInfo").textContent = e.message; });
  if (DS) onFormChange();
  const tests = await loadTests().catch(() => []);
  if (tests.length) openTest(tests[0].id);
  pollJob();
})();
