// Web demo (Phase 5): the same analysis as app/app.py (Gradio), run in the visitor's browser with
// ONNX Runtime Web. The files in model/ come from `python -m chestxray.webdemo`; see that module
// for what each one holds. Open index.html?selftest to compare the browser with Python.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { prepare, toGray } from "./preprocess.js";
import { heatmapFromCam, overlay } from "./heatmap.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation, so no threads

const MODEL_DIR = "model/";
const TOLERANCE = { scores: 1e-4, heatmap: 1e-3 };
const $ = (id) => document.getElementById(id);
const percent = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const seconds = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

let params = null;
let session = null;
let current = null; // last analysis: {view, cams, gridHeight, gridWidth, values}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const data = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { data.set(chunk, offset); offset += chunk.length; }
  return data;
}

// Any image the browser can decode -> 8-bit grayscale, like Pillow's convert("L")
async function decode(source) {
  const blob = source instanceof Blob ? source : await (await fetch(source)).blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();
  return { gray: toGray(data, canvas.width, canvas.height), width: canvas.width, height: canvas.height };
}

// Raw sigmoid score (float32, as PyTorch) -> value shown: Platt recalibration fitted on validation
function calibrate(score, [a, b]) {
  const s = Math.min(Math.max(score, 1e-12), 1 - 1e-12);
  return 1 / (1 + Math.exp(-(a * Math.log(s / (1 - s)) + b)));
}

async function run(image) {
  const { view, input } = prepare(image.gray, image.width, image.height, params);
  const size = params.image_size;
  const output = await session.run({ input: new ort.Tensor("float32", input, [1, 3, size, size]) });
  const scores = Array.from(output.logits.data, (z) => Math.fround(1 / (1 + Math.exp(-z))));
  const [, , gridHeight, gridWidth] = output.cams.dims;
  const values = params.classes.map((c, k) => calibrate(scores[k], params.platt[c]));
  return { view, scores, values, cams: output.cams.data, gridHeight, gridWidth };
}

function heatmapOf(result, className) {
  const k = params.classes.indexOf(className);
  const cells = result.gridHeight * result.gridWidth;
  const cam = result.cams.subarray(k * cells, (k + 1) * cells);
  return heatmapFromCam(cam, result.gridHeight, result.gridWidth, params.image_size);
}

// Focus classes first, each group by descending value (chestxray.demo.build_result)
function tableRows(values) {
  const rows = params.classes.map((c, k) => ({
    className: c,
    focus: params.focus.includes(c),
    value: values[k],
    threshold: params.thresholds[c],
  }));
  rows.sort((a, b) => (b.focus - a.focus) || (b.value - a.value));
  return rows.map((r) => ({ ...r, above: r.value >= r.threshold }));
}

function showResult(result) {
  const rows = tableRows(result.values);
  const above = rows.filter((r) => r.above).map((r) => params.names_pt[r.className]);
  $("summary").textContent = above.length ? `Achados acima do limiar: ${above.join(", ")}.` : params.texts.no_finding;

  const body = $("table").querySelector("tbody");
  body.replaceChildren(...rows.map((r) => {
    const tr = document.createElement("tr");
    tr.className = (r.focus ? "focus" : "") + (r.above ? " above" : "");
    const cells = [r.focus ? params.texts.focus_group : params.texts.other_group, params.names_pt[r.className],
      `${percent.format(100 * r.value)}%`, `${percent.format(100 * r.threshold)}%`,
      r.above ? params.texts.above : params.texts.below];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    }
    return tr;
  }));
  $("result").hidden = false;
  showHeatmap();
}

function showHeatmap() {
  if (!current) return;
  const className = $("klass").value;
  const heat = heatmapOf(current, className);
  const size = params.image_size;
  const canvas = $("heatmap");
  canvas.width = size;
  canvas.height = size;
  const pixels = overlay(current.view, heat, params.heatmap.lut, params.heatmap.alpha);
  canvas.getContext("2d").putImageData(new ImageData(pixels, size, size), 0, 0);
  const value = current.values[params.classes.indexOf(className)];
  $("caption").innerHTML = "";
  $("caption").append("Grad-CAM de ", Object.assign(document.createElement("strong"),
    { textContent: params.names_pt[className] }), `: ${params.label.toLowerCase()} ${percent.format(100 * value)}%`);
}

async function analyse(source, name) {
  if (!session) return;
  try {
    setStatus(`Analisando ${name}…`);
    const start = performance.now();
    const image = await decode(source);
    current = await run(image);
    showResult(current);
    setStatus(`${name}: análise feita no seu navegador em ${seconds.format((performance.now() - start) / 1000)} s.`);
  } catch (err) {
    console.error(err);
    setStatus(`Não foi possível analisar ${name}. Envie uma radiografia de tórax em PNG ou JPG.`, true);
  }
}

function buildInterface() {
  const notes = $("notes");
  for (const note of params.texts.notes) notes.append(Object.assign(document.createElement("p"), { textContent: note }));
  $("value-header").textContent = params.label;
  $("table-title").textContent = `${params.label} de cada doença e limiar escolhido na validação`;

  const select = $("klass");
  const ordered = [...params.focus, ...params.classes.filter((c) => !params.focus.includes(c))];
  for (const c of ordered) select.append(new Option(params.names_pt[c], c));
  select.addEventListener("change", showHeatmap);

  $("file").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) analyse(file, file.name);
  });

  const thumbs = $("examples");
  for (const name of params.examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thumb";
    button.title = name;
    button.append(Object.assign(document.createElement("img"), { src: `${MODEL_DIR}examples/${name}`, alt: name }));
    button.addEventListener("click", () => analyse(`${MODEL_DIR}examples/${name}`, name));
    thumbs.append(button);
  }
}

// ---------------------------------------------------------------- selftest (index.html?selftest)

function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

async function selftest() {
  const expected = await (await fetch(`${MODEL_DIR}selftest.json`)).json();
  const results = [];
  for (const [name, ref] of Object.entries(expected)) {
    const result = await run(await decode(`${MODEL_DIR}examples/${name}`));
    const heat = heatmapOf(result, ref.class);
    results.push({
      image: name,
      view: maxAbsDiff(result.view, ref.view.flat()),
      scores: maxAbsDiff(result.scores, ref.scores),
      heatmap: maxAbsDiff(heat, ref.heatmap.flat()),
      className: ref.class,
    });
  }
  const passed = results.every((r) => r.view === 0 && r.scores <= TOLERANCE.scores && r.heatmap <= TOLERANCE.heatmap);
  window.selftestResult = { passed, results };
  const section = $("selftest");
  section.hidden = false;
  section.querySelector("p").textContent = passed
    ? "Autoteste: o navegador reproduz o pipeline do Python."
    : "Autoteste: há diferenças acima da tolerância.";
  section.querySelector("tbody").replaceChildren(...results.map((r) => {
    const tr = document.createElement("tr");
    for (const text of [r.image, r.view, r.scores.toExponential(2), `${r.heatmap.toExponential(2)} (${r.className})`]) {
      tr.append(Object.assign(document.createElement("td"), { textContent: String(text) }));
    }
    return tr;
  }));
}

async function main() {
  try {
    params = await (await fetch(`${MODEL_DIR}params.json`)).json();
    buildInterface();
    const bytes = await fetchWithProgress(`${MODEL_DIR}model.onnx`, (received, total) =>
      setStatus(`Carregando o modelo: ${Math.round(received / 1e6)} de ${Math.round(total / 1e6)} MB…`));
    setStatus("Preparando o modelo…");
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    setStatus("Pronto. Escolha uma radiografia ou um dos exemplos.");
    document.body.classList.add("ready");
    if (new URLSearchParams(location.search).has("selftest")) await selftest();
  } catch (err) {
    console.error(err);
    setStatus("Não foi possível carregar o modelo. Recarregue a página.", true);
  }
}

main();
