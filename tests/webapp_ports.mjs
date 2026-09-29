// Runs the JavaScript ports of the web demo on the inputs of a fixture written by
// tests/test_webdemo.py and prints what they return, for Python to compare with the originals.
//   node tests/webapp_ports.mjs fixture.json > result.json

import { readFileSync } from "node:fs";
import { prepare, resize, toGray } from "../webapp/preprocess.js";
import { heatmapFromCam, overlay } from "../webapp/heatmap.js";

const fixture = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = {};

out.resize = fixture.resize.map((c) =>
  Array.from(resize(Uint8Array.from(c.pixels), c.width, c.height, c.out_width, c.out_height, c.filter)));

out.gray = fixture.gray.map((c) => Array.from(toGray(Uint8Array.from(c.rgba), c.width, c.height)));

out.prepare = fixture.prepare.map((c) => {
  const { view, input } = prepare(Uint8Array.from(c.pixels), c.width, c.height, fixture.params);
  return { view: Array.from(view), input: Array.from(input) };
});

out.heatmap = fixture.heatmap.map((c) => Array.from(heatmapFromCam(Float32Array.from(c.cam), c.height, c.width, c.size)));

out.overlay = fixture.overlay.map((c) => {
  const rgba = overlay(Uint8Array.from(c.view), Float32Array.from(c.heatmap), fixture.lut, fixture.alpha);
  return Array.from(rgba).filter((_, i) => i % 4 !== 3);
});

process.stdout.write(JSON.stringify(out));
