// Grad-CAM heatmap, ported from the Python pipeline (chestxray.gradcam): the model outputs, for
// each class, ReLU(sum_k alpha_k A_k) on the 7x7 grid (chestxray.webdemo.GradCamNet); here come
// pytorch-grad-cam's remaining steps (normalize to [0, 1], upsample with OpenCV's bilinear
// resize, normalize again) and the overlay of chestxray.gradcam.overlay_heatmap (matplotlib's
// jet colormap blended over the grayscale image). tests/test_webdemo.py compares both with Python.

// pytorch-grad-cam's scale_cam_image without resizing: x - min, then / (1e-7 + max)
function normalize(values) {
  let min = Infinity;
  for (const v of values) if (v < min) min = v;
  const out = new Float32Array(values.length);
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    out[i] = values[i] - min;
    if (out[i] > max) max = out[i];
  }
  const denom = Math.fround(1e-7 + max);
  for (let i = 0; i < out.length; i++) out[i] = out[i] / denom;
  return out;
}

// OpenCV's cv2.resize(INTER_LINEAR) of a float image: pixel centres aligned, edges replicated
export function resizeBilinear(src, height, width, outHeight, outWidth) {
  const axis = (outSize, inSize) => {
    const scale = inSize / outSize;
    const index = new Int32Array(outSize * 2);
    const weight = new Float32Array(outSize);
    for (let d = 0; d < outSize; d++) {
      let f = Math.fround((d + 0.5) * scale - 0.5);
      let s = Math.floor(f);
      f -= s;
      if (s < 0) { f = 0; s = 0; }
      if (s >= inSize - 1) { f = 0; s = inSize - 1; }
      index[d * 2] = s;
      index[d * 2 + 1] = Math.min(s + 1, inSize - 1);
      weight[d] = f;
    }
    return { index, weight };
  };
  const xs = axis(outWidth, width);
  const ys = axis(outHeight, height);
  const out = new Float32Array(outHeight * outWidth);
  for (let y = 0; y < outHeight; y++) {
    const r0 = ys.index[y * 2] * width;
    const r1 = ys.index[y * 2 + 1] * width;
    const fy = ys.weight[y];
    for (let x = 0; x < outWidth; x++) {
      const c0 = xs.index[x * 2];
      const c1 = xs.index[x * 2 + 1];
      const fx = xs.weight[x];
      const top = (1 - fx) * src[r0 + c0] + fx * src[r0 + c1];
      const bottom = (1 - fx) * src[r1 + c0] + fx * src[r1 + c1];
      out[y * outWidth + x] = (1 - fy) * top + fy * bottom;
    }
  }
  return out;
}

// One class's map on the feature grid -> heatmap in [0, 1] at the network input size
export function heatmapFromCam(cam, gridHeight, gridWidth, size) {
  const up = resizeBilinear(normalize(cam), gridHeight, gridWidth, size, size);
  for (let i = 0; i < up.length; i++) if (up[i] < 0) up[i] = 0;
  return normalize(up);
}

// RGBA pixels: (1 - alpha) * gray + alpha * jet(heatmap), as uint8 like numpy's cast
export function overlay(view, heatmap, lut, alpha) {
  const n = lut.length;
  const out = new Uint8ClampedArray(view.length * 4);
  for (let i = 0; i < view.length; i++) {
    const gray = Math.fround(view[i] / 255);
    let k = Math.floor(Math.fround(heatmap[i] * n));
    if (k >= n) k = n - 1;
    if (k < 0) k = 0;
    const color = lut[k];
    for (let c = 0; c < 3; c++) {
      const v = Math.min(1, Math.max(0, (1 - alpha) * gray + alpha * color[c]));
      out[i * 4 + c] = Math.floor(255 * v);
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}
