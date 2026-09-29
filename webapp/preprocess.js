// Image preparation, ported from the Python pipeline (chestxray.inference.prepare_image) so the
// browser feeds the network exactly the same numbers:
//   grayscale (Pillow's convert("L")) -> Lanczos to the stored size (256) -> bilinear to the
//   network size (224, torchvision's Resize on a PIL image) -> 3 channels, ImageNet normalization.
// The resizing is Pillow's own algorithm (libImaging/Resample.c): separable filters, horizontal
// pass then vertical pass, 8-bit fixed-point coefficients, and rounding to 8 bits after each pass.
// tests/test_webdemo.py compares every step with Pillow.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

function sinc(x) {
  if (x === 0.0) return 1.0;
  x *= Math.PI;
  return Math.sin(x) / x;
}

export const FILTERS = {
  bilinear: { support: 1.0, fn: (x) => { x = Math.abs(x); return x < 1.0 ? 1.0 - x : 0.0; } },
  lanczos: { support: 3.0, fn: (x) => (x >= -3.0 && x < 3.0 ? sinc(x) * sinc(x / 3.0) : 0.0) },
};

// Pillow's precompute_coeffs + normalize_coeffs_8bpc
function coefficients(inSize, outSize, filter) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1.0 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = filter.fn((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      let w = x < xmax ? k[x] : 0.0;
      if (x < xmax && ww !== 0.0) w /= ww;
      kk[xx * ksize + x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

// Resize an 8-bit grayscale image (Uint8Array, row-major) like Pillow's Image.resize.
export function resize(src, width, height, outWidth, outHeight, filterName) {
  if (width === outWidth && height === outHeight) return Uint8Array.from(src);
  const filter = FILTERS[filterName];
  const horiz = coefficients(width, outWidth, filter);
  const vert = coefficients(height, outHeight, filter);
  const needHorizontal = outWidth !== width;
  const needVertical = outHeight !== height;

  let img = src;
  let imgWidth = width;
  let rowOffset = 0;
  if (needHorizontal) {
    // only the rows the vertical pass reads
    const first = vert.bounds[0];
    const last = vert.bounds[outHeight * 2 - 2] + vert.bounds[outHeight * 2 - 1];
    const rows = needVertical ? last - first : height;
    const start = needVertical ? first : 0;
    const tmp = new Uint8Array(outWidth * rows);
    for (let yy = 0; yy < rows; yy++) {
      const srcRow = (yy + start) * width;
      for (let xx = 0; xx < outWidth; xx++) {
        const xmin = horiz.bounds[xx * 2];
        const xmax = horiz.bounds[xx * 2 + 1];
        const base = xx * horiz.ksize;
        let ss = ONE / 2;
        for (let x = 0; x < xmax; x++) ss += src[srcRow + x + xmin] * horiz.kk[base + x];
        tmp[yy * outWidth + xx] = clip8(ss);
      }
    }
    img = tmp;
    imgWidth = outWidth;
    rowOffset = needVertical ? first : 0;
  }
  if (!needVertical) return img;
  const out = new Uint8Array(outWidth * outHeight);
  for (let yy = 0; yy < outHeight; yy++) {
    const ymin = vert.bounds[yy * 2] - rowOffset;
    const ymax = vert.bounds[yy * 2 + 1];
    const base = yy * vert.ksize;
    for (let xx = 0; xx < outWidth; xx++) {
      let ss = ONE / 2;
      for (let y = 0; y < ymax; y++) ss += img[(y + ymin) * imgWidth + xx] * vert.kk[base + y];
      out[yy * outWidth + xx] = clip8(ss);
    }
  }
  return out;
}

// Pillow's convert("L") of an RGB(A) image: L = (19595 R + 38470 G + 7471 B + 0x8000) >> 16.
// The alpha channel is ignored, as in Pillow.
export function toGray(rgba, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

// Grayscale image -> {view: the 8-bit image the network sees (size x size),
//                     input: Float32Array 1x3xsize x size, normalized}
export function prepare(gray, width, height, params) {
  const stored = params.stored_size;
  const size = params.image_size;
  const reduced = resize(gray, width, height, stored, stored, "lanczos");
  const view = resize(reduced, stored, stored, size, size, "bilinear");
  const plane = size * size;
  const input = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const mean = Math.fround(params.mean[c]);
    const std = Math.fround(params.std[c]);
    for (let i = 0; i < plane; i++) {
      // torchvision: ToTensor (x / 255 in float32), then (x - mean) / std in float32
      const x = Math.fround(view[i] / 255);
      input[c * plane + i] = Math.fround(Math.fround(x - mean) / std);
    }
  }
  return { view, input };
}
