import { orderVerticalColumns } from './pipeline.mjs';

export function prepareCrop(crop, maskOutline = false) {
  const prepared = document.createElement('canvas');
  const scale = Math.min(crop.width < 160 ? 3 : 2, 1600 / Math.max(crop.width, crop.height));
  prepared.width = Math.ceil(crop.width * scale) + 24;
  prepared.height = Math.ceil(crop.height * scale) + 24;
  const ctx = prepared.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, prepared.width, prepared.height);
  ctx.drawImage(crop, 12, 12, prepared.width - 24, prepared.height - 24);
  const { width, height } = prepared;
  const image = ctx.getImageData(0, 0, width, height);
  const gray = new Uint8Array(width * height);
  const stride = width + 1;
  const sums = new Uint32Array(stride * (height + 1));
  for (let y = 0; y < height; y++) {
    let row = 0;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const p = i * 4;
      gray[i] = (image.data[p] * 3 + image.data[p + 1] * 6 + image.data[p + 2]) / 10;
      row += gray[i];
      sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + row;
    }
  }

  // Candidate boxes come from bright interiors. Ignore the background outside
  // each row's interior, keeping the dark letter strokes between its edges.
  const edges = new Int32Array(height * 2);
  if (maskOutline) {
    const inset = Math.round(width * 0.05);
    for (let y = 0; y < height; y++) {
      let left = 12;
      let right = width - 13;
      while (left <= right && gray[y * width + left] < 225) left++;
      while (right >= left && gray[y * width + right] < 225) right--;
      edges[y * 2] = left + inset;
      edges[y * 2 + 1] = right - inset;
    }
  }

  const radius = 16;
  for (let y = 0; y < height; y++) {
    const top = Math.max(0, y - radius);
    const bottom = Math.min(height, y + radius + 1);
    for (let x = 0; x < width; x++) {
      const left = Math.max(0, x - radius);
      const right = Math.min(width, x + radius + 1);
      const sum = sums[bottom * stride + right] - sums[bottom * stride + left]
        - sums[top * stride + right] + sums[top * stride + left];
      const mean = sum / ((right - left) * (bottom - top));
      const i = y * width + x;
      const inside = !maskOutline || (x >= edges[y * 2] && x <= edges[y * 2 + 1]);
      const value = inside && gray[i] < Math.max(60, mean - 18) ? 0 : 255;
      image.data.fill(value, i * 4, i * 4 + 3);
    }
  }
  ctx.putImageData(image, 0, 0);
  return prepared;
}

export function readText(data, vertical) {
  if (!vertical || !data.lines?.length) return String(data.text ?? '').trim();
  const columns = data.lines.filter(line => line.bbox && line.text?.trim()).map(line => ({
    bbox: [line.bbox.x0, line.bbox.y0],
    text: line.text.trim().replace(/\s+/gu, ''),
  }));
  return orderVerticalColumns(columns).map(column => column.text).join('') || String(data.text ?? '').trim();
}

export async function recognizeCrop(worker, crop, vertical, validate, maskOutline = false) {
  const image = prepareCrop(crop, maskOutline);
  const attempts = [];
  for (const psm of vertical ? ['5', '3'] : ['6', '11']) {
    await worker.setParameters({ tessedit_pageseg_mode: psm });
    const { data } = await worker.recognize(image);
    const result = { text: readText(data, vertical), confidence: data.confidence / 100, psm };
    attempts.push(result);
    if (!validate(result)) return { ...result, attempts };
  }
  const best = attempts.filter(item => item.text).sort((a, b) => b.confidence - a.confidence)[0];
  return { ...(best ?? attempts[0]), attempts };
}
