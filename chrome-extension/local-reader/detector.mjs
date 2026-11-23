const MAX_REGIONS = 35;

export function detectBubbles({ data, width, height }) {
  const area = width * height;
  const white = new Uint8Array(area);
  const queue = new Int32Array(area);
  for (let i = 0; i < area; i++) {
    const p = i * 4;
    white[i] = data[p + 3] >= 240 && Math.min(data[p], data[p + 1], data[p + 2]) >= 225;
  }

  const candidates = [];
  for (let i = 0; i < area; i++) {
    if (!white[i]) continue;
    let end = 0;
    function visit(index) {
      if (!white[index]) return;
      white[index] = 0;
      queue[end++] = index;
    }

    visit(i);
    let left = width, right = 0, top = height, bottom = 0;
    for (let cursor = 0; cursor < end; cursor++) {
      const index = queue[cursor];
      const x = index % width;
      const y = Math.floor(index / width);
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
      if (x > 0) visit(index - 1);
      if (x + 1 < width) visit(index + 1);
      if (y > 0) visit(index - width);
      if (y + 1 < height) visit(index + width);
    }

    if (!left || !top || right === width - 1 || bottom === height - 1) continue;
    const w = right - left + 1;
    const h = bottom - top + 1;
    const size = w * h;
    const fill = end / size;
    if (w < 20 || h < 24 || size > area * 0.10 || w / h > 3 || h / w > 5 || fill < 0.56) continue;

    // Look for ink inside white space, away from the outline and corners.
    let ink = 0;
    let pixels = 0;
    for (let y = top + Math.ceil(h * 0.22); y < bottom - h * 0.22; y++) {
      for (let x = left + Math.ceil(w * 0.23); x < right - w * 0.23; x++) {
        if (data[(y * width + x) * 4] < 120) ink++;
        pixels++;
      }
    }
    if (ink < 8 || ink / pixels > 0.32) continue;
    candidates.push([left, top, w, h]);
  }

  return candidates.sort((a, b) => a[1] - b[1] || b[0] - a[0]).slice(0, MAX_REGIONS);
}
