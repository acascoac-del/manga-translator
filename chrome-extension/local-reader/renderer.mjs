export function addCaption(layer, canvas, box, text, size) {
  const caption = document.createElement('div');
  caption.className = 'caption';
  caption.textContent = text;
  caption.style.left = 100 * (box.x + box.width / 2) / canvas.width + '%';
  caption.style.top = 100 * (box.y + box.height) / canvas.height + '%';
  caption.style.fontSize = size + 'px';
  layer.append(caption);
  return caption;
}
