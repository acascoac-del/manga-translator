// Side-by-side notes: never paint over the illustration or change its source.
export function showReviewedTranslation(image, record, { confirmed = false } = {}) {
  if (!confirmed || record.status !== 'pending_review' || record.reason || !record.translation?.trim()) throw new Error('review_required');
  if (!image.isConnected) throw new Error('image_disconnected');
  const host = image.ownerDocument.createElement('aside');
  host.setAttribute('aria-label', 'Traducción revisada de MangaTranslate AI');
  const root = host.attachShadow({ mode: 'closed' });
  const text = image.ownerDocument.createElement('p');
  text.textContent = record.translation;
  const undo = image.ownerDocument.createElement('button');
  undo.textContent = 'Quitar traducción';
  undo.addEventListener('click', () => host.remove());
  root.append(text, undo);
  image.after(host);
  return () => host.remove();
}
