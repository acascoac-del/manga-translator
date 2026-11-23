import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { detectBubbles } from '../detector.mjs';
import { validateJapanese } from '../../local-ai/pipeline.mjs';
import { localTranslator, reviewableSpanish } from '../translator.mjs';

test('detects an enclosed region with ink and skips blank and open regions', () => {
  const width = 300;
  const height = 200;
  const data = new Uint8ClampedArray(width * height * 4);
  function fill(left, top, w, h, color) {
    for (let y = top; y < top + h; y++) {
      for (let x = left; x < left + w; x++) data.set([color, color, color, 255], (y * width + x) * 4);
    }
  }
  fill(0, 0, width, height, 20);
  fill(35, 25, 50, 65, 255);
  fill(57, 44, 7, 24, 15);
  fill(120, 25, 50, 65, 255);
  fill(0, 112, 48, 60, 255);
  fill(14, 131, 7, 24, 15);
  fill(180, 75, 115, 120, 255);
  fill(231, 112, 7, 24, 15);
  assert.deepEqual(detectBubbles({ data, width, height }), [[35, 25, 50, 65]]);
});

test('rejects low-confidence, mixed OCR and malformed model output', () => {
  assert.equal(validateJapanese({ text: '電車が来ました', confidence: 0.84 }), 'low_ocr_confidence');
  assert.equal(validateJapanese({ text: '電車が来ましたＳ', confidence: 0.99 }), 'mixed_or_non_japanese');
  assert.equal(reviewableSpanish('電車が来ました', '電車が来ました'), false);
  assert.equal(reviewableSpanish('電車が来ました', 'Llegó el tren.'), true);
});

test('reports unsupported models without calling create', async () => {
  const previous = globalThis.Translator;
  globalThis.Translator = {
    availability: async options => {
      assert.deepEqual(options, { sourceLanguage: 'ja', targetLanguage: 'es' });
      return 'unavailable';
    },
    create: () => assert.fail('unavailable model should not be created'),
  };
  try {
    await assert.rejects(localTranslator(() => {})('ja'), /ja → es/);
  } finally {
    if (previous) globalThis.Translator = previous;
    else delete globalThis.Translator;
  }
});

test('build includes MV3 scripts, WASM and Japanese models', async () => {
  const dist = new URL('../dist/', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('manifest.json', dist), 'utf8'));
  assert.equal(manifest.manifest_version, 3);
  assert.match(manifest.content_security_policy.extension_pages, /script-src 'self' 'wasm-unsafe-eval'/);
  const files = ['reader.js', 'pipeline.mjs', 'detector.mjs', 'ocr.mjs', 'translator.mjs', 'renderer.mjs',
    'vendor/worker.min.js', 'vendor/tesseract.min.js', 'vendor/core/tesseract-core-simd-lstm.wasm',
    'lang/jpn.traineddata.gz', 'lang/jpn_vert.traineddata.gz', 'lang/LICENSE'];
  for (const file of files) assert.ok((await readFile(new URL(file, dist))).length, file);
});
