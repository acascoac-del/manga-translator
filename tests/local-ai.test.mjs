import test from 'node:test';
import assert from 'node:assert/strict';
import { validateBubble, validateJapanese, orderVerticalColumns, browserTranslator, reviewPage, PLAN } from '../chrome-extension/local-ai/pipeline.mjs';
const page = { width: 1000, height: 1400 };
const bubble = { bbox: [100, 100, 100, 200], kind: 'speech_bubble', confidence: .97 };
test('reject illustration candidates, huge and off-page boxes, and invalid confidence', () => {
  assert.equal(validateBubble(bubble, page), null);
  for (const candidate of [{...bubble, kind:'text'}, {...bubble, bbox:[0,0,1000,1000]}, {...bubble, bbox:[990,0,100,100]}, {...bubble, confidence:NaN}, {...bubble, confidence:2}]) assert.ok(validateBubble(candidate,page));
});
test('OCR rejects low confidence and mixed or non-Japanese text', () => {
  assert.equal(validateJapanese({text:'こんにちは',confidence:.95}),null);
  for(const result of [{text:'こんにちは',confidence:.2},{text:'Mini Sashima desbordante S',confidence:.99},{text:'こんにちはABC',confidence:.99},{text:'あ',confidence:.99}]) assert.ok(validateJapanese(result));
});
test('vertical columns read right to left without mutating adapter output', () => {
 const columns=[{bbox:[10,0,5,20]},{bbox:[100,20,5,20]},{bbox:[100,0,5,20]}];
 assert.deepEqual(orderVerticalColumns(columns),[columns[2],columns[1],columns[0]]);
 assert.equal(columns[0].bbox[0],10);
});
test('local translator fails closed without API or installed model', async () => {
 await assert.rejects(browserTranslator(null),/unavailable/);
 let created=false;
 await assert.rejects(browserTranslator({availability:async()=> 'downloadable',create:()=>{created=true;}}),/downloadable/);
 assert.equal(created,false);
 const api={availability:async options=>{assert.deepEqual(options,{sourceLanguage:'ja',targetLanguage:'es'});return 'available';},create:async()=>({translate:async()=> 'Hola'})};
 assert.equal(await (await browserTranslator(api)).translate('こんにちは'),'Hola');
});
test('rejected bubble never reaches OCR; rejected OCR never reaches translation',async()=>{
 let ocrCalls=0, translationCalls=0;
 const records=await reviewPage({page,detector:{detect:async()=>[{...bubble,kind:'illustration'},bubble]},ocr:{recognize:async()=>{ocrCalls++;return {text:'ABC',confidence:.99};}},translator:{translate:async()=>{translationCalls++;}}});
 assert.equal(ocrCalls,1);assert.equal(translationCalls,0);assert.ok(records.every(r=>r.status==='rejected' && r.reason));
});
test('valid output stays pending review, with per-bubble diagnostics and no painting',async()=>{
 const records=await reviewPage({page,detector:{detect:async()=>[bubble]},ocr:{recognize:async(p,b,options)=>{assert.equal(options.language,'jpn_vert');return {text:'こんにちは',confidence:.95};}},translator:{translate:async()=> 'Hola'}});
 assert.equal(records[0].status,'pending_review');assert.equal(records[0].sourceText,'こんにちは');assert.equal(records[0].translation,'Hola');assert.equal(records[0].reason,null);assert.ok(records[0].elapsedMs>=0);
});
test('missing model errors and untranslated outputs remain rejected',async()=>{
 for(const translate of [async()=>{throw Error('missing_model');},async()=> 'こんにちは',async()=> '']) {
 const [record]=await reviewPage({page,detector:{detect:async()=>[bubble]},ocr:{recognize:async()=>({text:'こんにちは',confidence:.99})},translator:{translate}});
 assert.equal(record.status,'rejected');assert.ok(record.reason);
 }
});
test('billing remains disabled',()=>{assert.equal(PLAN.billingActive,false);assert.equal(PLAN.premiumUSDMonthly,2.99);});

test('model availability timeout fails closed', async()=>{
 await assert.rejects(browserTranslator({availability:()=>new Promise(()=>{}),create:()=>{throw Error('must not create');}},10),/local_translation_timeout/);
});
