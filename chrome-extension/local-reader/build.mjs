import fs from 'node:fs';
import path from 'node:path';
const root=process.cwd();
const vendor=path.join(root,'vendor');
fs.mkdirSync(path.join(vendor,'core'),{recursive:true});
function copy(src,dest){if(!fs.existsSync(src))throw Error(`Falta: ${src}. Ejecutá npm install antes.`);fs.copyFileSync(src,dest);console.log('Copiado:',dest);}
const tess=path.join(root,'node_modules','tesseract.js','dist');
const core=path.join(root,'node_modules','tesseract.js-core');
copy(path.join(tess,'tesseract.min.js'),path.join(vendor,'tesseract.min.js'));
copy(path.join(tess,'worker.min.js'),path.join(vendor,'worker.min.js'));
for(const name of ['tesseract-core.wasm.js','tesseract-core.wasm','tesseract-core-simd.wasm.js','tesseract-core-simd.wasm','tesseract-core-lstm.wasm.js','tesseract-core-lstm.wasm','tesseract-core-simd-lstm.wasm.js','tesseract-core-simd-lstm.wasm'])copy(path.join(core,name),path.join(vendor,'core',name));
console.log('\nOCR preparado. Cargá esta carpeta con chrome://extensions > Cargar descomprimida.');
