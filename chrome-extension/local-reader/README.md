# MangaTranslate AI 0.4 local

El lector local se carga como una extensión MV3 independiente. La extensión MangaTranslatorAR y su API siguen disponibles en sus rutas actuales. Esta variante parte del lector 0.3.1 y aborda el [Issue #1](https://github.com/acascoac-del/manga-translator/issues/1).

## Desarrollo desde el repositorio

```sh
cd chrome-extension/local-reader
npm ci --ignore-scripts
npm run build
npm run check
```

En Chrome de escritorio, abrí `chrome://extensions`, activá el modo desarrollador y cargá `chrome-extension/local-reader/dist` como descomprimida. Para actualizar, ejecutá `git pull`, repetí la compilación y pulsá **Recargar**. El directorio generado contiene el manifiesto, el worker, WASM y los modelos de OCR. El primer uso de Chrome Translator puede descargar paquetes gestionados por Chrome.

## Flujo local y revisión

- El detector busca zonas claras cerradas con trazos oscuros en el interior y filtra por tamaño y forma. Los globos abiertos, oscuros o con bordes interrumpidos pueden requerir selección manual. Una ilustración con estas características también puede aparecer como candidata.
- El OCR elimina el contorno de los candidatos en el recorte, mejora el contraste con un umbral local, amplía 2–3× cuando el tamaño lo permite y usa `jpn_vert` con PSM 5 y 3 según el resultado. La lectura agrupa el texto de un globo y ordena sus columnas de derecha a izquierda. Se conservó Tesseract WASM porque puede distribuirse en MV3 junto con el worker y sus modelos. Manga OCR está especializado en manga y queda como alternativa para comparar con una distribución local compatible con MV3 y un corpus etiquetado. [Manga OCR](https://github.com/kha-white/manga-ocr)
- El lector no envía al traductor OCR automático por debajo del 90 % ni texto mezclado que falle la validación. El registro desplegable muestra caja, texto, confianza, PSM, intentos, tiempo, traducción y motivo de descarte. El lote usa el par japonés → español y traduce un globo completo. El contexto entre globos y la fidelidad semántica quedan pendientes de evaluación. [Chrome Translator](https://developer.chrome.com/docs/ai/translator-api)
- Las propuestas quedan para revisión. Confirmar crea una anotación extraíble que mantiene intactos los píxeles de la imagen. **Deshacer** la retira y **Guardar original** exporta la imagen como PNG sin anotaciones.
- Free y Premium a US$ 2,99 por mes siguen previstos. Los cobros están desactivados y el lector funciona sin servidor propio de inferencia.

## Prueba en Chrome

`node tests/chrome-smoke.mjs` carga el paquete en Chrome mediante CDP. Definí `CHROME_BIN` con la ruta de Chrome y usá una instalación con fuentes japonesas. En un perfil compatible, `TEST_LOCAL_TRANSLATION=1` prueba además la propuesta automática del primer globo, con descarga inicial si Chrome la requiere.

Las tres páginas sintéticas tienen una referencia japonesa conocida. Chrome 157 en Windows detectó un globo en cada una. Las tres transcripciones coincidieron con su referencia. Las confianzas fueron 0,95, 0,93 y 0,84, por lo que la tercera se descartó. El primer globo generó la propuesta local «buen tiempo hoy» sin añadir una anotación. Crear y retirar una anotación dejó idéntico el PNG del lienzo en las tres páginas. Queda pendiente medir falsos positivos, OCR y fidelidad de traducción en tres páginas de manga real con autorización y referencias revisadas.

## Modelos OCR

El contenido descomprimido de los archivos heredados coincide con [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast/tree/87416418657359cb625c412a48b6e1d6d41c29bd), bajo Apache 2.0. La licencia se incluye en `lang/LICENSE`.

| Modelo | SHA-256 del contenido descomprimido |
| --- | --- |
| jpn | `1f5de9236d2e85f5fdf4b3c500f2d4926f8d9449f28f5394472d9e8d83b91b4d` |
| jpn_vert | `bf1e2640954691797e2dc14f38533e601b59ee37958698ae0f0b81dc6f09c71b` |
