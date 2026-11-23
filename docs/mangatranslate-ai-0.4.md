# MangaTranslate AI 0.4: primera entrega experimental

Referencia: https://github.com/acascoac-del/manga-translator/issues/1

## Auditoría y alcance

El HEAD inicial eddfac7 contiene MangaTranslatorAR 1.0, una extensión MV3 conectada a una API Python. No contiene la versión 0.3.1 ni Tesseract jpn_vert. El Issue #1 confirma que esa versión se entregó externamente. El usuario identificó su fuente en `C:\Users\chevi\Downloads\MangaTranslate-AI-0.3.1-SAFE-SOURCE`, inaccesible desde Codex Cloud. Importar esa fuente es un requisito para corregir y reproducir los fallos específicos de 0.3.1. No se reemplazó la aplicación existente ni se cambió su numeración.

La entrada experimental del popup abre una página de diagnóstico local; no procesa automáticamente la página activa. El flujo anterior sigue dependiendo de su servidor. Esta entrega NO convierte todavía la extensión existente a ejecución íntegramente local.

## Módulos nuevos

- `local-ai/pipeline.mjs`: contratos de detector y OCR, validación de candidatos de globo (geometría, tamaño máximo del 15%, clase y confianza mínima 0,9), rechazo de OCR espurio o confianza inferior a 0,9, orden de columnas verticales, registro por región y cola de revisión. Los umbrales son conservadores y provisionales: necesitan calibración sobre páginas reales. Una puntuación alta no demuestra corrección semántica.
- Adaptador `browserTranslator`: comprueba el modelo japonés → español mediante la API Translator del navegador. Solo utiliza un modelo ya disponible; no descarga ni recurre a servicios remotos. La API puede faltar o no admitir el par en la versión de Chrome/plataforma elegida. La disponibilidad debe comprobarse también en la página de extensión real.
- `local-ai/overlay.mjs`: anotación adyacente con Shadow DOM, texto seguro y acción de quitar. Exige confirmación explícita. No cambia `src`, no pinta rectángulos ni aplica inpainting. Todavía no está conectada al procesamiento de la página activa.
- `local-ai/status.html`: diagnóstico accesible desde el popup. Free + Premium previsto a US$ 2,99 mensuales; sin cobros, claves ni llamadas de IA pagas.

Los adapters de detección y OCR reales NO están implementados. Pasar opciones `jpn_vert`, escala 3 y PSM 5/6 al contrato no equivale a ejecutar preprocesamiento o Tesseract. No se anuncian regiones como traducidas: las salidas válidas quedan `pending_review`. El contador final debe depender de confirmaciones reales cuando se conecte la interfaz.

## Integración pendiente y elección de OCR

1. Incorporar 0.3.1 en una rama de GitHub preservando su contenido; revisar sus dependencias y licencias antes de integrarlas. No ejecutar scripts desconocidos sin inspección.
2. Separar detector de globos y detector de texto; medir falsos positivos sobre ilustraciones. Usar propuestas conservadoras, nunca cajas de texto como prueba suficiente de globo.
3. Comparar Tesseract WASM empaquetado con jpn_vert (crop aislado, contraste, binarización adaptable, escala 2/3, PSM 5/6 y columnas de derecha a izquierda) frente a Manga OCR con ONNX/WebGPU, si existe exportación compatible y licencia válida. Tesseract ofrece distribución local conocida pero sus confianzas requieren calibración; Manga OCR se especializa en manga, con mayor peso y complejidad de runtime. No elegir por intuición ni usar OCR Python en el flujo sin servidor.
4. Empaquetar JS, workers y WASM localmente, documentar licencia/origen/checksums de modelos y adaptar CSP de MV3 solo según necesidades verificadas. No incorporar scripts remotos. El manifiesto heredado conserva permisos amplios y rutas de API; revisar su reducción cuando se migre el flujo.
5. Verificar traducción japonesa → española real, sin tratar idioma o confianza OCR como prueba de fidelidad. La API Translator no ofrece contexto entre globos: queda pendiente probar un modelo local contextual y conservar orden de lectura.
6. Evaluar al menos tres páginas verticales con licencia o permiso, transcripción japonesa y traducción de referencia revisadas manualmente. El repositorio solo aporta manga_test.jpg sin referencias: no basta para demostrar precisión. Anotar por globo bbox, texto original, CER, confianza, tiempo, traducción y motivo de descarte. No subir corpus con derechos inciertos.

## Pruebas y flujo GitHub

Desde la raíz: `node --test tests/local-ai.test.mjs`. Son pruebas unitarias con adapters simulados, no una evaluación OCR ni traducción real. CI comprueba estas pruebas, sintaxis de módulos y MV3.

`tests/browser-smoke.html` permite comprobar en Chromium confirmación, creación y retirada de anotación y conservación de imagen. Servir la raíz con `python -m http.server 8765 --bind 127.0.0.1` y abrir internamente `/tests/browser-smoke.html`. Su prueba del traductor informa disponibilidad; no inventa una salida. Esto no sustituye cargar la extensión MV3 en Chrome y validar tres páginas reales.

Para desarrollar sin ZIP: clonar una vez o abrir el repositorio en Codex, seleccionar la rama `develop/mangatranslate-ai-0.4`, editar, ejecutar pruebas, hacer commits y abrir PR contra main. En la PC, `git pull` y Recargar en chrome://extensions actualiza la carpeta `chrome-extension/` cargada como descomprimida. Chrome aún requiere acceso a una carpeta local durante desarrollo; GitHub/Codex no actualizan automáticamente una extensión instalada. No se publican pagos ni Chrome Web Store en esta fase.

## Resultado de esta instancia

- 9/9 pruebas unitarias aprobadas, incluida disponibilidad del modelo que no responde.
- Sintaxis de los módulos y manifiesto MV3 verificadas; `git diff --check` sin errores.
- Chromium headless solicitó la página y ambos módulos, pero no produjo resultado DOM antes del límite externo de 15/20 segundos. Ensayo incompleto: no se afirma que el overlay haya pasado la prueba de navegador ni que la traducción local funcione aquí.
- La rama se publicó con Git HTTPS. La creación automática de PR mediante `gh pr create` recibió `Forbidden` de `api.github.com/graphql`; no hay PR creado confirmado. La importación de 0.3.1 y la evaluación real siguen pendientes.
