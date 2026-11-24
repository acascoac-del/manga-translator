# MangaTranslator Cloud

La edición comercial incluye web, cuentas verificadas, 50 créditos de prueba y un paquete prepago de US$6 por 1.000 imágenes estándar. La extensión comercial y el traductor de tu PC tienen instalaciones independientes.

## Prueba en la PC

```powershell
python -m venv commercial/.venv
commercial/.venv/Scripts/python.exe -m pip install -r commercial/requirements.txt
$env:COMMERCIAL_DEBUG = '1'
commercial/.venv/Scripts/python.exe commercial/manage.py migrate
commercial/.venv/Scripts/python.exe commercial/manage.py runserver 127.0.0.1:8010
```

Abrí http://127.0.0.1:8010. En desarrollo, el enlace de verificación aparece en la consola. La edición comercial usa SQLite para esta prueba. Producción utiliza PostgreSQL. Los controles de GPU y pagos están desactivados por defecto.

## RTX A5000 en RunPod Pods

El Pod es una instancia dedicada. La GPU se factura mientras el Pod está activo. Para tráfico variable y cobro por segundos de worker, usá la variante Serverless descrita más abajo.

1. Construí las imágenes desde la raíz. Usá un registro privado para la imagen de producción. Los pesos se mantienen en un volumen separado de Git.

   ```sh
   docker build -t mangatranslator-engine:commercial .
   docker build -f commercial/pod/Dockerfile -t REGISTRY/manga-pod:VERSION .
   docker push REGISTRY/manga-pod:VERSION
   ```

2. En el panel RunPod elegí una RTX A5000 de 24 GB, la imagen recién publicada y almacenamiento persistente montado en `/workspace`. Exponé `8010/http` a través del proxy HTTPS y, si vas a administrar desde la PC, `22/tcp` con tu clave pública SSH mediante `PUBLIC_KEY`. PostgreSQL permanece en loopback.
3. Cargá `COMMERCIAL_SECRET_KEY`, `POSTGRES_PASSWORD` y `COMMERCIAL_PUBLIC_URL` con el origen HTTPS del proxy que muestra RunPod. Si el Pod inicia antes de conocer su URL, completá las variables y volvé a iniciarlo. Configurá también SMTP y `MAIL_FROM` para activar cuentas. Confirmá que el proxy transmite la cabecera HTTPS esperada por Django y que `/health/` responde antes de aceptar tráfico.
4. Desde el cuadro **Connect** copiá a tu PC el comando SSH directo. Puedes transferir pesos mediante SCP con el mismo host, puerto y clave pública registrada. Ubicá CTD y LaMa bajo `/workspace/manga-models/weights`. Probá inglés y japonés y medí el rendimiento antes de activar `TRANSLATIONS_ENABLED=1`.
5. Configurá primero Stripe de prueba, `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` y el webhook `/webhooks/stripe/`. Probá compras y reembolsos. Habilitá `PAYMENTS_ENABLED=1` únicamente cuando el comercio sea elegible y hayas verificado el endpoint y completado condiciones y privacidad.
6. Generá el paquete de Chrome con `python commercial/build_extension.py --origin https://TU-ORIGEN`. El ZIP queda en `commercial/extension-build/`. El paquete funciona con la cuenta de la edición comercial. La publicación en Chrome Web Store se hace desde la cuenta del desarrollador.

Guardá copias de PostgreSQL fuera del Pod y verificá que se puedan restaurar. Eliminar un Pod puede eliminar su disco de volumen. Para producción mantené una copia separada del volumen de trabajo.

## Variante Serverless

Desplegá `commercial/compose.yml` en un host CPU, con `commercial/.env.example` como referencia. Construí la imagen de inferencia con `docker build -f commercial/runpod/Dockerfile -t REGISTRY/manga-serverless:VERSION .`. Usá un endpoint con workers mínimos en cero y una tarea por worker. El worker recibe `COMMERCIAL_CONTROL_URL` y el servidor CPU recibe las dos variables de RunPod. Prepará los pesos en el volumen del worker. Las reservas se confirman tras validar el resultado. Un envío ambiguo se concilia con `python manage.py reconcile_job` tras comprobar el panel de RunPod.
