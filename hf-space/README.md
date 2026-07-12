# MangaTranslatorAR

MangaTranslatorAR es una potente API y extensión de navegador (Chrome/Edge) para traducir mangas y cómics en tiempo real mediante un flujo de trabajo impulsado por IA. Funciona en tu propio entorno local o en la nube (como HuggingFace Spaces).

## Arquitectura de Traducción

El proceso se divide en los siguientes pasos (Pipeline):
1. **Detección (CTD)**: Encuentra las burbujas de diálogo y el texto.
2. **OCR (Manga OCR)**: Extrae el texto japonés original de las imágenes.
3. **Traducción**: Traduce el texto detectado al idioma deseado usando Google Translate, Ollama Local o LibreTranslate.
4. **Inpainting (Lama)**: Borra el texto original del manga limpiamente reconstruyendo el fondo.
5. **Renderizado**: Escribe el texto traducido sobre el fondo limpio.

---

## Instalación y Ejecución Local (con Docker)

La forma más sencilla de correr **MangaTranslatorAR** de forma local es mediante Docker, lo que garantiza que todas las dependencias (PyTorch, OpenCV, CUDA, etc.) estén correctamente aisladas.

### Requisitos previos
- [Docker](https://docs.docker.com/get-docker/) y [Docker Compose](https://docs.docker.com/compose/install/) instalados.
- Tarjeta Gráfica Nvidia (opcional pero altamente recomendada) y los [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Pasos

1. Clona este repositorio o navega hasta la carpeta del proyecto.
2. Levanta los contenedores usando Docker Compose:
   ```bash
   docker-compose up --build -d
   ```
3. La API estará disponible en `http://localhost:8001` (por defecto).
4. *(Opcional)* Si vas a usar **Ollama** o **LibreTranslate** para la traducción local, asegúrate de tenerlos corriendo en sus respectivos puertos (ej. `http://localhost:11434` y `http://localhost:5000`).

---

## Despliegue en HuggingFace Spaces

**MangaTranslatorAR** está diseñado para ser desplegado fácilmente en [HuggingFace Spaces](https://huggingface.co/spaces). Hemos creado una carpeta pre-configurada (`hf-space/`) que es 100% compatible tanto con el Free Tier de CPU como con los Spaces de pago de GPU.

### Cómo subirlo a HuggingFace

1. Crea un nuevo Space en tu cuenta de HuggingFace (elige la opción **Docker** como SDK).
2. Clona el repositorio de tu nuevo Space en tu computadora.
3. Copia **todo el contenido** que está dentro de la carpeta `hf-space/` de este proyecto y pégalo en el repositorio de tu Space.
4. Haz `git add .`, `git commit -m "Initial commit"` y `git push`.
5. HuggingFace construirá el Dockerfile de forma automática.
6. ¡Listo! La API estará alojada y lista para funcionar.

### Descargar la Extensión desde HuggingFace
Hemos agregado un endpoint especial en la versión de HuggingFace. Una vez que tu Space esté corriendo, puedes descargar la extensión lista para instalar y conectada yendo a:
```
https://huggingface.co/spaces/TU_USUARIO/TU_SPACE/download-extension
```
Descomprime el `.zip` y cárgalo en Chrome/Edge habilitando el "Modo Desarrollador" en la página de extensiones.

---

## Configuración de la Extensión

1. Abre tu navegador basado en Chromium (Chrome, Edge, Brave, etc.) y navega a `chrome://extensions/`.
2. Habilita el **Modo de desarrollador** (esquina superior derecha).
3. Haz clic en **Cargar descomprimida** (Load unpacked) y selecciona la carpeta `chrome-extension/`.
4. Una vez instalada, haz clic en el ícono de **MangaTranslatorAR**.
5. Ingresa la URL de la API:
   - **Uso Local**: `http://localhost:8001`
   - **Uso en HuggingFace**: La URL de tu Space (ej. `https://acsaco-mangatranslatorar.hf.space`)
6. ¡Disfruta traduciendo mangas en tu navegador con un solo clic!
