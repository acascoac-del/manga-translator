ARG ENGINE_IMAGE=mangatranslator-engine:commercial
FROM ${ENGINE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends postgresql supervisor openssh-server ca-certificates && rm -rf /var/lib/apt/lists/*
COPY commercial/requirements.txt /tmp/commercial-requirements.txt
RUN pip install --no-cache-dir -r /tmp/commercial-requirements.txt
COPY commercial/config /srv/config
COPY commercial/billing /srv/billing
COPY commercial/templates /srv/templates
COPY commercial/static /srv/static
COPY commercial/manage.py /srv/manage.py
COPY commercial/runpod/handler.py /srv/runpod_worker/handler.py
COPY commercial/pod/ /opt/manga-pod/
RUN sed -i 's/\r$//' /opt/manga-pod/start.sh
COPY fonts/ /app/fonts/
ENV PYTHONPATH=/app:/srv PYTHONDONTWRITEBYTECODE=1 COMMERCIAL_POD_MODE=1
WORKDIR /srv
HEALTHCHECK NONE
EXPOSE 8010 22
ENTRYPOINT ["/bin/bash", "/opt/manga-pod/start.sh"]
