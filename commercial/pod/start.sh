#!/usr/bin/env bash
set -euo pipefail
: "${COMMERCIAL_PUBLIC_URL:?Set the HTTPS URL of your RunPod proxy}"
: "${COMMERCIAL_SECRET_KEY:?Set a random Django secret}"
: "${POSTGRES_PASSWORD:?Set a random PostgreSQL password}"
export POSTGRES_HOST=127.0.0.1 POSTGRES_USER=manga POSTGRES_DB=manga
export COMMERCIAL_DATA_DIR=/workspace/manga-data HF_HOME=/workspace/manga-models/hf
export MODEL_DIR=/workspace/manga-models/weights
export EASYOCR_MODULE_PATH=/workspace/manga-models/easyocr
pg_bin="$(pg_config --bindir)"
export POD_PG_BIN="$pg_bin"
export POD_PG_DATA=/workspace/manga-postgres
install -d -m 0700 -o postgres -g postgres "$POD_PG_DATA"
install -d -o manga -g manga "$COMMERCIAL_DATA_DIR" "$HF_HOME" "$MODEL_DIR" "$EASYOCR_MODULE_PATH" /srv/staticfiles
# llama.cpp uses this fixed path. Point it into the persistent volume.
if [[ -d /app/models/weights && ! -L /app/models/weights ]]; then
    rmdir /app/models/weights
    ln -s "$MODEL_DIR" /app/models/weights
fi
if [[ ! -s "$POD_PG_DATA/PG_VERSION" ]]; then
    runuser -u postgres -- "$pg_bin/initdb" -D "$POD_PG_DATA" --auth-local=peer --auth-host=scram-sha-256
fi
runuser -u postgres -- "$pg_bin/pg_ctl" -D "$POD_PG_DATA" -o "-h 127.0.0.1" -w start
trap 'runuser -u postgres -- "$pg_bin/pg_ctl" -D "$POD_PG_DATA" -m fast -w stop' EXIT
runuser -u postgres -- "$pg_bin/psql" -v ON_ERROR_STOP=1 <<'SQL'
\getenv manga_password POSTGRES_PASSWORD
SELECT 'CREATE ROLE manga LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname='manga') \gexec
ALTER ROLE manga PASSWORD :'manga_password';
SELECT 'CREATE DATABASE manga OWNER manga' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='manga') \gexec
SQL
runuser -u manga --preserve-environment -- /opt/venv/bin/python manage.py migrate --noinput
runuser -u manga --preserve-environment -- /opt/venv/bin/python manage.py collectstatic --noinput
runuser -u postgres -- "$pg_bin/pg_ctl" -D "$POD_PG_DATA" -m fast -w stop
trap - EXIT
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    install -d -m 0700 /root/.ssh /run/sshd
    umask 077
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    ssh-keygen -A
    /usr/sbin/sshd -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password
fi
exec supervisord -n -c /opt/manga-pod/supervisord.conf
