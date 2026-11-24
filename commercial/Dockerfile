FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && useradd -m -u 10001 manga
COPY . /srv
RUN mkdir -p /srv/data /srv/staticfiles && chown -R manga:manga /srv
USER manga
EXPOSE 8010
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8010", "--workers", "2", "--timeout", "45", "--access-logfile", "-"]
