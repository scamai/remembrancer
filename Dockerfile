# One image for Remembrancer: the pipeline as a command, and the MCP server
# over stdio (one person) or streamable HTTP (a team, behind your own proxy).
# compose.yaml mounts the memory and your config.json; see the README.
#
# v0.1 reads memory next to the code, so each data folder under /app is a
# volume mount point: events/, context/, state/ and registry/.
FROM python:3.12-slim

# safe.directory: a repository you mount for ingest belongs to your user, not
# to the image's; without it git refuses to read it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

WORKDIR /app
COPY requirements.txt .
# pip runs as root only while the image builds, and installs into the image's
# own Python, so its warning about root and its notice of a newer pip do not
# apply to the person building it.
RUN pip install --no-cache-dir --root-user-action=ignore --disable-pip-version-check \
    -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 app \
 && mkdir -p events context state registry \
 && chown app:app events context state registry

USER app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
VOLUME ["/app/events", "/app/context", "/app/state", "/app/registry"]
EXPOSE 8790
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["help"]
