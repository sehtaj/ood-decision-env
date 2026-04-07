FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# install git (needed sometimes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# copy full project
COPY . /app

# install uv (if not present)
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv; \
    fi

# install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# set env
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# IMPORTANT: HuggingFace requires 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]