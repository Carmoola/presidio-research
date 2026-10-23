FROM ghcr.io/data-privacy-stack/presidio-analyzer:latest
USER root
RUN pip install --no-cache-dir transformers torch \
    --extra-index-url https://download.pytorch.org/whl/cpu
USER presidio
