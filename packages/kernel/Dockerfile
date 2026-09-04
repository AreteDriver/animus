# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install build dependencies for packages with native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install the unpublished sibling package before building Kernel.
COPY packages/types/ /build/types/
RUN pip install --no-cache-dir /build/types

# Copy Kernel source from the monorepo build context.
COPY packages/kernel/ /build/kernel/
WORKDIR /build/kernel

# Build and install the wheel
RUN pip install --no-cache-dir build \
    && python -m build . \
    && pip install --no-cache-dir dist/*.whl

# Clean up build artifacts
RUN rm -rf /build

WORKDIR /app

# Verify installation
RUN python -c "import animus_kernel; print(animus_kernel.__version__)"

# Healthcheck: ensure the package imports cleanly
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import animus_kernel"

CMD ["python", "-m", "animus_kernel"]
