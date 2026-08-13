# active-slam-rl container
#
# Builds a self-contained environment with everything needed to run the
# pipeline: Python + all requirements.txt packages, plus build-essential
# and cmake so you can also compile the native FS2D library (native/fs2d/)
# inside the container once you have the real Constructor University
# source (see native/fs2d/README.md) -- you don't need it to run anything
# in this image, the NumPy Fourier-Mellin fallback works out of the box.
#
# Build:
#   docker build -t active-slam-rl .
#
# Run the test suite:
#   docker run --rm active-slam-rl pytest tests/ -v
#
# Train (writes into ./results on your host via the bind mount -- see
# docker-compose.yml for the easier way to do this):
#   docker run --rm -v "$(pwd)/results:/app/results" active-slam-rl \
#       python scripts/run_training.py --config configs/quick_demo.yaml
#
# See the "Docker" section of README.md for the full walkthrough.

FROM python:3.11-slim

# build-essential + cmake: only needed if/when you compile native/fs2d's
# real C/C++ library inside the container. Everything else (training,
# eval, visualization, tests) runs fine without them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first so Docker can cache the (slow) pip
# install layer separately from the (fast-changing) source code.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project and install it in editable mode, so
# `import active_slam_rl` works from anywhere without PYTHONPATH hacks.
COPY . .
RUN pip install --no-cache-dir -e .

# Where results/ lives -- mount a host volume here (see docker-compose.yml)
# so trained models, logs, plots, and GIFs persist outside the container.
VOLUME ["/app/results"]

# No default long-running process -- this is a CLI toolbox, not a service.
# `docker run active-slam-rl <command>` overrides this; bare `docker run
# active-slam-rl` runs the test suite as a quick self-check.
CMD ["pytest", "tests/", "-v"]
