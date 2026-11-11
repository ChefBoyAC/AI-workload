# Phase C: Astra-SIM analytical + MoE/Rail generators (RAM-conscious defaults).
# Build:  docker compose build
# Run:    docker compose run --rm moe-rail-sim

FROM ubuntu:22.04

LABEL project="moe-rail-astra-sim"
LABEL phase="C"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASTRA_SIM_HOME=/opt/astra-sim \
    PATH="/opt/venv/bin:$PATH" \
    GENERATE=1

### ---- system deps (aligned with upstream astra-sim Dockerfile) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    coreutils \
    curl \
    wget \
    git \
    gcc \
    g++ \
    make \
    cmake \
    zlib1g-dev \
    libboost-dev \
    libboost-program-options-dev \
    openmpi-bin \
    openmpi-doc \
    libopenmpi-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install "PyYAML>=6.0" "protobuf>=4.25,<6"

### ---- Abseil (protobuf dependency) ----
ARG ABSL_VER=20240722.0
WORKDIR /opt
RUN curl -fsSL --retry 5 --retry-all-errors -o abseil-cpp-${ABSL_VER}.tar.gz \
      https://github.com/abseil/abseil-cpp/releases/download/${ABSL_VER}/abseil-cpp-${ABSL_VER}.tar.gz \
    && tar -xf abseil-cpp-${ABSL_VER}.tar.gz \
    && rm abseil-cpp-${ABSL_VER}.tar.gz
WORKDIR /opt/abseil-cpp-${ABSL_VER}/build
RUN cmake .. \
      -DCMAKE_CXX_STANDARD=14 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="/opt/abseil-cpp-${ABSL_VER}/install" \
    && cmake --build . --target install --config Release --parallel 2
ENV absl_DIR="/opt/abseil-cpp-${ABSL_VER}/install"

### ---- Protobuf ----
# 29.0 matches upstream astra-sim Dockerfile; curl+retries avoids flaky GitHub asset redirects.
ARG PROTOBUF_VER=29.0
WORKDIR /opt
RUN curl -fsSL --retry 5 --retry-all-errors -o protobuf-${PROTOBUF_VER}.tar.gz \
      https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VER}/protobuf-${PROTOBUF_VER}.tar.gz \
    && tar -xf protobuf-${PROTOBUF_VER}.tar.gz \
    && rm protobuf-${PROTOBUF_VER}.tar.gz
WORKDIR /opt/protobuf-${PROTOBUF_VER}/build
RUN cmake .. \
      -DCMAKE_CXX_STANDARD=14 \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_ABSL_PROVIDER=package \
      -DCMAKE_INSTALL_PREFIX="/opt/protobuf-${PROTOBUF_VER}/install" \
    && cmake --build . --target install --config Release --parallel 2
ENV PATH="/opt/protobuf-${PROTOBUF_VER}/install/bin:$PATH" \
    protobuf_DIR="/opt/protobuf-${PROTOBUF_VER}/install" \
    PROTOBUF_FROM_SOURCE=True
RUN /opt/venv/bin/pip install "protobuf==5.${PROTOBUF_VER}"

### ---- Clone + build Astra-SIM analytical (congestion-unaware) ----
# Pin later via ASTRA_SIM_REF once a smoke build is proven.
ARG ASTRA_SIM_REF=master
WORKDIR /opt
RUN git clone --branch "${ASTRA_SIM_REF}" https://github.com/astra-sim/astra-sim.git ${ASTRA_SIM_HOME} \
    && cd ${ASTRA_SIM_HOME} \
    && git submodule update --init --recursive
WORKDIR ${ASTRA_SIM_HOME}
# Limit parallelism inside the build script environment to reduce peak RAM.
ENV CMAKE_BUILD_PARALLEL_LEVEL=2
RUN ./build/astra_analytical/build.sh \
    && ls -la build/astra_analytical/build/bin \
    && test -x build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
       -o -x build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware

### ---- Build Astra-SIM ns-3 backend (matrix default for fair scheduler compare) ----
# Cap -j to 2: full nproc OOMs easily on Docker Desktop during ns-3 compile.
RUN sed -i 's/-j $(nproc)/-j 2/g' ./build/astra_ns3/build.sh \
    && ./build/astra_ns3/build.sh \
    && ls -la extern/network_backend/ns-3/build/scratch/ \
    && test -n "$(ls extern/network_backend/ns-3/build/scratch/ns3*-AstraSimNetwork* 2>/dev/null)"

### ---- Install Chakra from Astra-SIM submodule (Text → ET converter) ----
WORKDIR ${ASTRA_SIM_HOME}/extern/graph_frontend/chakra
RUN /opt/venv/bin/pip install -e . || /opt/venv/bin/pip install .
# Chakra's checked-in et_def_pb2 was gencoded with protobuf 7.x; runtime must match
# major version or `chakra_converter Text` fails (gencode 7 vs runtime 5).
RUN /opt/venv/bin/pip install "protobuf==7.35.1"

### ---- Project code (matrix kit) ----
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN /opt/venv/bin/pip install -r /workspace/requirements.txt
COPY config /workspace/config
COPY src /workspace/src
COPY workloads /workspace/workloads
COPY topologies /workspace/topologies
COPY backends /workspace/backends
COPY schedulers /workspace/schedulers
COPY experiments /workspace/experiments
COPY shared /workspace/shared
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh
RUN chmod +x /workspace/docker/entrypoint.sh

ENV PYTHONPATH=/workspace \
    WORKLOAD=llm-v1 \
    TOPOLOGY=rail \
    SCHEDULER=nccl \
    BACKEND=ns3 \
    GENERATE=1

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
