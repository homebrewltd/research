FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS clang18_image

# Install dependencies
RUN apt-get -qq update; \
    apt-get install -qqy --no-install-recommends \
        gnupg2 wget ca-certificates apt-transport-https \
        autoconf automake cmake dpkg-dev file make patch libc6-dev

# Install LLVM
#RUN echo "deb https://apt.llvm.org/jammy llvm-toolchain-jammy-18 main" \
#        > /etc/apt/sources.list.d/llvm.list && \
#    wget -qO /etc/apt/trusted.gpg.d/llvm.asc \
#        https://apt.llvm.org/llvm-snapshot.gpg.key && \
#    apt-get update && \
#    apt-get install -y -t llvm-toolchain-jammy-18 clang-18 clangd-18 clang-tidy-18 clang-format-18 lld-18 libc++-18-dev libc++abi-18-dev && \
#    for f in /usr/lib/llvm-18/bin/*; do ln -sf "$f" /usr/bin; done && \
#    rm -rf /var/lib/apt/lists/*

FROM clang18_image AS base_image

# Add python PPA
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa -y

# Install packages
RUN apt-get update && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3.11 \
    #python3-pip \
    python3.11-dev \
    zsh \
    nvtop \
    btop \
    tmux \
    git \
    curl \
    ca-certificates \
    gettext \
    unzip \
    fd-find \
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

# Set clang as the default compiler
#RUN ln -sf /usr/bin/clang /usr/bin/cc \
#  && ln -sf /usr/bin/clang++ /usr/bin/c++ \
#  && cc --version \
#  && c++ --version

# Update the alternatives for Python 3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100

# Fix cuda clang issue
# Command to append content to the clangd config file
RUN mkdir -p /root/.config/clangd && \
    echo "CompileFlags:" >> /root/.config/clangd/config.yaml && \
    echo "  Add:" >> /root/.config/clangd/config.yaml && \
    echo "    - --cuda-gpu-arch=sm_89" >> /root/.config/clangd/config.yaml && \
    echo "  Remove:" >> /root/.config/clangd/config.yaml && \
    echo "    - --generate-code=arch=*" >> /root/.config/clangd/config.yaml && \
    echo "    - -forward-unknown-to-host-compiler" >> /root/.config/clangd/config.yaml

ENV HOME=/home/runner

RUN mkdir -p /home/runner

ARG RUNNER_VERSION=2.317.0

ARG RUNNER_UID=1000
ARG DOCKER_GID=1001

RUN adduser --disabled-password --gecos "" --uid $RUNNER_UID runner \
    && groupadd docker --gid $DOCKER_GID \
    && usermod -aG sudo runner \
    && usermod -aG docker runner \
    && echo "%sudo   ALL=(ALL:ALL) NOPASSWD:ALL" > /etc/sudoers \
    && echo "Defaults env_keep += \"DEBIAN_FRONTEND\"" >> /etc/sudoers

# cd into the user directory, download and unzip the github actions runner
RUN cd /home/runner && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

RUN chown -R runner:runner /home/runner && /home/runner/actions-runner/bin/installdependencies.sh

ADD ./start.sh /home/runner/start.sh

RUN chmod +x /home/runner/start.sh

# Add /usr/local/cuda-11.7/compat to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ENTRYPOINT ["/bin/bash", "/home/runner/start.sh"]

USER runner