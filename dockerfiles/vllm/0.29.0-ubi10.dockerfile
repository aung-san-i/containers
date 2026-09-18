# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Red Hat UBI 10 based Intel(R) Open Middleware Xe (OMIX) + vLLM image.
#
# Keeps the layering used by the Ubuntu images, because an OMIX image is
# published for UBI 10:
#   compute-runtime:*-devel-ubi10.2 -> omix:0.4.0-devel-ubi10 -> this file
# Compare with 0.29.0-rockylinux10.dockerfile, which has to install the GPU
# stack itself because no Rocky Linux OMIX base image exists.
#
# Upstream vLLM is built unpatched. intel/llm-scaler has no omix-vllm patch set
# for 0.29.0 (that lineage stops at omix-vllm-0.21.0), and upstream XPU support
# no longer needs it: requirements/xpu.txt pins triton==3.7.2+xpu,
# torch==2.13.0 and vllm_xpu_kernels itself, from https://wheels.vllm.ai/xpu/.
# For the patched build see 0.21.0-llmscaler-ubi10.dockerfile.
#
# Build (from the repository root):
#   DOCKER_BUILDKIT=1 docker build -t intel/vllm:0.29.0-ubi10 \
#     -f dockerfiles/vllm/0.29.0-ubi10.dockerfile .
#
# Serve:
#   docker run --rm -it --device /dev/dri -p 8000:8000 \
#     intel/vllm:0.29.0-ubi10 \
#     vllm serve <model> --host 0.0.0.0 --port 8000

ARG BASE_IMAGE=intel/omix:0.4.0-devel-ubi10
FROM ${BASE_IMAGE}

LABEL image.vllm.version=0.29.0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

########################################
# Install Python and create a virtual environment
########################################

# RHEL derivatives ship venv as part of python3-libs; there is no python3-venv
# package as on Debian/Ubuntu.
RUN dnf install -y \
        python3 \
        python3-devel \
        python3-pip && \
    python3 --version && \
    dnf clean all && \
    rm -rf /var/cache/dnf && \
    mkdir -p /opt && \
    python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip setuptools wheel

########################################
# Install dependencies
########################################

# RPM equivalents of the Debian packages used by 0.21.0-ubuntu24.04.dockerfile:
#   libsndfile1 -> libsndfile   libsm6   -> libSM
#   libxext6    -> libXext      libgl1   -> mesa-libGL
#   libaio-dev  -> libaio-devel
# lsb-release is intentionally omitted: RHEL 10 no longer ships LSB packages.
# git and wget already come from the compute-runtime base of this image.
# No FFmpeg is installed: vLLM 0.29.0 defaults its video backend to opencv, whose
# wheel bundles FFmpeg. UBI 10 ships no FFmpeg at all, and RPM Fusion's EL10 build
# is unsatisfiable there (needs libSDL2, libvpx, snappy, speex, libvdpau, ... none
# of which exist in the UBI repo set). That is also why no extra repositories
# (EPEL / CRB / RPM Fusion) are enabled in this file.
RUN dnf install -y \
        libsndfile \
        libSM \
        libXext \
        mesa-libGL \
        libaio-devel \
        numactl && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# No manual oneCCL install here. torch 2.13.0+xpu pins oneccl==2022.0.0 (built for
# oneAPI 2026, libsycl.so.9) and links libccl.so.1 directly. Installing oneCCL
# 2021.15.9 and then removing the bundled one -- as 0.21.0-llmscaler-* does -- breaks
# torch import: that build needs libsycl.so.8, which no longer exists in this stack.

########################################
# Install vLLM
########################################

ENV VLLM_VERSION=0.29.0

RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm

WORKDIR /opt/vllm

# requirements/xpu.txt carries its own --extra-index-url entries
# (https://wheels.vllm.ai/xpu/ and https://download.pytorch.org/whl/xpu), so the
# triton / torch / vllm_xpu_kernels pins do not need to be repeated or corrected
# here as they did for 0.21.0.
RUN --mount=type=cache,target=/root/.cache/pip \
    git checkout v${VLLM_VERSION} && \
    pip install -v -r requirements/xpu.txt && \
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v

ENV VLLM_TARGET_DEVICE=xpu
# XPU workers must not fork; spawn is required for multi-GPU tensor parallelism.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# No ENTRYPOINT here on purpose: the OMIX base image already declares
#   ENTRYPOINT ["/bin/bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec \"$@\"", "--"]
# so the oneAPI environment is sourced for whatever command is passed to
# "docker run". 0.29.0-rockylinux10.dockerfile has to declare it itself.
