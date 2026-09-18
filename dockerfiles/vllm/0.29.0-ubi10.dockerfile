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
# Extra repositories
########################################

# Needed for ffmpeg, which RHEL does not ship (RPM Fusion supplies it) and which
# pulls build dependencies from EPEL. ffmpeg is not optional here: vLLM 0.29.0
# lists torchcodec in requirements/xpu.txt, and torchcodec links against the
# FFmpeg shared libraries at import time.
#
# epel-release is not in the UBI repositories, so it is installed from its URL.
# CodeReady Builder is called "crb" on RHEL/Rocky but
# "ubi-10-codeready-builder-rpms" on UBI, and may already be enabled; enabling
# it is therefore best-effort.
RUN set -xe && \
    dnf install -y "https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm" && \
    { dnf config-manager setopt crb.enabled=1 || \
      dnf config-manager setopt ubi-10-codeready-builder-rpms.enabled=1 || \
      dnf config-manager --set-enabled crb || true; } && \
    dnf install -y "https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm" && \
    dnf clean all && \
    rm -rf /var/cache/dnf

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
RUN dnf install -y \
        ffmpeg \
        libsndfile \
        libSM \
        libXext \
        mesa-libGL \
        libaio-devel \
        numactl && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# This Intel(R) oneAPI Collective Communications Library (oneCCL) contains several enhancements for Intel(R) Arc(TM) Pro graphics
# For details, please refer to https://github.com/uxlfoundation/oneCCL/releases/tag/2021.15.9
ARG ONECCL_INSTALLER="intel-oneccl-2021.15.9.14_offline.sh"
ARG ONECCL_INSTALLER_SHA256="f7ab81b6ed1b10dd35fadec366a78046d8af214888dfd625047ce8953d5aa4ef"
RUN wget --progress=dot:giga "https://github.com/uxlfoundation/oneCCL/releases/download/2021.15.9/${ONECCL_INSTALLER}" && \
    printf "%s  %s\n" "${ONECCL_INSTALLER_SHA256}" "${ONECCL_INSTALLER}" > /tmp/oneccl.sha256 && \
    sha256sum -c /tmp/oneccl.sha256 && \
    rm -f /tmp/oneccl.sha256 && \
    bash "${ONECCL_INSTALLER}" -a --silent --eula accept && \
    rm "${ONECCL_INSTALLER}" && \
    echo "source /opt/intel/oneapi/setvars.sh --force" >> /root/.bashrc && \
    echo "source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force" >> /root/.bashrc && \
    rm -f /opt/intel/oneapi/ccl/latest && \
    ln -s /opt/intel/oneapi/ccl/2021.15 /opt/intel/oneapi/ccl/latest

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
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v && \
    # remove PyTorch bundled oneCCL to avoid conflicts with the oneCCL installed above
    pip uninstall -y oneccl oneccl-devel

ENV VLLM_TARGET_DEVICE=xpu
# XPU workers must not fork; spawn is required for multi-GPU tensor parallelism.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# No ENTRYPOINT here on purpose: the OMIX base image already declares
#   ENTRYPOINT ["/bin/bash", "-c", "source /opt/intel/oneapi/setvars.sh --force && exec \"$@\"", "--"]
# so the oneAPI environment is sourced for whatever command is passed to
# "docker run". 0.29.0-rockylinux10.dockerfile has to declare it itself.
