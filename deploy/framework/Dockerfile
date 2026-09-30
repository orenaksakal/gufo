# Toolchain for this Framework Desktop's existing ROCm 10 runtime.
FROM registry.fedoraproject.org/fedora:44@sha256:8938dce2600de0b78f5ef8d1541192f207fdafb7414d83957f6687147aa8998b AS toolchain

RUN <<'EOF'
cat > /etc/yum.repos.d/rocm.repo <<'REPO'
[amdrocm-stable]
name=ROCm 10.0.0
baseurl=https://stable.repo.amd.com/rocm/core/packages/rhel10/x86_64
enabled=1
priority=50
gpgcheck=1
gpgkey=https://stable.repo.amd.com/rocm/gpg/packages.gpg
REPO
EOF

RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
      gcc gcc-c++ cmake ninja-build pkgconf-pkg-config git-core \
      libicu-devel libcurl-devel openssl-devel libpng-devel libjpeg-turbo-devel \
      amdrocm-core-devel10.0-gfx1151 python3 clang-tools-extra \
    && dnf clean all

ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/core/lib/rocm_sysdeps/lib:/opt/rocm/core/lib
WORKDIR /source
