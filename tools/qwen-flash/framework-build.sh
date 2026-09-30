#!/usr/bin/env bash
# Build the fork with the machine's ROCm 10 stack, without host ROCm packages.
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
image=${GUFO_TOOLCHAIN_IMAGE:-gufo-framework-toolchain:rocm10}
if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build --target toolchain -t "$image" -f "$root/deploy/framework/Dockerfile" "$root/deploy/framework"
fi
revision=$(git -C "$root" rev-parse HEAD)
if [[ -n $(git -C "$root" status --porcelain) ]]; then
    revision+=-dirty
fi
docker run --rm --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$root,dst=/source" -w /source \
    -e "GUFO_REVISION=$revision" -e "BUILD_JOBS=${BUILD_JOBS:-4}" \
    "$image" bash -euc '
        cmake --preset release -DCMAKE_CXX_FLAGS="-include format" \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGUFO_REVISION="$GUFO_REVISION"
        cmake --build --preset release --parallel "$BUILD_JOBS" --target gufo
    '
