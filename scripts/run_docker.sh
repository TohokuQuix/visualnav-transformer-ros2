#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${IMAGE_TAG:-visualnav_transformer:latest}"
torch_variant="${TORCH_VARIANT:-nvidia}"

docker_args=(
    run -it --rm --net=host
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-}"
    --mount "type=bind,source=$repo_root,target=/visualnav-transformer"
)

case "$torch_variant" in
    nvidia) docker_args+=(--gpus=all) ;;
    amd) docker_args+=(--device=/dev/kfd --device=/dev/dri) ;;
    cpu) ;;
    *) echo "Invalid TORCH_VARIANT '$torch_variant' (expected: nvidia, amd or cpu)" >&2; exit 1 ;;
esac

exec docker "${docker_args[@]}" "$@" "$image_tag"
