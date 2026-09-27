#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
weights_dir="$repo_root/model_weights"
weights_file="$weights_dir/nomad.pth"
image_tag="${IMAGE_TAG:-visualnav_transformer:latest}"
ros_distro="${ROS_DISTRO:-jazzy}"
torch_variant="${TORCH_VARIANT:-nvidia}"

mkdir -p "$weights_dir"
if [[ ! -s "$weights_file" ]]; then
    echo "Downloading NoMaD weights to $weights_file"
    docker run --rm \
        --env "HOST_UID=$(id -u)" --env "HOST_GID=$(id -g)" \
        --mount "type=bind,source=$weights_dir,target=/weights" \
        python:3.12-slim sh -ec '
            python -m pip install --quiet gdown
            temporary=$(mktemp /weights/.nomad.XXXXXX)
            trap '\''rm -f "$temporary"'\'' EXIT
            gdown "https://drive.google.com/uc?id=1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb" -O "$temporary"
            test -s "$temporary"
            mv "$temporary" /weights/nomad.pth
            chown "$HOST_UID:$HOST_GID" /weights/nomad.pth
        '
fi

docker build \
    --build-arg "ROS_DISTRO=$ros_distro" \
    --build-arg "TORCH_VARIANT=$torch_variant" \
    -t "$image_tag" "$repo_root"
