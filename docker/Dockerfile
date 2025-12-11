FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ARG USERNAME
ARG USER_UID
ARG USER_GID=$USER_UID
ARG USER_GNAMES
ARG USER_GADD_ARGS
ARG PIP_REQ_FILE
ARG APT_REQ_FILE

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    build-essential ca-certificates python3 python3-dev python3-distutils git vim wget cmake python3-pip libgl1 libglib2.0-0

USER 0
RUN IFS=";" read -a myarr <<< "$USER_GADD_ARGS" && \
    unset 'myarr[${#arr[@]}-1]' && \
    for i in ${!myarr[@]}; do \
        if [[ ${myarr[$i]} != "sudo"* ]]; then \
            groupadd -f -g ${myarr[$i]}; \
        fi \
    done
RUN  useradd -u $USER_UID -g $USER_GID -m $USERNAME
RUN for g in $USER_GNAMES; do \
        usermod -a -G $g $USERNAME; \
    done
# Allow sudo
RUN     apt-get install -y sudo \
        && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
        && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

RUN python3 -m pip install --upgrade pip

USER 0
RUN mkdir -p /.local && chmod -R 777 /.local
RUN mkdir -p /.avalanche && chmod -R 777 /.avalanche
RUN mkdir -p /.config && chmod -R 777 /.config
RUN touch /.netrc && chmod 777 /.netrc
RUN mkdir -p /.cache && chmod -R 777 /.cache
USER $USERNAME

USER 0
ADD $APT_REQ_FILE /apt_requirements.txt
RUN apt-get update && cat /apt_requirements.txt | xargs apt-get install -y
USER $USERNAME

ADD $PIP_REQ_FILE /requirements.txt
RUN pip install -r /requirements.txt

RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

CMD ["bash"]
WORKDIR /home/$USERNAME/exp

# Build via:
# ./build.sh -- -t nasiri/llamagen:latest .
# Run via:
# docker run --rm -it --gpus "device=0" --shm-size=8g \
#     -v $(pwd):/app \
#     -v $(pwd)/../checkpoints:/checkpoints \
#     -v /home/nasiri/datasets/shared/imagenet:/datasets/imagenet \
#     -w /app \
#     nasiri/llamagen:latest \
#     python3 scripts/reconstruct_imagenet.py
# eval via (change folder path accordingly):
#     docker run --rm -it --gpus "device=0" --shm-size=8g \
#   -v /home/nasiri/storage/vq-gan/taming-transformers:/ref \
#   -v /home/nasiri/storage/llamagen/LlamaGen:/llamagen \
#   -w /ref/eval \
#   nasiri/evaluator:latest \
#   python /ref/eval/evaluator.py \
#   /llamagen/imagenet_val_gt_256.npz \
#   /llamagen/reconstructions/VQ-16-imagenet-size-256-size-256-codebook-size-16384-dim-8-seed-0-ts-1765387998.npz
