FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y tmux wget git tree
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
RUN pip install jupyter 
RUN git clone https://github.com/open-mmlab/mmclassification.git && \
    cd mmclassification && \
    pip install -e .