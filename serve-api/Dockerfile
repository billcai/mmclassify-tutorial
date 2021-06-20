FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y tmux wget git tree
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv
RUN pip install mmcv
RUN git clone https://github.com/open-mmlab/mmclassification.git && \
    cd mmclassification && \
    pip install -e .
RUN wget -O model.pth https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth
RUN pip install uvicorn[standard] fastapi python-multipart
COPY api.py .
COPY inference.py .

ENTRYPOINT uvicorn api:app --host 0.0.0.0 --reload