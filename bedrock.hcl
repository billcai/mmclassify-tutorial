version = "1.0"

train {
  step "train" {
    image = "pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel"
    install = [
      "apt-get update && apt-get install -y tmux wget git tree",
      "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv",
      "pip install bdrk==0.9.1",
      "pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html",
      "git clone https://github.com/open-mmlab/mmclassification.git && cd mmclassification && pip install -e . & cd .."
    ]
    script = [{sh = ["python task_train.py"]}]
    resources {
      cpu = "2"
      memory = "14G"
      gpu = "1"
    }
  }

  parameters {
    BUCKET_NAME = "basisai-samples"
    DATA_DIR = "shellfish"
    EXECUTION_DATE = "2020-10-01"
    NUM_EPOCHS = "10"
    BATCH_SIZE = "16"
  }
}