"""
Script to train model.
"""
import os
import shutil
import time

import bdrk
import wget
import tarfile
import mmcv
from mmcv import Config, DictAction
from mmcls.apis import set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import collect_env, get_root_logger
import os.path as osp
import time
import copy
from bedrock_client.bedrock.api import BedrockApi

def download_data():
    url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
    if 'dataset.tar.gz' not in os.listdir(os.getcwd()):
        wget.download(url, 'dataset.tar.gz')
    if 'food-101' not in os.listdir(os.getcwd()):
        fname = 'dataset.tar.gz'
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()
    cur_dir = os.getcwd()
    image_dir = os.path.join(cur_dir,'food-101','images')
    CLASSES = {}
    classes_file = os.path.join(cur_dir,'food-101','meta','classes.txt')
    with open(classes_file) as f:
        for count,x in enumerate(f.readlines()):
            CLASSES[x[:-1]] = count
    test_labels = os.path.join(cur_dir,'food-101','meta','test.txt')
    test_dir = os.path.join(cur_dir,'food-101','test')
    os.makedirs(test_dir,exist_ok=True)
    test_samples = []
    new_test_labels = os.path.join(cur_dir,'food-101','meta','test_imagenet.txt')
    with open(test_labels) as f:
        for x in f.readlines():
            image_location = os.path.join(image_dir,x[:-1]+'.jpg')
            new_location = os.path.join(test_dir,x[:-1]+'.jpg')
            os.makedirs(os.path.dirname(new_location),exist_ok=True)
            if os.path.exists(image_location):
            shutil.move(
                image_location,
                new_location
            )
            test_samples.append(
                new_location+' '+str(CLASSES[x.split('/')[0]])
            )
    with open(new_test_labels,'w') as f:
        f.write('\n'.join(test_samples) + '\n')
    return image_dir

def setup_and_train_model(image_dir):
    cfg = Config.fromfile(
    'mmclassification/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py')
    new_test_labels = os.path.join(cur_dir,'food-101','meta','test_imagenet.txt')
    cfg.data.train.data_prefix = image_dir
    cfg.data.val.data_prefix = '/'
    cfg.data.val.ann_file = new_test_labels
    cfg.data.test.data_prefix = '/'
    cfg.data.test.ann_file = new_test_labels
    cfg.data.samples_per_gpu = 384
    cfg.model.head.num_classes = 101
    cfg.work_dir = 'food-101-workdir'
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, 'shufflenet_food101.config'))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    wget.download("ttps://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth")
    cfg.resume_from = os.path.join(
        os.getcwd(),
        'shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth'
    )
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    logger.info(f'Config:\n{cfg.pretty_text}')
    model = build_classifier(cfg.model)
    model.init_weights()
    datasets = [build_dataset(cfg.data.train)]
    val_dataset = copy.deepcopy(cfg.data.val)
    val_dataset.pipeline = cfg.data.train.pipeline
    datasets.append(build_dataset(val_dataset))
    cfg.gpu_ids=range(1)
    cfg.seed = None
    meta['seed'] = None
    cfg.lr_config = dict(policy='step', gamma=0.98, step=1)
    # the epochs of the pretrained model is at 300, so we only train for
    # 10 additional epochs (310 = 300+10)
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=310)
    cfg.workflow = [('train', 1),('val',1)]
    train_model(
        model,
        datasets,
        cfg,
        distributed=False,
        validate=True,
        timestamp=timestamp,
        device='cuda',
        meta=meta)
    return cfg.work_dir

def main():
    """Train"""
    print("\nPyTorch Version:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device found = {device}")

    if device.type == "cuda":
        print("  Number of GPUs:", torch.cuda.device_count())
        print("  Device properties:", torch.cuda.get_device_properties(0))

    print("\nDownload data")
    start = time.time()
    image_dir = download_data()
    print(f"  Time taken = {time.time() - start:.0f} secs")

    print("\nTrain model")
    work_dir = setup_and_train_model(image_dir)

    print("\nSave artefacts and results")
    shutil.copy2(work_dir+"latest.pth","/artefact/")


if __name__ == "__main__":
    bdrk.init()
    with bdrk.start_run():
        main()