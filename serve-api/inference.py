from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import torch
import numpy as np
def inference_model(model, img):
    """Inference image(s) with the classifier.
    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        results = []
        for i in range(5):
            pred_score = np.max(scores, axis=1)[0]
            pred_label = np.argmax(scores, axis=1)[0]
            result = {'pred_label': int(pred_label), 'pred_score': float(pred_score)}
            result['pred_class'] = model.CLASSES[result['pred_label']]
            results.append(result)
            scores[0][pred_label] = -999999
    return results