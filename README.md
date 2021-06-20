# Tutorial for using mmclassification

## Outline
This is a tutorial for using mmclassification to:
- train an image classification model on your own custom dataset
- run inference using your trained model
- serve the model using a plain vanilla FastAPI wrapper (you should really look at [TorchServe](https://github.com/pytorch/serve) for production workloads, mmclassification has out of the box support for it.)

## Prerequisites
To run the training and inference notebooks, you need a Google account to access Colab.
1. When in Google Colab, click on  `File` > `Open Notebook` and select Github as the source. 
2. For Training, put in `https://github.com/billcai/mmclassify-tutorial/blob/master/food_model_classifier.ipynb` as the link.
3. For Inference, put in `https://github.com/billcai/mmclassify-tutorial/blob/master/food_model_inference.ipynb` as the link.
Load the notebooks and they should be good to go.

For serving the model, we have a Dockerfile, which you can build using
```
cd serve-api
docker build . -t mmclassify-api:latest
```
This requires you to have a GPU-enabled device that has compute ability 7.5 and above (basically a GTX 1060 and above roughly).
Then, assuming that you have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, you can start serving the model with
```
docker run --name mmclassify-api -p 8000:8000 --gpus all mmclassify-api:latest
```
You would be able to see the OpenAPI docs and test your API at [http://localhost:8000/docs](http://localhost:8000/docs).