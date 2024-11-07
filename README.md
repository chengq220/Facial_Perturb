Code for the paper "Leaving Some Facial Features Behind"
## Description
The proposed training approach involves learning to cluster features. Using attention mechanisms, the image is segmented into various localized features, which are then used to train a classifier on this newly extracted information.

![network](images/model.png)

The following tables shows the performance of different models for facial emotion recognition.

![network](images/performance.png)

### Get Started
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install additional dependencies by using `pip install -r requirements.txt`

### Data
1. Navigate to `/data` and run `bash getDataset.sh` to download and extract dataset

### Training
```python train.py```

### Inference
Navigate to ```predict.py``` and input the path to the desired image
