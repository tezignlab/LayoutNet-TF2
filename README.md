# LayoutNet in TensorFlow 2.3  

Reimplement of [LayoutNet](https://xtqiao.com/projects/content_aware_layout/) in TensorFLow 2.3.  

**Reference**:  
Zheng, Xinru, et al. "Content-aware generative modeling of graphic design layouts." ACM Transactions on Graphics (TOG) 38.4 (2019): 1-15.

## Introduction  

LayoutNet is a content-aware graphic design layout generation model. It is able to synthesize layout designs based on the visual and textual semantics of user inputs.  

## Example Results  

## LayoutNet on Colab  

We have put demo and the whole training pipeline on Colab, you can click [here](https://colab.research.google.com/drive/1IUkqqyevxW-N8hnXMIpx8JcAotkcTX4O?usp=sharing) to play with it.  

## Prepare Dataset

Download the following files from [here](https://drive.google.com/drive/folders/1-4kCVyJneBnACnaDgUGs8J1MMS6WavQT?usp=sharing). And place them in the repo's folder, the same level as `main.py`.

- `checkpoints` (~200M): this is the pre-trained LayoutGAN model 
- `dataset` (~1G): the processed dataset in TFRecords format
- `sample` (5.2M): the processed sample visual (visfea), text (texfea), semantic (semvec) features and some sample generation results 

The folder structure should be:

```shell
├── README.md
├── checkpoints
│   ├── checkpoint
│   ├── ckpt-300.data-00000-of-00001
│   └── ckpt-300.index
├── config.py
├── dataset
│   └── layout_1205.tfrecords
├── dataset.py
├── demo.py
├── main.py
├── model.py
├── preprocessing.py
├── requirements.txt
└── sample
    ├── imgSel_128.txt
    ├── layout
    ...
```

The path to the datasets are specified in `config.py`:

```python
# configuration for the supervisor
logdir = "./log"
sampledir = "./example"
max_steps = 30000
sample_every_n_steps = 100
summary_every_n_steps = 1
save_model_secs = 120
checkpoint_basename = "layout"
checkpoint_dir = "./checkpoints"
filenamequeue = "./dataset/layout_1205.tfrecords"
min_after_dequeue = 5000
num_threads = 4
```

## Getting Started  

### With Docker (GPU, Linux only)  

> With Docker, only NVIDIA driver is required, no need to configure CUDA and cuDNN.  

- Firstly, install `NVIDIA Container Toolkit` following the [official guidance](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).  

- Build your image using `Dockerfile.gpu`.

```shell
docker build -t layoutnet -f Dockerfile.gpu .
```

- Run `nvidia-smi` in your container to check if everything goes right.  

```shell
docker run --rm --gpus all layout nvidia-smi
```

### With Docker (CPU, Linux / Windows / MacOS)  

- Build image using `Dockerfile`.

```shell
docker build -t layoutnet .
```

- Run your image

```shell
docker run -it --volume `pwd`:/LayoutNet layoutnet /bin/bash
```

### On Local Machine  

> If you want to use GPU on local machine, make sure you have installed right version of CUDA and cuDNN before running the following command. You can check it from [here](https://www.tensorflow.org/install/source#build_the_package).  

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train Model  

```shell
python main.py --train
```

## Test Model  

```shell
python main.py --test
```
