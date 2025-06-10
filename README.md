# ViSt3D: Video Stylization with 3D CNN
<div align="center">

[![Conference](https://img.shields.io/badge/NeurIPS%202023-000000)](https://openreview.net/pdf?id=2EiqizElGO)&nbsp;
[![Project page](https://img.shields.io/badge/Project%20page-ViSt3D-pink)](https://ayush202.github.io/projects/ViSt3D.html)&nbsp;

</div>

## Overview
This repository contains the dataset information and the inference code used in the paper:

*ViSt3D: Video Stylization with 3D CNN*

Ayush Pande, Gaurav Sharma

NeurIPS 2023

https://github.com/ayush202/ViSt3D/assets/16152273/9939a190-1d95-4b23-8432-3098335a1b35

https://github.com/ayush202/ViSt3D/assets/16152273/bdab7e87-e03d-4c6b-96a8-3c529d40c500

## Updates
[2024-07-28] Inference code and checkpoints added

[2024-03-05] Results content videos and style images uploaded. Please see data folder

[2024-01-03] Dataset Uploaded. Please see motionclips_dataset.txt

## Prerequisites
* Linux or macOS
* Python 3
* graphics card with memory >= 24 GB
* RAM with memory >= 90 GB

## Getting Started

* Clone this repository

```shell
git clone https://github.com/ayush202/ViSt3D.git
cd ViSt3D
```
* Install the requirements by running the below command:

``` shell
      pip install -r requirements.txt
```

## Inference

* Download pretrained model from [Google Drive](https://drive.google.com/file/d/1izz7PiDEhiYwB-RA0Zki85RCNrt_M7qY/view?usp=sharing), and unzip:

```shell
unzip checkpoints.zip
rm checkpoints.zip
```
* Configure content_video and style paths for desired content video and the style image in test_ViSt3D.sh.

* Then, simply run:

```shell
bash test_ViSt3D.sh
```
* Check the results under **output/** folder.


## File Description
* motionclips_dataset: Each entry is of the form clip_<*clip_no*>,<*start_frame_no*>,<*youtube_url*>. Each clip consists of 16 frames. *start_frame_no* tells the frame number in the video where the clip starts. 

## Citation
If you find ideas useful for your research work, please cite:

```
@inproceedings{pande2023vist3d,
  title={ViSt3D: Video Stylization with 3D CNN},
  author={Pande, Ayush and Sharma, Gaurav},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Acknowledgements
* The dataset is curated based on the repository of **[sports-1m-dataset](https://github.com/gtoderici/sports-1m-dataset/tree/master)** by George Toderici *et al.*
* The pretrained C3D model is taken from the repository of **[c3d-pytorch](https://github.com/DavideA/c3d-pytorch/tree/master)** by Davide Abati
