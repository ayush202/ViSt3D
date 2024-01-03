# ViSt3D: Video Stylization with 3D CNN

## Overview
This repository contains the dataset information used in the paper:

*ViSt3D: Video Stylization with 3D CNN*

Ayush Pande, Gaurav Sharma

NeurIPS 2023

![](our_results/basic_results1.mp4)

![](our_results/basic_results2.mp4)

![](our_results/basic_results3.mp4)

## Updates
[2024-01-03] Dataset Uploaded. Please see motionclips_dataset.txt

## File Description
* motionclips_dataset: Each entry is of the form clip_<*clip_no*>,<*start_frame_no*>,<*youtube_url*>. Each clip consists of 16 frames. *start_frame_no* tells the frame number in the video where the clip starts. 

## Citation
*If you find ideas useful for your research work, please cite:*

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