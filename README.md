# Image-Memorability-Phase-2

## Requirements
* Python 3.7
* Pytorch 1.4.0
* Other basic computing modules

## Instructions
1. `output` directory contains the trained encoding model for 8 subjects in the NSD dataset.
2. `encoding.py` is called when loading the encoding model to NeuroGen.
3. `getROImask.py` is used to get the ROI mask for the 24 used ROIs. 
4. `getmaskedROI.py` is used to get the voxel response within certain ROI.
5. `getmaskedROImean.py` is used to get the mean voxel response within certain ROI.
6. `neurogen.py` is the main script for NeuroGen, and can be called by

`python neurogen.py --roi 1 --steps 1000 --gpu 0 --lr 0.01 --subj 1 --reptime 1 --truncation 1`

7. `visualize.py` contains some useful functions to save images and visualize them.
8. `pytorch_pretrained_biggan` is available here: https://github.com/huggingface/pytorch-pretrained-BigGAN

Note: `getROImask.py`, `getmaskedROI.py` and `getmaskedROImean.py` deal with the NSD data which has not been released yet and are not necessary to run NeuroGen at this time. Paths in all scripts may need to change according to needs.

## Acknowledgment
We sincerely thank the authors of following open-source projects:

NeuroGen: activation optimized image synthesis for discovery neuroscience:
https://doi.org/10.1016/j.neuroimage.2021.118812

Controlling Memorability of Face Images:
https://doi.org/10.48550/arXiv.2202.11896
