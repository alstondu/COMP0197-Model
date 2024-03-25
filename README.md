# mixup-vit-img-cls

This repo contains personal implementation and evalutation of vision-transfromer-based image classification on CIFAR-10 Dataset, using mixup algorithm (two sampling method) for data augmentation.

------
# Pre-requisites
+ python=3.12 
+ pytorch=2.2 
+ torchvision=0.17

------
# Overview

### Task 1: Stochastic Minibatch Gradient Descent for Linear Models
* Implemented a least square solver for fitting the polynomial functions

* Implemented implement a stochastic minibatch gradient descent algorithm for fitting the polynomial functions

#### Task 1a
* Implemented an SGD for for fitting the polynomial functions with degree of polynomial as a learnable parameter

### Task 2: Data-augmented vision transformers
* Performed image classification based on visual transformer on CIFAR-10 Dataset. Used mixup algorithm (two sampling method) for data augmentation

### Task 3: Ablation Study
* Conducted Ablation study to investigate the impact of the mixup sampling method



For detailed tasks descriotion and instructions, please refer to [Instruction](https://github.com/alstondu/mixup-vit-img-cls/blob/master/Task%20Description.pdf).

The trained model can be found [here](https://drive.google.com/drive/folders/1pNqsunV3p02Ih1bI8mSEEagxhJi3JaSU?usp=sharing).
