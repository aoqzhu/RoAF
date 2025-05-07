# Complementarity and Adaptive Fusion for Robust Multimodal Sentiment Analysis


1.Anonymous submitted to NeurIPS 2025.
![The framework of RoAF](imgs/framework.jpg)

## Data Preparation
MOSI/MOSEI/CH-SIMS Download: Please see [MMSA](https://github.com/thuiar/MMSA)

## Environment
The basic training environment for the results in the paper is Pytorch  1.8.2, Python 3.9.19 with NVIDIA Tesla A40. 

## Training
You can quickly run the code with the following command:
```
bash train.sh
```

## Evaluation
After the training is completed, the checkpoints corresponding to the three random seeds (1111,1112,1113) can be used for evaluation. For example, evaluate the the model's binary classification accuracy in MOSI:
```
CUDA_VISIBLE_DEVICES=0 python robust_evaluation.py --config_file configs/eval_mosi.yaml --key_eval Has0_acc_2
```

