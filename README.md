# Codes for my M.Tech (Research) thesis

## Note
> The implementations are hard-coded. I will be uploading a clean version of my thesis codes soon. Thanks! And, meanwhile, if you have any question, please reach out to me here or on Twitter (@dbp_patel_1994).


## _BARE_
> Contains codes corresponding to the paper [_Adaptive Sample Selection for Robust Learning under Label Noise_](https://arxiv.org/abs/2106.15292) which corresponds to [Ch-3 of my master's thesis](https://dbp1994.github.io/files/deep-patel-iisc-masters-thesis_compressed.pdf).

## memorization_and_overparam
> Contains codes corresponding to the paper [_Memorization in Deep Neural Networks: Does the Loss Function Matter?_](https://link.springer.com/chapter/10.1007/978-3-030-75765-6_11) which corresponds to [Ch-4 of my master's thesis](https://dbp1994.github.io/files/deep-patel-iisc-masters-thesis_compressed.pdf).


## Things to be done:
- [X] Create a file for the conda environment details [**_conda_env_details.txt_**)]
- [ ] Create a file containing hyperparameter details for all the experiments [**_hyperparam_details.md_**] [[desired format](https://github.com/HanxunH/Active-Passive-Losses/blob/master/configs/cifar10/sym/gce.yaml)]
- [ ] Upload all the baseline codes in **_BARE_** folder
- [X] Upload codes for my algorithms
- [X] Upload codes pertaining to experiments on memorization and overparameterization
- [] _Modularize_ your codes (along the lines of [this](https://github.com/hrayrhar/limit-label-memorization/releases/tag/v0.1)/[this](https://github.com/hrayrhar/limit-label-memorization) PyTorch-based repository)

## Instructions to run experiments:
- To be upated.

## Relevant links for the baseline algorithms:

### Robust Loss Functions
- Generalized Cross-Entropy (NeurIPS'18) [[paper](https://arxiv.org/abs/1805.07836)]
- Symmetric Cross Entropy Loss (ICCV'19) [[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html)]
<!--- 
- Normalised Loss Functions (ICML'20) [[paper](https://arxiv.org/abs/2006.13554)] [[official code](https://github.com/HanxunH/Active-Passive-Losses/)]
-->

### Loss Correction Based

- Forward Loss Correction [[paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html)]
<!--- - Meta Loss-Correction (CVPR'20) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Training_Noise-Robust_Deep_Neural_Networks_via_Meta-Learning_CVPR_2020_paper.pdf)] [[official code](https://github.com/ZhenWang-PhD/Training-Noise-Robust-Deep-Neural-Networks-via-Meta-Learning)]
-->

### Regularization Based
- Dimensionality Driven Learning (ICML'18) [[paper](http://proceedings.mlr.press/v80/ma18d)]
- Meta-Ren (ICML'18) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_Probabilistic_End-To-End_Noise_Correction_for_Learning_With_Noisy_Labels_CVPR_2019_paper.pdf)] [[official code](https://github.com/uber-research/learning-to-reweight-examples)] [[unofficial code](https://github.com/danieltan07/learning-to-reweight-examples)] [[unofficial code](https://github.com/tanyuqian/learning-data-manipulation)]
- Unsupervised Label Noise Modelling & Loss Correction [[paper](http://proceedings.mlr.press/v97/arazo19a.html)]
- Meta Net (NeurIPS'19) [[paper](https://papers.nips.cc/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)] [[official code](https://github.com/xjtushujun/meta-weight-net)]
- DivideMix (ICLR'20) [[paper](https://openreview.net/forum?id=HJgExaVtwr)] [[official code](https://github.com/LiJunnan1992/DivideMix)]
- JoCoR (CVPR'20) [[paper](https://arxiv.org/pdf/2003.02752.pdf)] [[official code](https://github.com/hongxin001/JoCoR)]
<!---
- Meta MLNT (CVPR'19) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_to_Learn_From_Noisy_Labeled_Data_CVPR_2019_paper.pdf)] [[official code](https://github.com/LiJunnan1992/MLNT)]
- LIMIT (ICML'20) [[paper](https://arxiv.org/abs/2002.07933)] [[official code](https://github.com/hrayrhar/limit-label-memorization)]
-->

### Label-cleaning Based
- Joint Optimization (CVPR'18) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Tanaka_Joint_Optimization_Framework_CVPR_2018_paper.html)]
- 
<!--- 
- SELFIE (ICML'19) [[paper](http://proceedings.mlr.press/v97/song19b/song19b.pdf)] [[official code](https://github.com/kaist-dmlab/SELFIE)]
- PENCIL (CVPR'19) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_Probabilistic_End-To-End_Noise_Correction_for_Learning_With_Noisy_Labels_CVPR_2019_paper.pdf)] [[official code](https://github.com/yikun2019/PENCIL)] [[unofficial code](https://github.com/JacobPfau/PENCIL)] [[unofficial code](https://github.com/ljmiao/PENCIL)]
-->

