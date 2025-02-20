# Toward Robust Non-Transferable Learning: A Survey and Benchmark

[![arXiv](https://img.shields.io/badge/Arxiv-2502.13593-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.13593)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. 

We bridge this gap by presenting the first comprehensive survey on NTL and introducing **NTLBench**, the first benchmark to evaluate NTL performance and robustness within a unified framework. 

![](figs/abstract.png)

This figure shows the comparison of 5 methods (<font color="#7884AC">NTL</font>, <font color="#7884AC">CUTI-domain</font>, <font color="#7884AC">H-NTL</font>, <font color="#7884AC">SOPHON</font>, <font color="#7884AC">CUPI-domain</font>) on CIFAR \& STL with VGG-13, where we evaluate non-transferability performance and post-training robustness against 5 <font color="#E4785F">source domain fine-tuning (SourceFT)</font> attacks, 4 <font color="#8151BA">target domain fine-tuning (TargetFT)</font> attacks, and 6 <font color="#59A4B7">source-free domain adaptation (SFDA)</font> attacks (higher value means better performance/robustness). 


## Survey

**NTL** (ICLR 2022): [Non-Transferable Learning: A New Approach for Model Ownership Verification and Applicability Authorization](https://arxiv.org/pdf/2106.06916) \
**UNTL** (EMNLP 2022): [Unsupervised Non-transferable Text Classification](https://arxiv.org/pdf/2210.12651) \
**CUTI-domain** (CVPR 2023): [Model barrier: A compact un-transferable isolation domain for model intellectual property protection](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.pdf) \
**DSO** (ICCV 2023): [Domain Specified Optimization for Deployment Authorization](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Domain_Specified_Optimization_for_Deployment_Authorization_ICCV_2023_paper.pdf) \
**H-NTL** (ICLR 2024): [Improving non-transferable representation learning by harnessing content and style](https://openreview.net/pdf?id=FYKVPOHCpE) \
**ArchLock** (ICLR 2024): [ArchLock: Locking DNN Transferability at the Architecture Level with a Zero-Cost Binary Predictor](https://openreview.net/pdf?id=e2YOVTenU9) \
**TransNTL** (CVPR 2024): [Your Transferability Barrier is Fragile: Free-Lunch for Transferring the Non-Transferable Learning](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_Your_Transferability_Barrier_is_Fragile_Free-Lunch_for_Transferring_the_Non-Transferable_CVPR_2024_paper.pdf) \
**MAP** (CVPR 2024): [MAP: MAsk-Pruning for Source-Free Model Intellectual Property Protection](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_MAP_MAsk-Pruning_for_Source-Free_Model_Intellectual_Property_Protection_CVPR_2024_paper.pdf) \
**SOPHON** (IEEE S&P 2024): [SOPHON: Non-Fine-Tunable Learning to Restrain Task Transferability For Pre-trained Models](https://arxiv.org/pdf/2404.12699) \
**CUPI-domain** (TPAMI 2024): [Say No to Freeloader: Protecting Intellectual Property of Your Deep Model](https://arxiv.org/pdf/2408.13161)
**NTP** (ECCV 2024): [Non-transferable Pruning](https://arxiv.org/pdf/2410.08015)

![](figs/review.png)


## NTLBench

We include following methods:
- [[Paper](https://arxiv.org/pdf/2106.06916)][[Code](https://github.com/conditionWang/NTL)] `NTL` (ICLR 2022) 
- [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.pdf)][[Code](https://github.com/LyWang12/CUTI-Domain)] `CUTI-domain` (CVPR 2023) 
- [[Paper](https://openreview.net/pdf?id=FYKVPOHCpE)][[Code](https://github.com/tmllab/NTLBench)] `HNTL` (ICLR 2024) 
- [[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_Your_Transferability_Barrier_is_Fragile_Free-Lunch_for_Transferring_the_Non-Transferable_CVPR_2024_paper.pdf)][[Code](https://github.com/tmllab/2024_CVPR_TransNTL)] `TransNTL` (CVPR 2024) 
- [[Paper](https://arxiv.org/pdf/2404.12699)][[Code](https://github.com/ChiangE/Sophon)] `SOPHON` (IEEE S&P 2024)
- [[Paper](https://arxiv.org/pdf/2408.13161)][[Code](https://github.com/LyWang12/CUPI-Domain)] `CUPI-domain` (T-PAMI 2024)

<!-- We propose the ﬁrst NTL benchmark (NTLBench), which contains a standard and uniﬁed training and evaluation process. NTLBench supports 5 SOTA NTL methods, 9 datasets (more than 116 domain pairs), 5 network architectures families, and 15 post-training attacks from 3 attack settings, providing more than 40,000 experimental conﬁgurations. -->

```
NTLBench will be released soon.
```


## Citation
If you find this work useful in your research, please consider citing our paper:
```
@article{hong2025toward,
  title={Toward Robust Non-Transferable Learning: A Survey and Benchmark},
  author={Ziming Hong and Yongli Xiang and Tongliang Liu},
  journal={arXiv preprint arXiv:2502.13593},
  year={2025}
}
```

## Acknowledgement
Parts of this project were inspired by the following projects. We thank their contributors for their excellent work: [NTL](https://github.com/conditionWang/NTL), [CUTI-domain](https://github.com/LyWang12/CUTI-Domain), [SOPHON](https://github.com/ChiangE/Sophon), [CUPI-domain](https://github.com/LyWang12/CUPI-Domain), [TransNTL](https://github.com/tmllab/2024_CVPR_TransNTL), [SFDA](https://github.com/tntek/source-free-domain-adaptation), [DomainBed](https://github.com/facebookresearch/DomainBed)