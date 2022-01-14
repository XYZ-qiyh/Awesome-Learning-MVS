## Unsupervised Learning MVS

#### CVPRW 2019
+ Learning Unsupervised Multi-View Stereopsis via Robust Photometric Consistency [[Project](https://tejaskhot.github.io/unsup_mvs/)] [[paper](https://tejaskhot.github.io/unsup_mvs/)] [[Github](https://github.com/tejaskhot/unsup_mvs)] (Central idea: a warping-based view synthesis loss, use the estimated depth map for image synthesis.)

#### 3DV 2019
+ MVS2: Deep Unsupervised Multi-view Stereo with Multi-View Symmetry [[paper](https://ieeexplore.ieee.org/document/8885975)]

#### ICIP 2021
+ M3VSNet: Unsupervised multi-metric multi-view stereo network [[paper](https://ieeexplore.ieee.org/abstract/document/9506469)] [[Github](https://github.com/whubaichuan/M3VSNet)]

#### AAAI 2021
+ Self-supervised Multi-view Stereo via Effective Co-Segmentation and Data-Augmentation [[paper](https://www.aaai.org/AAAI21Papers/AAAI-2549.XuH.pdf)] [[Github](https://github.com/ToughStoneX/Self-Supervised-MVS)]

#### CVPR 2021
+ Self-supervised Learning of Depth Inference for Multi-view Stereo [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Self-Supervised_Learning_of_Depth_Inference_for_Multi-View_Stereo_CVPR_2021_paper.pdf)] [[supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Yang_Self-Supervised_Learning_of_CVPR_2021_supplemental.pdf)] [[Github](https://github.com/JiayuYANG/Self-supervised-CVP-MVSNet)]

#### ICCV2021
+ Digging into Uncertainty in Self-supervised Multi-view Stereo [[Github](https://github.com/ToughStoneX/U-MVS)] [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Digging_Into_Uncertainty_in_Self-Supervised_Multi-View_Stereo_ICCV_2021_paper.pdf)] [[supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Xu_Digging_Into_Uncertainty_ICCV_2021_supplemental.pdf)]

#### 3DV 2021
+ Deep Multi-View Stereo gone wild, ArXiv 2104.15119 [[Project](https://imagine.enpc.fr/~darmonf/wild_deep_mvs/)] [[Github](https://github.com/fdarmon/wild_deep_mvs)]

#### Blog
Unsupervised Multi-View Stereo — An Emerging Trend [[Link](https://medium.com/analytics-vidhya/unsupervised-multi-view-stereo-an-emerging-trend-4d3034e23e9e)]



<!--

### Semi-Supervised Methods
+ A Novel Semi-supervised Learning Method for Multi-view Stereo



#### Weakly-supervised stereo matching

+ Unsupervised Adaption using *Confidence Guided Loss*
+ Semi-supervised stereo matching: sparse Lidar and photometric consistency
+ Unsupervised Learning of Stereo Matching: in an iterative manner using Left-Right consistency Check

-->



#### Benchmark Performance

|    Methods    | Acc. ↓  | Comp. ↓ | Overall ↓ | TnT@f-score ↑ |
| :-----------: | :---: | :---: | :-----: | :---------: |
|   Unsup_MVS   | 0.881 | 1.073 |  0.977  |      —      |
|     MVS^2     | 0.760 | 0.515 |  0.637  |    37.21    |
|    M3VSNet    | 0.636 | 0.531 |  0.583  |    37.67    |
|     JDACS     | 0.571 | 0.515 |  0.543  |    45.48    |
|     U-MVS     | 0.470 | 0.430 |  0.450  |      —      |
| MVSNet (Sup.) | 0.396 | 0.527 |  0.462  |    43.48    |
| COLMAP(Geo.)  | 0.401 | 0.661 |  0.531  |    42.14    |

