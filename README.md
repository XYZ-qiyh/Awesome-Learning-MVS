## Awesome-Learning-MVS (Methods and Datasets)

[Update] Looking for PatchMatch-based MVS? 👉 [[Awesome-PatchMatch-MVS](https://github.com/XYZ-qiyh/Awesome-PatchMatch-MVS)]

### Learning-based MVS Methods
1. Volumetric methods (SurfaceNet)
2. Depthmap based methods (MVSNet/R-MVSNet and so on)

#### ICCV2017
  + SurfaceNet: An End-to-end 3D Neural Network for Multiview Stereopsis [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Ji_SurfaceNet_An_End-To-End_ICCV_2017_paper.pdf)] [[Github](https://github.com/mjiUST/SurfaceNet)] [[T-PAMI](https://ieeexplore.ieee.org/document/9099504)]
  + Learning a Multi-View Stereo Machine [[paper](https://papers.nips.cc/paper/2017/file/9c838d2e45b2ad1094d42f4ef36764f6-Paper.pdf)] (LSMs can produce two kinds of outputs - voxel occupancy grids decoded from 3D Grid or per-view depth maps decoded after a projection operation.)
  + Learned Multi-Patch Similarity [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_ICCV_2017/supplemental/Hartmann_Learned_Multi-Patch_Similarity_ICCV_2017_supplemental.pdf)] (Note: Learning to measure multi-image patch similiarity, NOT end-to-end learning MVS pipeline)

#### CVPR2018
+ DeepMVS: Learning Multi-view Stereopsis [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_DeepMVS_Learning_Multi-View_CVPR_2018_paper.pdf)] [[project](https://phuang17.github.io/DeepMVS/index.html)]

#### ECCV2018

+ MVSNet: Depth Inference for Unstructured Multi-view Stereo [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf)] [[supp](https://yoyo000.github.io/papers/yao2018mvsnet_supp.pdf)] [[Github](https://github.com/YoYo000/MVSNet)] [[WeChat article](https://mp.weixin.qq.com/s/AQfpCHyXeRRSFPcRu_mUjA)]

#### CVPR2019

+ Recurrent MVSNet for High-resolution Multi-view Stereo Depth Inference  [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yao_Recurrent_MVSNet_for_High-Resolution_Multi-View_Stereo_Depth_Inference_CVPR_2019_paper.pdf)]  [[supp](https://openaccess.thecvf.com/content_CVPR_2019/supplemental/Yao_Recurrent_MVSNet_for_CVPR_2019_supplemental.pdf)] [[Github](https://github.com/YoYo000/MVSNet)]

#### ICCV2019

+ Point-Based Multi-View Stereo Network  [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Point-Based_Multi-View_Stereo_Network_ICCV_2019_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Chen_Point-Based_Multi-View_Stereo_ICCV_2019_supplemental.pdf)] [[Github](https://github.com/callmeray/PointMVSNet)] [[T-PAMI](https://ieeexplore.ieee.org/abstract/document/9076298)] (Point-MVSNet performs multi-view stereo reconstruction in a *coarse-to-fine* fashion, learning to predict the 3D flow of each point to the groundtruth surface based on geometry priors and 2D image appearance cues)
+ P-MVSNet: Learning Patch-wise Matching Confidence Aggregation for Multi-view Stereo [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_P-MVSNet_Learning_Patch-Wise_Matching_Confidence_Aggregation_for_Multi-View_Stereo_ICCV_2019_paper.pdf)]
+ MVSCRF: Learning Multi-view Stereo with Conditional Random Fields [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xue_MVSCRF_Learning_Multi-View_Stereo_With_Conditional_Random_Fields_ICCV_2019_paper.pdf)]

#### AAAI2020

+ Learning Inverse Depth Regression for Multi-View Stereo with Correlation Cost Volume [[paper](https://arxiv.org/pdf/1912.11746.pdf)] [[Github](https://github.com/GhiXu/CIDER)]

#### CVPR2020

+ Cascade Cost Volume for High-Resolutoin Multi-View Stereo and Stereo Matching [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gu_Cascade_Cost_Volume_for_High-Resolution_Multi-View_Stereo_and_Stereo_Matching_CVPR_2020_paper.pdf)] [[Github](https://github.com/alibaba/cascade-stereo)] [[WeChat article](https://mp.weixin.qq.com/s/C8Gpg4eCqFnFWK6Wnu7lCw)]
+ Deep Stereo using Adaptive Thin Volume Representation with Uncertainty Awareness [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Deep_Stereo_Using_Adaptive_Thin_Volume_Representation_With_Uncertainty_Awareness_CVPR_2020_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Cheng_Deep_Stereo_Using_CVPR_2020_supplemental.pdf)] [[Github](https://github.com/touristCheng/UCSNet)]

+ Cost Volume Pyramid Based Depth Inference for Multi-View Stereo [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Cost_Volume_Pyramid_Based_Depth_Inference_for_Multi-View_Stereo_CVPR_2020_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yang_Cost_Volume_Pyramid_CVPR_2020_supplemental.pdf)] [[Github](https://github.com/JiayuYANG/CVP-MVSNet)]

+ Fast-MVSNet: Sparse-to-Dense Multi-View Stereo with Learned Propagation and Gauss-Newton Refinement [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Fast-MVSNet_Sparse-to-Dense_Multi-View_Stereo_With_Learned_Propagation_and_Gauss-Newton_Refinement_CVPR_2020_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yu_Fast-MVSNet_Sparse-to-Dense_Multi-View_CVPR_2020_supplemental.pdf)] [[Github](https://github.com/svip-lab/FastMVSNet)] [[WeChat article](https://mp.weixin.qq.com/s/ePvgrIhYtYTR1sQWUn0XhQ)]

+ Attention-Aware Multi-View Stereo [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_Attention-Aware_Multi-View_Stereo_CVPR_2020_paper.pdf)]

+ A Novel Recurrent Encoder-Decoder Structure for Large-Scale Multi-view Stereo Reconstruction from An Open Aerial Dataset [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_A_Novel_Recurrent_Encoder-Decoder_Structure_for_Large-Scale_Multi-View_Stereo_Reconstruction_CVPR_2020_paper.pdf)] [[Github](https://github.com/gpcv-liujin/REDNet)] [[data](http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html)]


#### ECCV2020

+ Pyramid Multi-view Stereo Net with Self-adaptive View aggregation [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540732.pdf)] [[Github](https://github.com/yhw-yhw/PVAMVSNet)]
+ Dense Hybird Recurrent Multi-view Stereo Net with Dynamic Consistency Checking [[paper](https://deepai.org/publication/dense-hybrid-recurrent-multi-view-stereo-net-with-dynamic-consistency-checking)] [[Github](https://github.com/yhw-yhw/D2HC-RMVSNet)]


#### BMVC2020
+ Visibility-aware Multi-view Stereo Network [[paper](https://arxiv.org/abs/2008.07928)] [[Github](https://github.com/jzhangbs/Vis-MVSNet)]

#### WACV2021
+ Long-range Attention Network for Multi-View Stereo [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Long-Range_Attention_Network_for_Multi-View_Stereo_WACV_2021_paper.pdf)]

#### CVPR2021
+ PatchmatchNet: Learned Multi-View Patchmatch Stereo [[paper](https://arxiv.org/pdf/2012.01411.pdf)] [[Github](https://github.com/FangjinhuaWang/PatchmatchNet)]

#### ICCV2021
+ AA-RMVSNet: Adaptive Aggregation Recurrent Multi-view Stereo Network [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wei_AA-RMVSNet_Adaptive_Aggregation_Recurrent_Multi-View_Stereo_Network_ICCV_2021_paper.pdf)] [[supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Wei_AA-RMVSNet_Adaptive_Aggregation_ICCV_2021_supplemental.pdf)] [[Github](https://github.com/QT-Zhu/AA-RMVSNet)]
+ EPP-MVSNet: Epipolar-Assembling Based Depth Prediction for Multi-View Stereo [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_EPP-MVSNet_Epipolar-Assembling_Based_Depth_Prediction_for_Multi-View_Stereo_ICCV_2021_paper.pdf)] [[Github](https://github.com/Ascend/mindspore/tree/master/model_zoo/research/cv/eppmvsnet)]
+ Just a Few Points are All You Need for Multi-view Stereo: A Novel Semi-supervised Learning Method for Multi-view Stereo [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Just_a_Few_Points_Are_All_You_Need_for_Multi-View_ICCV_2021_paper.pdf)] [[supp](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Kim_Just_a_Few_ICCV_2021_supplemental.pdf)]

#### 3DV 2021
+ Deep Multi-View Stereo gone wild. [[paper](https://arxiv.org/abs/2104.15119v2)]  [[Project](https://imagine.enpc.fr/~darmonf/wild_deep_mvs/)] [[Github](https://github.com/fdarmon/wild_deep_mvs)]

#### CVPR 2022
+ Generalized Binary Search Network for Highly-Efficient Multi-View Stereo [[paper](https://arxiv.org/abs/2112.02338)] [[Github](https://github.com/MiZhenxing/GBi-Net)]
+ Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation and Focal Loss [[paper](https://arxiv.org/abs/2201.01501)] [[Github](https://github.com/prstrive/UniMVSNet)]

#### Journal Paper

+ MVSNet++: Learning Depth-Based Attention Pyramid Features for Multi-View Stereo. IEEE TIP [[paper](https://ieeexplore.ieee.org/document/9115828)]
+ HighRes-MVSNet: A Fast Multi-View Stereo Network for Dense 3D Reconstruction From High-Resolution Images. IEEE Access [[paper](https://ieeexplore.ieee.org/document/9319163)]
+ AACVP-MVSNet: Attention-aware cost volume pyramid based multi-view stereo network for 3D reconstruction. ISPRS Journal of Photogrammetry and Remote Sensing [[paper](https://www.sciencedirect.com/science/article/pii/S0924271621000794)] [[Github](https://github.com/ArthasMil/AACVP-MVSNet)]

#### Survey Paper
+ A Survey on Deep Learning Techniques for Stereo-based Depth Estimation. IEEE T-PAMI [[ArXiv](https://arxiv.org/abs/2006.02535)] [[IEEE Xplore](https://ieeexplore.ieee.org/document/9233988)]
+ Deep Learning for Multi-view Stereo via Plane Sweep: A Survey [[paper](https://arxiv.org/abs/2106.15328)]
+ Multi-view stereo in the Deep Learning Era: A comprehensive revfiew [[paper](https://www.sciencedirect.com/science/article/pii/S0141938221001062)]

#### ArXiv Paper
+ PVSNet: Pixelwise Visibility-Aware Multi-View Stereo Network [[paper](https://arxiv.org/abs/2007.07714)]
+ DDR-Net: Learning Multi-Stage Multi-View Stereo With Dynamic Depth Range [[paper](https://arxiv.org/abs/2103.14275)]  [[Github](https://github.com/Tangshengku/DDR-Net)]
+ Non-local Recurrent Regularization Networks for Multi-view Stereo [[paper](https://arxiv.org/abs/2110.06436)]
+ TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers [[paper](https://arxiv.org/abs/2111.14600)]  [[Github](https://github.com/MegviiRobot/TransMVSNet)]


#### To Be Continued...


### Multi-view Stereo Benchmark
<!--  
+ **Middlebury** [CVPR06']
  + A Comparison and Evaluation of Multi-View Stereo Reconstruction Algorithms [[paper](https://vision.middlebury.edu/mview/seitz_mview_cvpr06.pdf)] [[website](https://vision.middlebury.edu/mview/)]

+ **EPFL** [CVPR08']
  + On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery [[paper](https://infoscience.epfl.ch/record/126393)]
[Strecha] [Fountain]
-->

+ **DTU** [CVPR2014, IJCV2016]
  + Large-scale data for multiple-view stereopsis [paper: [CVPR2014](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf), [IJCV2016](https://link.springer.com/content/pdf/10.1007/s11263-016-0902-9.pdf)] [[website](http://roboimagedata.compute.dtu.dk/?page_id=36)] [[Eval code](https://github.com/Todd-Qi/MVSNet-PyTorch/tree/master/evaluations/dtu)] [[video](https://www.bilibili.com/video/BV1k5411G7NA/)]

+ **Tanks and Temples** [ACM ToG2017]
  + Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction  [[paper](https://docs.google.com/uc?export=download&id=0B-ePgl6HF260bGJkdFBCemRLZGM)] [[supp](https://docs.google.com/uc?export=download&id=0B-ePgl6HF260MGhQX0dCcmdHbFk)] [[website](https://www.tanksandtemples.org/)] [[Github](https://github.com/intel-isl/TanksAndTemples)] [[leaderboard](https://www.tanksandtemples.org/leaderboard/)]

+ **ETH3D** [CVPR2017]
  + A Multi-View Stereo Benchmark with High-Resolution Images and Multi-Camera Videos [[paper](https://www.eth3d.net/data/schoeps2017cvpr.pdf)] [[supp](https://www.eth3d.net/data/schoeps2017cvpr-supp.pdf)] [[website](https://www.eth3d.net/)] [[Github](https://github.com/ETH3D)]

+ **BlendedMVS** [CVPR2020]
  + BlendedMVS: A Large-Scale Dataset for Generalized Multi-View Stereo Network [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yao_BlendedMVS_A_Large-Scale_Dataset_for_Generalized_Multi-View_Stereo_Networks_CVPR_2020_paper.pdf)] [[supp](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Yao_BlendedMVS_A_Large-Scale_CVPR_2020_supplemental.pdf)] [[Github](https://github.com/YoYo000/BlendedMVS)] [[visual](https://github.com/kwea123/BlendedMVS_scenes)] [[WeChat article](https://mp.weixin.qq.com/s/PZ-G8W-cg5oSVLd_cfX92A)]

+ **GigaMVS** [T-PAMI2021]
  + GigaMVS: A Benchmark for Ultra-large-scale Gigapixel-level 3D Reconstruction [[paper](https://ieeexplore.ieee.org/document/9547729)] [[website](http://www.gigamvs.com/)]

+ **Multi-sensor large-scale dataset for multi-view 3D reconstruction** [ArXiv2022]
  + Multi-sensor large-scale dataset for multi-view 3D reconstruction [[paper](https://arxiv.org/pdf/2203.06111.pdf)] [[website](http://adase.group/3ddl/projects/sk3d)]


### Large-scale Real-world Scenes
1. Chinese Style Architectures
  + http://vision.ia.ac.cn/zh/data/index.html, provided by CASIA.

2. Western Style Architectures
  + https://colmap.github.io/datasets.html, provided by COLMAP.
  + [ImageDataset_SceauxCastle](https://github.com/openMVG/ImageDataset_SceauxCastle), provided by OpenMVG.

3. Aerial Dataset
  + http://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html, provided by WHU.

Welcome to contribute to this Repo!

<!--
Multi-view stereo for face recon
[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bai_Deep_Facial_Non-Rigid_Multi-View_Stereo_CVPR_2020_paper.pdf) && [Github](https://github.com/zqbai-jeremy/DFNRMVS)
--->

### Update: WeChat Group
<img src="/photo/WeChat_Group.jpg" width=270 height=585>
