# 🔧 Stereo Toolbox

A comprehensive stereo matching toolbox for efficient development and research.


## 📦 Installation

```
pip install stereo_toolbox
```


## 🔄 Datasets

| Status | Identifier | Train | Val | Test | Noc. Mask | Description |
| :----: | ---------- | :---: | :-: | :--: | :----------------: | ----------- | 
| ✅ | [SceneFlow_Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | 35K+ | 4.3K+ | - | ❌ | The most famous synthetic dataset for stereo matching pre-training. |
| ✅ | [KITTI2015_Dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | 200 | - | 200 | ✅ | Driving scene dataset. |
| ✅ | [KITTI2012_Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) | 192 | - | 195 | ✅ | Driving scene dataset. |
| ✅ | [MiddleburyEval3_Dataset](https://vision.middlebury.edu/stereo/submit3) | 15 | - | 15 | ✅ | Indoor and outdoor scene dataset. |
| ✅ | [ETH3D_Dataset](https://www.eth3d.net/datasets) | 27 | - | 20 | ✅ | Indoor scene dataset with grayscale images. |
| ✅ | [DrivingStereo_Dataset](https://drivingstereo-dataset.github.io/)| 174K+ | 7.7K+ | - | ❌ | Driving scene dataset with diverse weathers (sunny, cloudy, foggy, rainy). |
| ✅ | [Middlebury2021_Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/) | 24 | -  | - | ✅ | Indoor scene dataset. The non-occulusion masks are obtained using LRC by [StereoAnywhere](https://github.com/bartn8/stereoanywhere). |
| ✅ | [Sintel_Dataset](http://sintel.is.tue.mpg.de/stereo) | 1.0K+ | - | - | ✅ | Synthetic dataset derived from the open source 3D animated short film, Sintel.|
| ✅ | [HR_VS_Dataset](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view) | 780 | - | - | ❌ | Synthetic dataset rendered by Carla simulator. |
| ✅ | [Booster_Dataset](https://amsacta.unibo.it/id/eprint/6876/) | 228 | - | - | ✅ | Indoor dataset with specular and transparent surfaces. |
| ✅ | [CREStereo_Dataset](https://github.com/megvii-research/CREStereo) | 200K | - | - | ❌ | Synthetic dataset rendered by Blender with different shapes, lighting, texture, and smooth disparity distribution. |
| ❌ | [TartanAir_Dataset]() | | |
| ❌ | [FallingThings_Dataset](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation) | | |
| ✅ | [InStereo2k_Dataset](https://github.com/YuhuaXu/StereoDataset) | 2.0K+ | 50 | - | ❌ | Indoor dataset with high accuracy disparity maps. |
| ❌ | [Argoverse_Dataset](https://www.argoverse.org/av1.html#stereo-link) | 4.0K+ | 1.5K+ | 1.0K+ | ❌ | Driving scene dataset with details at the near and far range. |
| ✅ | [MonoTrap_Dataset](https://github.com/bartn8/stereoanywhere) | 0 | 26 | 0 | ❌ | Perspective illusion dataset specifically designed to challenge monocular depth estimation. |
| ✅ | [Holopix50k_Dataset](https://github.com/LeiaInc/holopix50k) |  41K+ | 4.9K+ | 2.4K+ | ❌ | In-the-wild Dataset contributed by users of the Holopix™ mobile social platform. |
| ❌ | [LayeredFlow](https://layeredflow.cs.princeton.edu) |


## 🧠 Models

| Status | Identifier | Architecture | Description |
| :----: | ---------- | ------------ | ----------- |
| ✅ | [PSMNet](https://github.com/JiaRenChang/PSMNet) | 3D Conv. | CVPR 2018 |
| ❌ | [GwcNet](https://github.com/xy-guo/GwcNet) | 3D Conv. | CVPR 2019 |
| ❌ | [AANet](https://github.com/haofeixu/aanet) | 2D Conv. | CVPR 2020 |
| ❌ | [CFNet](https://github.com/gallenszl/CFNet) | 3D Conv. | CVPR 2021 |
| ❌ | [RaftStereo](https://github.com/princeton-vl/RAFT-Stereo) | Iterative | 3DV 2021 |
| ❌ | [PCWNet](https://github.com/gallenszl/PCWNet) | 3D Conv. | ECCV 2022 |
| ❌ | [STTR](https://github.com/mli0603/stereo-transformer) | Transformer | ICCV 2021 |
| ❌ | [CREStereo](https://github.com/megvii-research/CREStereo) | Iterative | CVPR 2022 |
| ❌ | [IGEVStereo](https://github.com/gangweix/IGEV) | Iterative | CVPR 2023 |
| ❌ | [Selective-IGEVStereo](https://github.com/Windsrain/Selective-Stereo) | Iterative | CVPR 2024 |
| ❌ | [MoChaStereo](https://github.com/ZYangChen/MoCha-Stereo) | Iterative | CVPR 2024 |
| ❌ | [NMRF](https://github.com/aeolusguan/NMRF) | MRF | CVPR 2024 |


## 📉 Loss Functions
| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ❌ | L1_loss | |
| ❌ | smooth_l1_loss | |
| ❌ | photometric_loss | |
| ❌ | edge_aware_smoothness_loss | |
| ❌ | single_modal_cross_entropy_loss | |
| ❌ | multi_modal_cross_entropy_loss | |


## 🎨 Visualization

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | colored_disparity_map_Spectral_r | Disparity map pseudocolor visualization with Spectral_r colorbar. |
| ✅ | colored_dispairty_map_KITTI | Disparity map pseudocolor visualization with KITTI colorbar. |
| ❌ | colored_error_map_KITTI | Error map pseudocolor visualization with KITTI colorbar. |
| ❌ | colored_pointcloud | Point cloud visualization with real color derived from left image. |


## 📊 Evaluation

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | generalization_eval | Test generalization performance on the training sets of KITTI 2015/2012, Middlebury Eval3, and ETH3D. Outliers in the occ, noc, and all regions are reported.|
| ✅ | sceneflow_test | Evaluation on SceneFlow test set. EPE and outliers are reported.|
| ❌ | kitti2015_sub | Generate dispairty maps in KITTI 2015 submission format. |
| ❌ | kitti2012_sub |  Generate dispairty maps in KITTI 2012 submission format. |
| ❌ | middeval3_sub |  Generate dispairty maps in Middlebury Eval3 submission format. |
| ❌ | eth3d_sub |  Generate dispairty maps in ETH3D submission format. |


## 📏 Disparity Estimators

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
