# 🔧 Stereo Toolbox

A comprehensive stereo matching toolbox for efficient development and research.

## 📦 Installation

```
pip install stereo_toolbox
```

## 🔄 Dataloader

| Status | Identifier | Scale | Description |
| :----: | ---------- | ----- | ----------- | 
| ✅ | [SceneFlow_Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | 35K+ | The most famous synthetic dataset for stereo matching pre-training. |
| ✅ | [KITTI2015_Dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | 200+ | Driving scene dataset. |
| ✅ | [KITTI2012_Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) | 192 | Driving scene dataset. |
| ✅ | [MiddleburyEval3_Dataset](https://vision.middlebury.edu/stereo/submit3) | 15 | Indoor and outdoor scene dataset. |
| ✅ | [ETH3D_Dataset](https://www.eth3d.net/datasets) | 27 | Indoor scene dataset with grayscale images. |
| ✅ | [DrivingStereo_Dataset](https://drivingstereo-dataset.github.io/)| 174K+ | Driving scene dataset with diverse weathers (sunny, cloudy, foggy, rainy). |
| ✅ | [Middlebury2021_Dataset](https://vision.middlebury.edu/stereo/data/scenes2021/) | 24 | Indoor scene dataset. The non-occulusion masks are obtained using LRC by [StereoAnywhere](https://github.com/bartn8/stereoanywhere). |
| ✅ | [Sintel_Dataset](http://sintel.is.tue.mpg.de/stereo) | 1K+ | A synthetic dataset derived from the open source 3D animated short film, Sintel.|
| ✅ | [HR_VS_Dataset](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view) | 780 | A synthetic dataset derived from Carla. |
| ❌ | [Booster_Dataset](https://amsacta.unibo.it/id/eprint/6876/) |  |  |
| ❌ | [CREStereo_Dataset](https://github.com/megvii-research/CREStereo) | |
| ❌ | [TartanAir_Dataset]() | |
| ❌ | [FallingThings_Dataset]() | |
| ❌ | [InStereo2k_Dataset](https://github.com/YuhuaXu/StereoDataset) | 2K | |
| ❌ | [Argoverse_Dataset]() | |



## 🧠 Model Backbones

| Status | Identifier | Architecture | Description |
| :----: | ---------- | ------------ | ----------- |
| ❌ | [PSMNet]() | 3D Conv. | CVPR 2018 |
| ❌ | [GwcNet]() | 3D Conv. | CVPR 2019 |
| ❌ | [AANet]() | 2D Conv. | CVPR 2020 |
| ❌ | [CFNet]() | 3D Conv. | CVPR 2021 |
| ❌ | [RaftStereo]() | Iterative | 3DV 2021 |
| ❌ | [PCWNet]() | 3D Conv. | ECCV 2022 |
| ❌ | [STTR]() | Transformer | ICCV 2021 |
| ❌ | [CREStereo]() | Iterative | CVPR 2022 |
| ❌ | [IGEVStereo]() | Iterative | CVPR 2023 |
| ❌ | [Selective-IGEVStereo]() | Iterative | CVPR 2024 |
| ❌ | [MoChaStereo]() | Iterative | CVPR 2024 |
| ❌ | [NMRF]() | MRF | CVPR 2024 |




## 📉 Loss Functions
| Status | Identifier | Description |
| :----: | ---------- | ----------- |
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
| ❌ | generalization_eval | Test generalization performance on the training sets of KITTI 2015/2012, Middlebury Eval3, and ETH3D. |
| ❌ | sceneflow_test | Evaluation on SceneFlow test set. |
| ❌ | kitti2015_sub | Generate data with KITTI 2015 submission format. |
| ❌ | kitti2012_sub |  Generate data with KITTI 2012 submission format. |
| ❌ | middeval3_sub |  Generate data with Middlebury Eval3 submission format. |
| ❌ | eth3d_sub |  Generate data with ETH3D submission format. |