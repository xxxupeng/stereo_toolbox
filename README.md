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
| ✅ | [InStereo2k_Dataset](https://github.com/YuhuaXu/StereoDataset) | 2.0K+ | 50 | - | ❌ | Indoor dataset with high accuracy disparity maps. |
| ✅ | [Argoverse_Dataset](https://www.argoverse.org/av1.html#stereo-link) | 4.0K+ | 1.5K+ | 1.0K+ | ❌ | Driving scene dataset with details at the near and far range. |
| ✅ | [MonoTrap_Dataset](https://github.com/bartn8/stereoanywhere) | - | 26 | - | ❌ | Perspective illusion dataset specifically designed to challenge monocular depth estimation. |
| ✅ | [Holopix50k_Dataset](https://github.com/LeiaInc/holopix50k) |  41K+ | 4.9K+ | 2.4K+ | ❌ | In-the-wild Dataset contributed by users of the Holopix™ mobile social platform. |
| ❌ | [LayeredFlow](https://layeredflow.cs.princeton.edu) |
| ❌ | [TartanAir_Dataset]() | | |
| ❌ | [FallingThings_Dataset](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation) | | |

**Dataloader Return:**
- left image (color jitter if training)
- right image (color jitter and random mask if training)
- disparity ground-truth (nan if not available)
- non-occucluded mask (nan if not available)
- raw left image (not normalized)
- raw right image (not normalized)


## 🧠 Models

| Status | Identifier | Architecture | Description |
| :----: | ---------- | ------------ | ----------- |
| ✅ | [PSMNet](https://github.com/JiaRenChang/PSMNet) | 3D Conv. | CVPR 2018 |
| ✅ | [GwcNet](https://github.com/xy-guo/GwcNet) | 3D Conv. | CVPR 2019 |
| ❌ | [GANet](https://github.com/feihuzhang/GANet) | 3D Conv. | CVPR 2019, need to compile |
| ❌ | [AANet](https://github.com/haofeixu/aanet) | 2D Conv. | CVPR 2020, need to compile |
| ❌ | [DSMNet](https://github.com/feihuzhang/DSMNet) | 3D Conv. | ECCV 2020, need to compile |
| ✅ | [CFNet](https://github.com/gallenszl/CFNet) | 3D Conv. | CVPR 2021 |
| ✅ | [RaftStereo](https://github.com/princeton-vl/RAFT-Stereo) | Iterative | 3DV 2021 |
| ✅ | [PCWNet](https://github.com/gallenszl/PCWNet) | 3D Conv. | ECCV 2022 |
| ❌ | [STTR](https://github.com/mli0603/stereo-transformer) | Transformer | ICCV 2021 |
| ❌ | [CREStereo](https://github.com/megvii-research/CREStereo) | Iterative | CVPR 2022, implemented by [MegEngine](https://github.com/MegEngine/MegEngine) |
| ✅ | [IGEVStereo](https://github.com/gangweix/IGEV) | Iterative | CVPR 2023 |
| ❌ | [Selective-IGEVStereo](https://github.com/Windsrain/Selective-Stereo) | Iterative | CVPR 2024 |
| ❌ | [MoChaStereo](https://github.com/ZYangChen/MoCha-Stereo) | Iterative | CVPR 2024 |
| ❌ | [NMRF](https://github.com/aeolusguan/NMRF) | MRF | CVPR 2024 |
| ✅ | [MonSter](https://github.com/Junda24/MonSter) | Iterative | CVPR 2025 |
| ✅ | [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) | Iterative | CVPR 2025 |

- Unless otherwise specified, the maximum search disparity for cost volume filtering methods is defined as 192.
- Please refer to `stereo_toolbox/models/__init__.py` to see the changes in detail.


## 📉 Loss Functions
| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | photometric_loss | |
| ❌ | edge_aware_smoothness_loss | |
| ❌ | single_modal_cross_entropy_loss | |
| ❌ | multi_modal_cross_entropy_loss | |


## 📏 Disparity Estimators

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | softargmax_disparity_estimator | ICCV 2017 |
| ✅ | argmax_disparity_estimator | |
| ✅ | unimodal_disparity_estimator | ICCV 2019 |
| ✅ | dominant_modal_disparity_estimator | CVPR 2024 |


## 🎨 Visualization

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | colored_disparity_map_Spectral_r | Disparity map pseudocolor visualization with Spectral_r colorbar. |
| ✅ | colored_dispairty_map_KITTI | Disparity map pseudocolor visualization with KITTI colorbar. |
| ✅ | colored_error_map_KITTI | Error map pseudocolor visualization with KITTI colorbar. |
| ❌ | colored_pointcloud | Point cloud visualization with real color derived from left image. |


## 📊 Evaluation

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | sceneflow_test | Evaluation on SceneFlow **finalpass** test set. EPE and outliers are reported.|
| ✅ | generalization_eval | Test generalization performance on the training sets of KITTI 2015/2012, Middlebury Eval3, and ETH3D. Outliers in the occ, noc, and all regions are reported.|
| ❌ | kitti2015_sub | Generate dispairty maps in KITTI 2015 submission format. |
| ❌ | kitti2012_sub |  Generate dispairty maps in KITTI 2012 submission format. |
| ❌ | middeval3_sub |  Generate dispairty maps in Middlebury Eval3 submission format. |
| ❌ | eth3d_sub |  Generate dispairty maps in ETH3D submission format. |
| ❌ | speed_and_memery_test | Test inference speed and memory usage. |


**Table 1: Evaluation on the SceneFlow finalpass test set.**

| Model | Checkpoint | EPE | 1px | 2px | 3px |
| ----- | ---------- | :-: | :-: | :-: | :-: |
| PSMNet | [pretrained_sceneflow_new.tar](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view) | 1.1572 | 11.2908 | 6.4028 | 4.7803 |
| GwcNet_GC | [checkpoint_000015.ckpt](https://drive.google.com/file/d/1qiOTocPfLaK9effrLmBadqNtBKT4QX4S/view) | 0.9514 | 8.1138 | 4.6241 | 3.4730 |
| CFNet | [sceneflow_pretraining.ckpt](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view) | 1.2879 | 10.7195 | 7.3116 | 5.9251 |
| PCWNet_GC | [PCWNet_sceneflow_pretrain.ckpt](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view) |  1.0391 | 8.1380 | 4.6462 | 3.5443 |
| RAFTStereo | [raftstereo-sceneflow.pth](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J) | 0.7863 | 7.7104 | 4.8658 | 3.7327 |
| IGEVStereo | [sceneflow.pth](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX) | 0.6790 | 5.7491 | 3.7320 | 2.9069 |
| MonSter | [sceneflow.pth](https://huggingface.co/cjd24/MonSter/blob/main/sceneflow.pth) | 0.5201 | 4.5608 | 2.9705 | 2.3052 |
| DEFOMStereo-S | [defomstereo_vits_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 0.5592 | 5.9396 | 3.7223 | 2.8441 |
| DEFOMStereo-L | [defomstereo_vitl_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 0.4832 | 5.4918 | 3.4421 | 2.6136 |


**Table 2: Generalization evaluation on four real-world training sets.** For all datasets, we report the average error (EPE), outlier rates in occluded, non-occluded, and all regions. The outlier thresholds are set to 3, 3, 2, and 1 for KITTI 2015, KITTI 2012, Middlebury Eval3, and ETH3D, respectively.

| Model | Checkpoint | KITTI 2015 | | | | KITTI 2012 | | | | MiddEval3 | | | | ETH3D | | | |
| ----- | ---------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|       |            | EPE | Occ | Noc | All | EPE | Occ | Noc | All | EPE | Occ | Noc | All | EPE | Occ | Noc | All |
| PSMNet | [pretrained_sceneflow_new.tar](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view) | 4.0584 | 47.6432 | 28.1250 | 28.4160 | 3.8022 | 63.1951 | 26.5022 | 27.3239 | 9.8662 | 62.2950 | 30.1842 | 34.5084 | 2.3997 | 28.5613 | 14.7393 | 15.3888 |
| GwcNet_GC | [checkpoint_000015.ckpt](https://drive.google.com/file/d/1qiOTocPfLaK9effrLmBadqNtBKT4QX4S/view) | 2.3801 | 29.0696 | 12.1746 | 12.5331 | 1.7062 | 45.6458 | 11.9081 | 12.6712 | 6.0044 | 47.1304 | 20.4144 | 24.1094 | 1.9213 | 21.3749 | 10.4911 | 11.0878 |
| CFNet | [sceneflow_pretraining.ckpt](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view) | 1.9798 | 16.4189 | 5.8712 | 6.0967 | 1.0334 | 30.2510 | 4.5758 | 5.1527 | 5.7162 | 44.5492 | 16.3307 | 20.2219 | 0.5862 | 11.8926 | 5.5666 | 5.8700 |
| PCWNet_GC | [PCWNet_sceneflow_pretrain.ckpt](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view) | 1.7777 | 14.9532 | 5.5273 | 5.7416 | 0.9589 | 30.2184 | 4.0734 | 4.6669 | 3.1463 | 37.9880 | 12.1703 | 15.8633 | 0.5284 | 11.6673 | 5.2792 | 5.5360 |
| RAFTStereo | [raftstereo-sceneflow.pth](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J) | 1.1283 | 12.6979 | 5.3413 | 5.5269 | 0.9098 | 28.3453 | 4.2900 | 4.8351 | 1.5231 | 27.9966 | 9.0575 | 11.9563 | 0.3614 | 6.0158 | 2.8471 | 3.0412 |
| IGEVStereo | [sceneflow.pth](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX) | 1.1868 | 14.2606 | 5.5951 | 5.7924 | 1.0131 | 33.6624 | 4.9248 | 5.5936 | 1.5491 | 24.2787 | 7.2518 | 9.9079 | 0.7400 | 9.7601 | 4.0635 | 4.3856 |
| MonSter | [sceneflow.pth](https://huggingface.co/cjd24/MonSter/blob/main/sceneflow.pth) | 0.8884 | 9.6433 | 3.3003 | 3.4495 | 0.7334 | 18.8246 | 3.0310 | 3.3710 | 0.9325 | 18.4153 | 5.8567 | 7.6997 | 0.2724 | 3.5259 | 1.3234 | 1.4525 |
| DEFOMStereo-S | [defomstereo_vits_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 1.0819 | 13.6233 | 4.9982 | 5.1943 | 0.9024 | 23.5715 | 4.3982 | 4.8102 | 1.9487 | 23.8614 | 6.0614 | 8.7609 | 0.2733 | 4.9148 | 2.0263 | 2.1937 |
| DEFOMStereo-L | [defomstereo_vitl_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 1.0725 | 12.5722 | 4.7921 | 4.9853 | 0.8433 | 21.9474 | 3.8260 | 4.2137 | 0.8884 | 20.6396 | 4.3891 | 6.9092 | 0.2533 | 5.1446 | 2.0820 | 2.2437 |


**Table 3: Speed (s) and Memory (MB) Usage.** GPU: NVIDIA GeForce RTX 4090.

| Model | (480, 640) | |  (736, 1280) | | (1088, 1920) | |
| ----- | :---: | :----: | :---: | :----: | :---: | :----: |
|       | Speed | Memory | Speed | Memory | Speed | Memory |
| PSMNet | 0.0396 | 1787.69 | 0.1245 | 4956.50 | 0.2866 | 10687.22 |
| GwcNet_GC | 0.0386 | 1882.58 | 0.1326 | 5251.74 | 0.3093 | 11326.84 |
| CFNet | 0.0481 | 1966.13 | 0.1434 | 5374.05 | 0.3343 | 11526.54 |
| PCWNet_GC | 0.0888 | 3067.07 | 0.2769 | 8629.70 | 0.6419 | 18680.02 |
| RAFTStereo | 0.1967 | 914.25 | 0.3624 | 2227.85 | 0.7613 | 4598.91 |
| IGEVStereo | 0.2363 | 686.43 | 0.3501 | 1504.02 | 0.6741 | 2988.35 |
| MonSter | 0.3375 | 2399.86 | 0.7188 | 3841.63 | 1.8735 | 6537.50 |
| DEFOMStereo-S | 0.1957 | 1062.00 | 0.3423 | 2424.38 | 0.8829 | 4886.10 |
| DEFOMStereo-L | 0.2483 | 2451.85 | 0.5966 | 4005.69 | 1.7410 | 6816.45 |


## 🙏 Acknowledgements

We sincerely thank the authors of the models and datasets mentioned above.
