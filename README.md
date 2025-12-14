# 🔧 Stereo Toolbox

A comprehensive stereo matching toolbox for efficient development and research.


## 📦 Installation

```
pip install stereo_toolbox
```


## 🔄 Datasets (⚠️ deprecated)

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
| ✅ | [FallingThings_Dataset](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation) | 61K+ | 0 | 0 | ❌ | Synthetic dataset with object models and backgrounds of complex composition and high graphical quality. |
| ✅ | [VirtualKITTI2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) | 21K+ | 0 | 0 | ❌ | A more photo-realistic and better-featured version of the original virtual KITTI dataset. |
| ✅ | [LayeredFlow](https://layeredflow.cs.princeton.edu) | ❌ | 400 | 600 | ❌ |  Indoor and outdoor dataset with non-Lambertian objects. |



**Dataloader Return:**
- left image (color jitter if training)
- right image (color jitter and random mask if training)
- disparity ground-truth (nan if not available)
- non-occucluded mask (nan if not available)
- raw left image (not normalized)
- raw right image (not normalized)


## 🔄 Datasets V2

| Status | Identifier | Train | Val | Test | Noc. Mask | Description |
| :----: | ---------- | :---: | :-: | :--: | :----------------: | ----------- | 
| ✅ | [SceneFlow_Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) | 35K+ | 4.3K+ | - | ❌ | The most famous synthetic dataset for stereo matching pre-training. |
| ✅ | [KITTI2015_Dataset](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | 200 | - | 200 | ✅ | Driving scene dataset. |
| ✅ | [KITTI2012_Dataset](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) | 192 | - | 195 | ✅ | Driving scene dataset. |
| ✅ | [MiddleburyEval3_Dataset](https://vision.middlebury.edu/stereo/submit3) | 15 | - | 15 | ✅ | Indoor and outdoor scene dataset. |
| ✅ | [SCARED]() | 35 | - | | ❌ | Endoscopic (Porcine cadavers) datasets with structured light data. |
| ✅ | [MiddleburyEval3_Dataset](https://vision.middlebury.edu/stereo/submit3) | 15 | - | 15 | ✅ | Indoor and outdoor scene dataset. |
| ✅ | [ETH3D_Dataset](https://www.eth3d.net/datasets) | 27 | - | 20 | ✅ | Indoor scene dataset with grayscale images. |
| ✅ | [DrivingStereo_Dataset](https://drivingstereo-dataset.github.io/)| 174K+ | 7.7K+ | - | ❌ | Driving scene dataset with diverse weathers (sunny, cloudy, foggy, rainy). |

**Dataloader Return:**
Returns a dictionary containing the requested data types:
- ref (torch.Tensor): Reference image in C*H*W format, values in [0, 255].
- tgt (torch.Tensor): Target image in C*H*W format, values in [0, 255].
- gt_disp (torch.Tensor): Ground truth disparity map in H*W format, with 0 indicating invalid pixels.
- noc_mask (torch.Tensor): Non-occluded mask in H*W format, with Flase for occluded and True for non-occluded pixels.
- raw_ref (torch.Tensor): Unaugmented reference image in C*H*W format, values in [0, 255].
- raw_tgt (torch.Tensor): Unaugmented target image in C*H*W format, values in [0, 255].
- ref_filename (str): Filename of the reference image.
- top_pad (int): Number of pixels padded at the top during testing.
- right_pad (int): Number of pixels padded on the right during testing.


## 🧠 Models

| Status | Identifier | Architecture | Description |
| :----: | ---------- | ------------ | ----------- |
| ✅ | [PSMNet](https://github.com/JiaRenChang/PSMNet) | 3D Conv. | CVPR 2018, change `.cuda()` to `.to(x.device)`, optimize the cost volume building. |
| ✅ | [GwcNet](https://github.com/xy-guo/GwcNet) | 3D Conv. | CVPR 2019, two models `GwcNet_G` and `GwcNet_GC`. |
| ❌ | [GANet](https://github.com/feihuzhang/GANet) | 3D Conv. | CVPR 2019, need to compile |
| ❌ | [AANet](https://github.com/haofeixu/aanet) | 2D Conv. | CVPR 2020, need to compile. |
| ❌ | [DSMNet](https://github.com/feihuzhang/DSMNet) | 3D Conv. | ECCV 2020, need to compile. |
| ✅ | [CFNet](https://github.com/gallenszl/CFNet) | 3D Conv. | CVPR 2021, `mish` avtivation function only, return `pred1_s2` only when evaluation. |
| ✅ | [STTR](https://github.com/mli0603/stereo-transformer) | Transformer | ICCV 2021, return `output['disp_pred']` only when evaluation. |
| ✅ | [RaftStereo](https://github.com/princeton-vl/RAFT-Stereo) | Iterative | 3DV 2021, add default `self.args` in `__init__()`, reset left as positive direction (i.e. invert all outputs). |
| ✅ | [ACVNet](https://github.com/gangweix/acvnet) | 3D Conv. | CVPR 2022. |
| ❌ | [CREStereo](https://github.com/megvii-research/CREStereo) | Iterative | CVPR 2022, implemented by [MegEngine](https://github.com/MegEngine/MegEngine). |
| ✅ | [PCWNet](https://github.com/gallenszl/PCWNet) | 3D Conv. | ECCV 2022, rename class `PWCNet` as `PCWNet`, two models `PCWNet_G` and `PCWNet_GC`, `mish` avtivation function only, return `disp_finetune` only when evaluation. |
| ✅ | [IGEVStereo](https://github.com/gangweix/IGEV) | 3D Conv. + Iterative | CVPR 2023, timm==0.5.4. |
| ✅ | [GMStereo(UniMatch)](https://github.com/autonomousvision/unimatch) | Transformer | T-PAMI 2023, return `results_dict['flow_preds'][-1]` only when evaluation. |
| ✅ | [CroCoStereo(CroCov2)](https://github.com/naver/croco) | Transformer | ICCV 2023, set `overlap=0.7` |
| ✅ | [SelectiveStereo](https://github.com/Windsrain/Selective-Stereo) | 3D Conv. + Iterative | CVPR 2024, two models `SelectiveRAFT` and `SelectiveIGEV`, add default `self.args` in `__init__()`, timm==0.5.4. |
| ❌ | [MoChaStereo](https://github.com/ZYangChen/MoCha-Stereo) | Iterative | CVPR 2024. |
| ❌ | [NMRF](https://github.com/aeolusguan/NMRF) | MRF | CVPR 2024. |
| ✅ | [MonSter](https://github.com/Junda24/MonSter) | 3D Conv. + Iterative | CVPR 2025, add default `self.args` in `__init__()`, timm==0.5.4. |
| ✅ | [DEFOM-Stereo](https://github.com/Insta360-Research-Team/DEFOM-Stereo) | Iterative | CVPR 2025, add default `self.args` in `__init__()`, note that the used depthanythingv2 has additional interpolation step, timm<=0.6.5|
| ✅ | [FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 3D Conv. + Iterative | CVPR 2025 Best Paper Nomination,  add default `self.args` in `__init__()`, timm==0.6.5. |
| ✅ | [StereoAnywhere](https://github.com/bartn8/stereoanywhere) | Iterative | CVPR 2025, integrate depthanythingv2 into the forward process. |
| ✅ | [IGEVStereoPlusPluss (IGEV++)](https://github.com/gangweix/IGEV-plusplus) | 3D Conv. + Iterative | T-PAMI 2025， timm=0.5.4 |

- Unless specified, the maximum search disparity for cost volume filtering methods is set to 192.
- All predictions are output as a list during training, and only the final disparity map is output during inference.
- For all iterative methods, the default training and validation iterations are set to 22 and 32, respectively.
- Due to version dependency, please additionally install timm==0.5.4 and rename it to timm_0_5_4:
    ```
    wget https://github.com/huggingface/pytorch-image-models/archive/refs/tags/v0.5.4.zip
    # unzip the zip file
    cd pytorch-image-models-0.5.4
    # replace 'timm' in 'setup.py' with 'timm_0_5_4'
    # replace all the 'import timm' and 'from timm' with 'import timm_0_5_4' and 'from timm_0_5_4', respectively
    pip install .
    ```


## 📉 Loss Functions
| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | photometric_loss | |
| ✅ | smoothness_loss | |
| ❌ | triplet_photometric_loss | CVPR 2023, [NerfStereo](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo). |
| ❌ | single_modal_cross_entropy_loss | |
| ❌ | multi_modal_cross_entropy_loss | |


## 📏 Disparity Estimators

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | softargmax_disparity_estimator | [ICCV 2017](https://arxiv.org/abs/1703.04309). |
| ✅ | argmax_disparity_estimator | |
| ✅ | unimodal_disparity_estimator | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.pdf). |
| ✅ | dominant_modal_disparity_estimator | [CVPR 2024](https://github.com/xxxupeng/ADL). |


## 🎨 Visualization

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | colored_disparity_map_Spectral_r | Disparity map pseudocolor visualization with Spectral_r colorbar. |
| ✅ | colored_disparity_map_KITTI | Disparity map pseudocolor visualization with KITTI colorbar. |
| ✅ | colored_error_map_KITTI | Error map pseudocolor visualization with KITTI colorbar. |
| ❌ | colored_pointcloud | Point cloud visualization with real color derived from left image. |


## 📊 Evaluation

| Status | Identifier | Description |
| :----: | ---------- | ----------- |
| ✅ | sceneflow_test | Evaluation on SceneFlow **finalpass** test set. EPE and outliers are reported. Valid disparity range `0~maxdisp-1`, default `0~191`. |
| ✅ | generalization_eval | Test generalization performance on the training sets of KITTI 2015/2012, Middlebury Eval3, and ETH3D. Outliers in the occ, noc, and all regions are reported. Valid disparity range `0~maxdisp-1`, default `0~191`.|
| ✅ | speed_and_memery_test | Test inference speed and memory usage. |
| ✅ | drivingstereo_weather_test | Test generalization performance on different weathers of DrivingStereo test sets (half). |
| ✅ | benchmark_submission | Generate zip files for submitting to benchmarks (KITTI 2015 (verified), KITTI 2012 (verified), MiddEval3, and ETH3D). |


**Table 1: Evaluation on SceneFlow finalpass test set.**

| Model | Checkpoint | EPE | 1px | 2px | 3px |
| ----- | ---------- | :-: | :-: | :-: | :-: |
| PSMNet | [pretrained_sceneflow_new.tar](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view) | 1.1572 | 11.2908 | 6.4028 | 4.7803 |
| GwcNet_GC | [checkpoint_000015.ckpt](https://drive.google.com/file/d/1qiOTocPfLaK9effrLmBadqNtBKT4QX4S/view) | 0.9514 | 8.1138 | 4.6241 | 3.4730 |
| CFNet | [sceneflow_pretraining.ckpt](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view) | 1.2879 | 10.7195 | 7.3116 | 5.9251 |
| STTR<sup>&dagger;</sup> | [sceneflow_pretrained_model.pth.tar](https://drive.google.com/file/d/1R0YUpFzDRTKvjRfngF8SPj2JR2M1mMTF/view) | 4.5613 | 15.6220 | 12.3084 | 11.3189 |
| RAFTStereo | [raftstereo-sceneflow.pth](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J) | 0.7863 | 7.7104 | 4.8658 | 3.7327 |
| ACVNet | [sceneflow.ckpt](https://drive.google.com/drive/folders/1oY472efAgwCCSxtewbbA2gEtee-dlWSG) | 0.6860 | 5.1409 | 2.9201 | 2.1832 |
| PCWNet_GC | [PCWNet_sceneflow_pretrain.ckpt](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view) |  1.0391 | 8.1380 | 4.6462 | 3.5443 |
| IGEVStereo | [sceneflow.pth](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX) | 0.6790 | 5.7491 | 3.7320 | 2.9069 |
| GMStereo | [GMStereo-scale2-regrefine3-resumeflowthings-sceneflow](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth) | 0.6355 | 6.1353 | 3.4315 | 2.5237 |
| CroCoStereo | [crocostereo.pth](https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/crocostereo.pth) | 0.6822 | 5.1854 | 3.3273 |2.6104 |
| SelectiveRAFT | [sceneflow.pth](https://drive.google.com/drive/folders/14c5E8znK_F3wk-C_xiC4V2JT3yeDL48g) | 0.6956 | 5.7341 | 3.7000 | 2.8816 |
| SelectiveIGEV | [sceneflow.pth](https://drive.google.com/drive/folders/1VyBzwQJAsKPXFpkcCWFn_IiAWWjpWYbz) | 0.6048 | 5.3667 | 3.4717 | 2.6904 |
| MonSter<sup>&Dagger;</sup> | [sceneflow.pth](https://huggingface.co/cjd24/MonSter/blob/main/sceneflow.pth) | 0.5201 | 4.5608 | 2.9705 | 2.3052 |
| DEFOMStereo-S<sup>&Dagger;</sup> | [defomstereo_vits_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 0.5592 | 5.9396 | 3.7223 | 2.8441 |
| DEFOMStereo-L<sup>&Dagger;</sup> | [defomstereo_vitl_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 0.4832 | 5.4918 | 3.4421 | 2.6136 |
| FoundationStereo-S<sup>&Dagger;</sup> | [11-33-40/model_best_bp2.pth](https://drive.google.com/drive/folders/1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL) | 0.5165 | 4.0213 | 2.4983 | 1.9194 |
| FoundationStereo-L<sup>&Dagger;</sup> | [23-51-11/model_best_bp2.pth](https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-) | 0.4966 | 3.6243 | 2.2180 | 1.7123 |
| StereoAnywhere<sup>&Dagger;</sup> | [sceneflow.tar](https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x) | 0.9109 | 7.9459% | 5.0610 | 4.0071 |
| IGEV++ | [sceneflow.pth](https://drive.google.com/drive/folders/19WA5WeoSQGFxxyzq7ZhJFeptz4zwBNuq) | 0.6269 | 4.7347 | 2.8433 | 2.1624 |

- <sup>&dagger;</sup>w/o occluded mask input
- <sup>&Dagger;</sup>employed the foundation model (DepthAnything v2).


**Table 2: Generalization evaluation on four real-world training sets.** For all datasets, we report the average error (EPE), outlier rates in occluded, non-occluded, and all regions. The outlier thresholds are set to 3, 3, 2, and 1 for KITTI 2015, KITTI 2012, Middlebury Eval3, and ETH3D, respectively.

| Model | Checkpoint | KITTI 2015 | | | | KITTI 2012 | | | | MiddEval3 | | | | ETH3D | | | |
| ----- | ---------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|       |            | EPE | Occ | Noc | All | EPE | Occ | Noc | All | EPE | Occ | Noc | All | EPE | Occ | Noc | All |
| PSMNet | [pretrained_sceneflow_new.tar](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view) | 4.0584 | 47.6432 | 28.1250 | 28.4160 | 3.8022 | 63.1951 | 26.5022 | 27.3239 | 9.8662 | 62.2950 | 30.1842 | 34.5084 | 2.3997 | 28.5613 | 14.7393 | 15.3888 |
| GwcNet_GC | [checkpoint_000015.ckpt](https://drive.google.com/file/d/1qiOTocPfLaK9effrLmBadqNtBKT4QX4S/view) | 2.3801 | 29.0696 | 12.1746 | 12.5331 | 1.7062 | 45.6458 | 11.9081 | 12.6712 | 6.0044 | 47.1304 | 20.4144 | 24.1094 | 1.9213 | 21.3749 | 10.4911 | 11.0878 |
| CFNet | [sceneflow_pretraining.ckpt](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view) | 1.9798 | 16.4189 | 5.8712 | 6.0967 | 1.0334 | 30.2510 | 4.5758 | 5.1527 | 5.7162 | 44.5492 | 16.3307 | 20.2219 | 0.5862 | 11.8926 | 5.5666 | 5.8700 |
| STTR | [sceneflow_pretrained_model.pth.tar](https://drive.google.com/file/d/1R0YUpFzDRTKvjRfngF8SPj2JR2M1mMTF/view) | 2.1786 | 90.9327 | 6.8101 | 8.3029 | 2.8117 | 94.3034 | 7.1706 | 9.1719 | 8.6700 | 88.7300 | 19.3553 | 28.1827 | 2.2964 | 50.0450 | 15.8716 | 17.5654 |
| RAFTStereo | [raftstereo-sceneflow.pth](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J) | 1.1283 | 12.6979 | 5.3413 | 5.5269 | 0.9098 | 28.3453 | 4.2900 | 4.8351 | 1.5231 | 27.9966 | 9.0575 | 11.9563 | 0.3614 | 6.0158 | 2.8471 | 3.0412 |
| ACVNet | [sceneflow.ckpt](https://drive.google.com/drive/folders/1oY472efAgwCCSxtewbbA2gEtee-dlWSG) | 2.5105 | 32.8509 | 11.2934 | 11.7108 | 2.0233 | 54.4658 | 12.9433 | 13.8876 | 6.2429 | 47.3617 | 22.0709 | 25.6607 | 2.4436 | 19.6435 | 8.6531 | 9.1933 |
| PCWNet_GC | [PCWNet_sceneflow_pretrain.ckpt](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view) | 1.7777 | 14.9532 | 5.5273 | 5.7416 | 0.9589 | 30.2184 | 4.0734 | 4.6669 | 3.1463 | 37.9880 | 12.1703 | 15.8633 | 0.5284 | 11.6673 | 5.2792 | 5.5360 |
| IGEVStereo | [sceneflow.pth](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX) | 1.1868 | 14.2606 | 5.5951 | 5.7924 | 1.0131 | 33.6624 | 4.9248 | 5.5936 | 1.5491 | 24.2787 | 7.2518 | 9.9079 | 0.7400 | 9.7601 | 4.0635 | 4.3856 |
| GMStereo | [GMStereo-scale2-regrefine3-resumeflowthings-sceneflow](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth) | 1.1957 | 19.4742 | 5.3210 | 5.6365 | 1.1021 | 36.7635 | 4.7391 | 5.4961 | 2.1188 | 41.0546 | 12.7984 | 16.6042 | 0.4013 | 14.3825 | 5.5249 | 5.8759 |
| CroCoStereo | [crocostereo.pth](https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/crocostereo.pth) | 5.3563 | 44.5432 | 19.8355 | 20.2043 | 3.3760 | 41.0264 | 13.8920 | 14.4562 | - | - | - | - | - | - | - | - |
| [NerfStereo-RAFT](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)<sup>&dagger;</sup> | [raftstereo-NS.tar](https://drive.google.com/file/d/1zAX2q1Tr9EOypXv5kwkI4a_YTravdtsS/view) | 1.1330 | 14.6178 | 5.2269 | 5.4257 | 0.8592 | 26.9731 | 3.5119 | 4.0440 | 1.6247 | 31.0983 | 6.7877 | 10.3770 | 0.2992 | 8.3545 | 2.7778 | 3.0729 |
| SelectiveRAFT | [sceneflow.pth](https://drive.google.com/drive/folders/14c5E8znK_F3wk-C_xiC4V2JT3yeDL48g) | 1.2629 | 17.7190 | 6.0989 | 6.3532 | 1.0889 | 28.0310 | 4.9432 | 5.4576 | 1.6684 | 26.7379 | 7.5835 | 10.5572 | 0.3958 | 8.8286 | 3.8131 | 4.2670 |
| SelectiveIGEV | [sceneflow.pth](https://drive.google.com/drive/folders/1VyBzwQJAsKPXFpkcCWFn_IiAWWjpWYbz) | 1.2124 | 13.8184 | 5.7032 | 5.8859 | 1.0068 | 31.8457 | 5.0626 | 5.6780 | 1.3974 | 22.5942 | 6.7270 | 9.1742 | 0.4373 | 9.8115 | 4.0689 | 4.4284 |
| MonSter<sup>&Dagger;</sup> | [sceneflow.pth](https://huggingface.co/cjd24/MonSter/blob/main/sceneflow.pth) | 0.8884 | 9.6433 | 3.3003 | 3.4495 | 0.7334 | 18.8246 | 3.0310 | 3.3710 | 0.9325 | 18.4153 | 5.8567 | 7.6997 | 0.2724 | 3.5259 | 1.3234 | 1.4525 |
| DEFOMStereo-S<sup>&Dagger;</sup> | [defomstereo_vits_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 1.0819 | 13.6233 | 4.9982 | 5.1943 | 0.9024 | 23.5715 | 4.3982 | 4.8102 | 1.9487 | 23.8614 | 6.0614 | 8.7609 | 0.2733 | 4.9148 | 2.0263 | 2.1937 |
| DEFOMStereo-L<sup>&Dagger;</sup> | [defomstereo_vitl_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) | 1.0725 | 12.5722 | 4.7921 | 4.9853 | 0.8433 | 21.9474 | 3.8260 | 4.2137 | 0.8884 | 20.6396 | 4.3891 | 6.9092 | 0.2533 | 5.1446 | 2.0820 | 2.2437 |
| [ZeroStereo-RAFT](https://github.com/Windsrain/ZeroStereo/tree/main)<sup>&dagger;</sup> | [model.safetensors](https://drive.google.com/drive/folders/1x2SdUgXLv1rpiNT9xbY82wZch0ebZaJe) | 1.0306 | 11.1673 | 4.4509 | 4.6312 | 0.7484 | 20.5038 | 3.1816 | 3.5517 | 1.3451 | 23.8572 | 4.6174 | 7.5843 | 0.2346 | 6.3722 | 1.9073 | 2.2238 |
| [ZeroStereo-IGEV](https://github.com/Windsrain/ZeroStereo/tree/main)<sup>&dagger;</sup> | [model_192.safetensors](https://drive.google.com/drive/folders/1Wkvhw455SAgXukTzyzU2ltcv4lzDJB6F) | 1.0061 | 10.5266 | 4.3593 | 4.5312 | 0.7394 | 19.4140 | 3.1647 | 3.5043 | 1.1126 | 21.2663 | 4.8955 | 7.3997 | 0.2297 | 6.2541 | 1.9331 | 2.1894 |
| FoundationStereo-S<sup>&Dagger;</sup> | [11-33-40/model_best_bp2.pth](https://drive.google.com/drive/folders/1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL) | 0.8812 | 9.3458 | 3.1114 | 3.2651 | 0.6646 | 16.5310 | 2.5713 | 2.8640 | 0.5377 | 10.6189 | 1.3074 | 2.7566 | 0.1612 | 2.5606 | 0.7094 | 0.7776 |
| FoundationStereo-L<sup>&Dagger;</sup> | [23-51-11/model_best_bp2.pth](https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-) | 0.8746 | 8.1694 | 3.1106 | 3.2388 | 0.6692 | 15.3932 | 2.6666 | 2.9389 | 0.5060 | 8.5997 | 1.1179 | 2.2698 | 0.1517 | 2.2747 | 0.5694 | 0.6525 |
| StereoAnywhere<sup>&Dagger;</sup> | [sceneflow.tar](https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x) | 0.9531 | 11.2782 | 3.6955 | 3.8666 | 0.8347 | 21.6785 | 3.5198 | 3.9184 | 1.2584 | 20.7357 | 5.3881 | 7.629 | 0.2362 | 4.2515 | 1.5314 | 1.7172 |
| IGEV++ | [sceneflow.pth](https://drive.google.com/drive/folders/19WA5WeoSQGFxxyzq7ZhJFeptz4zwBNuq) | 1.2942 | 16.6714 | 6.2537 | 6.4695 | 1.1315 | 34.6407 | 5.7610 | 6.4504 | 2.3679 | 26.2245 | 7.2190 | 10.0922 | 0.4844 | 5.8613 | 4.2269 | 4.2871 |

- <sup>&dagger;</sup>trained on extra synthetic/real data.
- <sup>&Dagger;</sup>employed the foundation model (DepthAnything v2).
- CroCoStereo was trained with Middlebury and ETH3D datasets. 

**Table 3: Inference speed (s) and memory (MB) usage.** Device: NVIDIA GeForce RTX 4090.

| Model | (480, 640) | |  (736, 1280) | | (1088, 1920) | | #Param. | #Learnable param. |
| ----- | :---: | :----: | :---: | :----: | :---: | :----: | :----: | :----: |
|       | Speed | Memory | Speed | Memory | Speed | Memory | (M) | (M) |
| PSMNet | 0.0396 | 1787.69 | 0.1245 | 4956.50 | 0.2866 | 10687.22 |  5.22 |	5.22 |
| GwcNet_GC | 0.0386 | 1882.58 | 0.1326 | 5251.74 | 0.3093 | 11326.84 | 6.91 |	6.91 |
| CFNet | 0.0481 | 1966.13 | 0.1434 | 5374.05 | 0.3343 | 11526.54 | 23.05 |	23.05 |
| STTR | 0.1556 | 3036.80 | 0.8468 | 16588.08 | OOM | OOM | 2.51 |	2.51 |
| RAFTStereo | 0.1967 | 914.25 | 0.3624 | 2227.85 | 0.7613 | 4598.91 | 11.12 |	11.12 |
| ACVNet | 0.0494 | 2098.31 | 0.1664 | 6344.20 | 0.3848 | 14021.82 | 7.17 |	7.17 |
| PCWNet_GC | 0.0888 | 3067.07 | 0.2769 | 8629.70 | 0.6419 | 18680.02 | 35.94	 |35.94 |
| IGEVStereo | 0.2363 | 686.43 | 0.3501 | 1504.02 | 0.6741 | 2988.35 | 12.60 |	12.60 |
| GMStereo | 0.0571 | 937.78 | 0.2011 | 2792.14 | 0.6261 | 8412.21 | 7.35 | 7.35 |
| CroCoStereo | 0.1710 | 2292.61 | 1.1333 | 2319.24 | 3.1700 | 2370.83 | 437.42 | 437.42 |
| SelectiveRAFT | 0.1776 | 731.03 | 0.4253 | 1559.72 | 0.9899 | 3171.54 | 11.65 |	11.65 |
| SelectiveIGEV | 0.1853 | 600.90 | 0.3843 | 1406.60 | 0.8850 | 2895.57 | 13.14	 |13.14 |
| MonSter | 0.3375 | 2399.86 | 0.7188 | 3841.63 | 1.8735 | 6537.50 | 388.69	 |53.38 |
| DEFOMStereo-S | 0.1957 | 1062.00 | 0.3423 | 2424.38 | 0.8829 | 4886.10 | 43.29 | 18.51 |
| DEFOMStereo-L | 0.2483 | 2451.85 | 0.5966 | 4005.69 | 1.7410 | 6816.45 | 382.62 | 47.30 |
| FoundationStereo-S | 0.2792 | 4522.09 | 0.6896 | 7237.75 | 1.5627 | 12145.71 | 62.34 |37.55 |
| FoundationStereo-L | 0.3327 | 2811.62 | 0.8369  | 5569.83 | 1.7758 | 10555.90 | 374.52 | 39.20 |
| StereoAnywhere | 0.4172 | 2785.66 | 0.8389 | 7810.07 | 2.6662 | 22433.76 | 346.75	 | 11.43 |
| IGEV++ |  0.2881 | 741.00 | 0.4362 | 2066.62 | 0.8782 | 4619.21 | 14.53  |14.53 |


**Table 4: Generalization across different weathers.** The outlier threshold is set to 3.
| Model | Checkpoint | Sunny | | Cloudy | | Rainy | | Foggy | |
| ----- | ---------- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
|       |            | EPE | Outliers | EPE | Outliers | EPE | Outliers | EPE | Outliers | EPE | Outliers |
| PSMNet | [pretrained_sceneflow_new.tar](https://drive.google.com/file/d/1NDKrWHkwgMKtDwynXVU12emK3G5d5kkp/view) | 7.9699 | 40.1363 | 12.8784 | 43.9466 | 24.5091 | 56.187 | 31.558 | 69.6891 |
| GwcNet_GC | [checkpoint_000015.ckpt](https://drive.google.com/file/d/1qiOTocPfLaK9effrLmBadqNtBKT4QX4S/view) | 2.2694 | 17.1220 | 3.5672 | 25.5583 | 4.9620 | 28.1909 | 3.3859 | 29.2295 |
| CFNet | [sceneflow_pretraining.ckpt](https://drive.google.com/file/d/1gFNUc4cOCFXbGv6kkjjcPw2cJWmodypv/view) | 1.1168 |  4.6957 |  1.0915 |  5.3006 |  1.8753 |  12.4819 |  1.1242 |  5.5388 |
| STTR | [sceneflow_pretrained_model.pth.tar](https://drive.google.com/file/d/1R0YUpFzDRTKvjRfngF8SPj2JR2M1mMTF/view) |  2.6073 | 6.9961 | 2.4905 | 7.9241 | 7.8698 | 23.7624 | 2.2568 | 8.3199 |
| RAFTStereo | [raftstereo-sceneflow.pth](https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J) |  1.1015 | 4.2288 | 1.0457 | 4.1902 | 2.0409 | 12.7736 | 0.9909 | 3.0875 |
| ACVNet | [sceneflow.ckpt](https://drive.google.com/drive/folders/1oY472efAgwCCSxtewbbA2gEtee-dlWSG) |  2.5432 | 19.6405 | 4.1897 | 29.8733 | 12.3508 | 41.3112 | 5.8133 | 38.0457 |
| PCWNet_GC | [PCWNet_sceneflow_pretrain.ckpt](https://drive.google.com/file/d/18HglItUO7trfi-klXzqLq7KIDwPSVdAM/view) |  0.9841 | 3.5835 | 1.0074 | 3.6724 | 1.9833 | 10.5247 | 1.1282 | 5.1968 |
| IGEVStereo | [sceneflow.pth](https://drive.google.com/drive/folders/1yqQ55j8ZRodF1MZAI6DAXjjre3--OYOX) |  1.0485 | 4.5893 | 1.1052 | 5.1544 | 2.2975 | 15.4724 | 1.0657 | 4.4922 |
| GMStereo | [GMStereo-scale2-regrefine3-resumeflowthings-sceneflow](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth) | 1.3744 | 6.8031 | 1.3299 | 7.0328 | 2.9797 | 16.7326 | 1.5465 | 9.7642 |
| CroCoStereo | [crocostereo.pth](https://download.europe.naverlabs.com/ComputerVision/CroCo/StereoFlow_models/crocostereo.pth) | 2.0420 | 8.2903 | 1.4104 | 5.7141 | 2.3852 | 16.8024 | 1.6361 | 6.3033 |
| [NerfStereo-RAFT](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo)<sup>&dagger;</sup> | [raftstereo-NS.tar](https://drive.google.com/file/d/1zAX2q1Tr9EOypXv5kwkI4a_YTravdtsS/view) |  0.9003 | 2.8822 | 0.9145 | 2.9105 | 1.7485 | 10.2047 | 1.0682 | 3.9268 |
| SelectiveRAFT | [sceneflow.pth](https://drive.google.com/drive/folders/14c5E8znK_F3wk-C_xiC4V2JT3yeDL48g) |  1.1099 | 4.8376 | 1.0555 | 4.4836 | 1.8238 | 13.9435 | 0.9648 | 3.4256 |
| SelectiveIGEV | [sceneflow.pth](https://drive.google.com/drive/folders/1VyBzwQJAsKPXFpkcCWFn_IiAWWjpWYbz) |  1.1242 | 5.0513 | 1.1139 | 5.2406 | 2.0507 | 13.5095 | 1.0679 | 4.1028 |
| MonSter<sup>&Dagger;</sup> | [sceneflow.pth](https://huggingface.co/cjd24/MonSter/blob/main/sceneflow.pth) |  0.9857 | 3.4775 | 0.9318 | 3.1687 | 1.1267 | 5.2665 | 1.1023 | 5.0289 |
| DEFOMStereo-S<sup>&Dagger;</sup> | [defomstereo_vits_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) |  0.9678 | 3.7935 | 0.9836 | 4.0202 | 1.4416 | 12.9997 | 0.9787 | 3.4861 |
| DEFOMStereo-L<sup>&Dagger;</sup> | [defomstereo_vitl_sceneflow.pth](https://drive.google.com/drive/folders/1cZLcIjHlmUo986gkR6FbofG1cj5BT36x) |  0.9740 | 3.6134 | 0.9970 | 3.7463 | 1.5175 | 13.5251 | 0.9330 | 2.8767 |
| [ZeroStereo-RAFT](https://github.com/Windsrain/ZeroStereo/tree/main)<sup>&dagger;</sup> | [model.safetensors](https://drive.google.com/drive/folders/1x2SdUgXLv1rpiNT9xbY82wZch0ebZaJe) |  0.8423 | 2.6192 | 0.8457 | 2.4586 | 2.2867 | 13.6386 | 0.8388 | 1.7887 |
| [ZeroStereo-IGEV](https://github.com/Windsrain/ZeroStereo/tree/main)<sup>&dagger;</sup> | [model_192.safetensors](https://drive.google.com/drive/folders/1Wkvhw455SAgXukTzyzU2ltcv4lzDJB6F) |  0.8487 | 2.5887 | 0.8379 | 2.3636 | 1.7912 | 13.0688 | 0.8483 | 1.8949 |
| FoundationStereo-S<sup>&Dagger;</sup> | [11-33-40/model_best_bp2.pth](https://drive.google.com/drive/folders/1qKDRgdBJFRRRBf_UlInkmOiSzW9jiNDL) |  0.8651 | 2.7184 | 0.8791 | 2.5519 | 1.5783 | 13.4365 | 1.0134 | 3.3486 |
| FoundationStereo-L<sup>&Dagger;</sup> | [23-51-11/model_best_bp2.pth](https://drive.google.com/drive/folders/1BbhoPliFqPJlrtD65TgNX49sJYuYcwA-) |  0.9427 | 3.1979 | 0.8918 | 2.6189 | 5.3590 | 26.9548 | 2.1145 | 5.6173 |
| StereoAnywhere<sup>&Dagger;</sup> | [sceneflow.tar](https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x) |  0.9713 | 3.5070 | 0.9068 | 2.9285 | 1.3656 | 10.8506 | 0.9408 | 2.8968 |
| IGEV++ | [sceneflow.pth](https://drive.google.com/drive/folders/19WA5WeoSQGFxxyzq7ZhJFeptz4zwBNuq) | 1.1330 | 4.9870 | 1.1195 | 5.1955 | 3.3631 | 17.1510 | 1.2028 | 5.9750 |


## 🙏 Acknowledgements

We sincerely thank the authors of the models and datasets mentioned above.
