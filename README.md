# SWDet
**PyTorch implementation of "*SWDet: Anchor-based Object Detector for Solid Waste Detection in Aerial Images*",  [<a href="https://ieeexplore.ieee.org/document/9935119">IEEE JSTARS, 2022</a>].**<br><br>

## Highlights
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWDet.jpg" alt="SWDet" width="80%"></p>
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWAD_vis.jpg" alt="SWAD_vis"></p>
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWAD_result.jpg" alt="SWAD_result"></p>

## Benchmark
|     Method    | Backbone | AP50 | AP75 | Time | Size | Config |           Download                   |
|:-------------:|:--------:|:--------:|:--------:|:--------------:|:------:|:-------:|:--------------------------------------------:|
|   ATSS   | R-50-FPN  |    72.74    |    50.07     |        50.4       |  31.90M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss/atss_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1exN9eLLAVMucrk5WPsEkrgdEpyw7EtG-/view?usp=sharing) |
|  AutoAssign  | R-50-FPN |    75.67    |    50.16     |        49.7       |  35.98M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py) |[model](https://drive.google.com/file/d/10TSfbQSjx2o5x2rVfJLSrIoPqT1ldNhT/view?usp=sharing) |
|  Fovea  | R-50-FPN |    74.05    |    49.39     |        48.3       |  36.02M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py) |[model](https://drive.google.com/file/d/1EwFvlPhx-v_vKeTY4PZHibVTV8eUqHnY/view?usp=sharing) |
|  PAA  | R-50-FPN |    73.55    |    48.93     |        74.0       |  31.90M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/paa/paa_r50_fpn_1x_coco.py) |[model](https://drive.google.com/file/d/1rifsqrz_-Z_h_4RLS2tN8sWkZAV3-Q7B/view?usp=sharing) |
|  VFNet  | R-50-FPN |    71.08    |    47.86     |        55.6       |  32.49M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py) |[model](https://drive.google.com/file/d/13vy4QedQsxwcsxkbNYqhVKlszmu1VaT6/view?usp=sharing) |
|  YOLOF  | R-50-FPN |    76.57    |    47.64     |        32.7       |  42.16M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py) |[model](https://drive.google.com/file/d/1PCWwirl9PJ-y0IqpxRkocgGX1L3v1Fr9/view?usp=sharing) |
|  YOLOv5s  | CSPDarknet |    74.33    |    53.27     |        2.1       |  7.02M  | [config](models/yolov5s.yaml) |[model](https://drive.google.com/file/d/1zL9DgBbEBWtJLlugHmuif4yROwG_6gwc/view?usp=sharing) |
|  SWDet  | R-50-EAFPN |    74.29    |    55.08     |        5.0       |  17.90M  | [config](models/yolov5s_ResNet50_ECA_weights.yaml) |[model](https://drive.google.com/file/d/1__hNboBL23cX2aF2QfYafdEaHY3zBFew/view?usp=sharing) |
|  SWDet  | ADA-EAFPN |    77.58    |    58.39     |        9.4       |  33.85M  | [config](models/yolov5nsm/yolov5s_DLA_rep_ECA_weights.yaml) |[model](https://drive.google.com/file/d/1BX7MM2sYziH98nKv9obg7NyiN4uolPpH/view?usp=sharing) |


## Citation
```
@misc{yolov5,
   author={Ultralytics},
   title={YOLOv5},
   howpublished={\url{https://github.com/ultralytics/yolov5}}
}

@article{swdet,
  title={SWDet: Anchor-based Object Detector for Solid Waste Detection in Aerial Images},
  author={Liming Zhou, Xiaohan Rao, Yahui Li, Xianyu Zuo, Yang Liu, Yinghao Lin, and Yong Yang},
  journal= {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2022}
}

@article{mmdetection,
  title={{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
         Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
         Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
         Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
         Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
         and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Contact
**Any question regarding this work can be addressed to [shenhaibb@henu.edu.cn](shenhaibb@henu.edu.cn).**
