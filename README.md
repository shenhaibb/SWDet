# SWDet
**PyTorch implementation of "*SWDet: Anchor-based Object Detector for Solid Waste Detection in Aerial Images*"**

## Highlights
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWDet.jpg" alt="SWDet" width="80%"></p>
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWAD_vis.jpg" alt="SWAD_vis"></p>
<p align="center"> <img src="https://raw.github.com/shenhaibb/SWDet/main/imgs/SWAD_result.jpg" alt="SWAD_result"></p>

## Benchmark
|     Method    | Backbone | AP50 | AP75 | Time | Size | Config |           Download                   |
|:-------------:|:--------:|:--------:|:--------:|:--------------:|:------:|:-------:|:--------------------------------------------:|
|   ATSS   | R-50-FPN  |    72.74    |    50.07     |        50.4       |  31.90M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/atss/atss_r50_fpn_1x_coco.py) | [model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  AutoAssign  | R-50-FPN |    75.67    |    50.16     |        49.7       |  35.98M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  Fovea  | R-50-FPN |    74.05    |    49.39     |        48.3       |  36.02M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  PAA  | R-50-FPN |    73.55    |    48.93     |        74.0       |  31.90M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  VFNet  | R-50-FPN |    71.08    |    47.86     |        55.6       |  32.49M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_1x_coco.py) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  YOLOF  | R-50-FPN |    76.57    |    47.64     |        32.7       |  42.16M  | [config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  YOLOv5s  | CSPDarknet |    74.33    |    53.27     |        2.1       |  7.02M  | [config](models/yolov5s.yaml) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  SWDet  | R-50-EAFPN |    74.29    |    55.08     |        5.0       |  17.90M  | [config](models/yolov5s_ResNet50_ECA_weights.yaml) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |
|  SWDet  | ADA-EAFPN |    77.58    |    58.39     |        9.4       |  33.85M  | [config](models/yolov5nsm/yolov5s_DLA_rep_ECA_weights.yaml) |[model](https://pan.baidu.com/s/1VPsAB3Kb90IqJTluH6lFHw) |


## Citation
```
@misc{swdet,
   author={Liming Zhou, Xiaohan Rao, Yahui Li, Xianyu Zuo, Yang Liu, Yinghao Lin, and Yong Yang},
   title={SWDet: Anchor-based Object Detector for Solid Waste Detection in Aerial Images},
   howpublished={\url{https://github.com/shenhaibb/SWDet}},
   year={2022},
}

@misc{yolov5,
   author={Ultralytics},
   title={YOLOv5},
   howpublished={\url{https://github.com/ultralytics/yolov5}}
}
```

## Contact
**Any question regarding this work can be addressed to [shenhaibb@henu.edu.cn](shenhaibb@henu.edu.cn).**
