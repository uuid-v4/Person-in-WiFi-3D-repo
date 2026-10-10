# Low-cost Wi-Fi pose estimation with ESP32 receivers (fork of Person-in-WiFi 3D)

This fork rebuilds the Person-in-WiFi 3D pipeline with cheap hardware: **3 transmitters**
(Raspberry Pi 5 + PAU0F, 2x RTL8192EU dongles) and **3 ESP32-S3 receivers** streaming CSI over USB,
labelled by a webcam pose model, trained with the paper's PETR transformer in 2D mode.

**Full guide (hardware layout, firmware, Pi setup, recording, dataset, training, results):
[`csi_demo/README.md`](csi_demo/README.md)**

Held-out session (never seen in training). Green = camera label, red = prediction from Wi-Fi only:

![held-out prediction](csi_demo/docs/heldout_good.gif)

On the held-out session the Wi-Fi model places 54% of joints within 10% of the image (error 12.5%),
vs 30% / 15.5% for always predicting the average pose and 23% / 17.5% for the same model trained on
shuffled CSI (1 person, 1 room, 30 min of data).

The original project README follows.

---

# Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi

Wi-Fi signals, in contrast to cameras, offer privacy protection and occlusion resilience for some practical scenarios such as smart homes, elderly care, and virtual reality.
Recent years have seen remarkable progress in the estimation of single-person 2D pose, single-person 3D pose, and multi-person 2D pose. This paper takes a step forward by introducing Person-in-WiFi 3D, a pioneering Wi-Fi
system that accomplishes multi-person 3D pose estimation. Person-in-WiFi 3D has two main updates. Firstly, it has a greater number of Wi-Fi devices to enhance the capability for capturing spatial reflections from multiple individuals.
Secondly, it leverages the Transformer for end-to-end estimation. Compared to its predecessor, Person-in-WiFi 3D is storage-efficient and fast. We deployed a proof-of-concept system in 4m × 3.5m areas and collected a dataset of over
97K frames with seven volunteers. Person-in-WiFi 3D attains 3D joint localization errors of 91.7mm (1-person), 108.1mm (2-person), and 125.3mm (3-person), comparable to cameras and millimeter-wave radars.


Links to our Project: [Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi](https://aiotgroup.github.io/Person-in-WiFi-3D/)

Links to our Code   : [Person-in-WiFi 3D repo](https://github.com/aiotgroup/Person-in-WiFi3D)

 <img src="demo/demo mini.gif" width = "900" height = "300" alt="demo gif" align=center />

## Prerequisites

- Linux
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.1+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/#installation)

## Getting Started

### Installation

Please see [get_started.md](docs/get_started.md) for the basic usage of Opera.

## Acknowledgement

Opera is an open source project built upon [OpenMMLab](https://github.com/open-mmlab/). We appreciate all the contributors who implement this flexible and efficient toolkits.



### File Structrue
```bash
.
│  LICENSE
│  README.md
│  requirements.txt
│  setup.cfg
│  setup.py
├─data
│  ├─wifipose
│  │  ├─train_data
│  │  │     ├─ csi
│  │  │     ├─ keypoint
│  │  │     ├─ train_data_list.txt
│  │  ├─test_data
│  │  │     ├─ csi
│  │  │     ├─ keypoint
│  │  │     ├─ test_data_list.txt
├─config
│  ├─base
│  ├─wifi
│  │  ├─petr_wifi.py
├─docs
│  ├─get_started.md
├─opera
│  ├─apis
│  ├─core
│  ├─datasets
│  ├─models
│  ├─__init__.py
│  ├─version.py
├─requirements
│  ├─build.txt
│  ├─docs.txt
│  ├─mminstall.txt
│  ├─optional.txt
│  ├─readthedocs.txt
│  ├─runtime.txt
│  ├─tests.txt
├─result
├─third_party
│  ├─mmcv
│  ├─mmdet
├─tools
│  ├─dataset_converters
│  ├─dist_test.sh
│  ├─dist_train.sh
│  ├─eval_metric.py
│  ├─test_all.sh
│  ├─test.py
│  ├─train.py


```


## Citations

If you find our works useful in your research, please consider citing:
```BibTeX
@inproceedings{person3dyan,
  title={Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi },
  author={Yan, Kangwei and Wang, Fei and Qian, Bo and Ding, Han and Han, Jinsong and Wei, Xing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024}
}

@inproceedings{shi2022end,
  title={End-to-End Multi-Person Pose Estimation With Transformers},
  author={Shi, Dahu and Wei, Xing and Li, Liangqi and Ren, Ye and Tan, Wenming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11069--11078},
  year={2022}
}


```

## License

This project is released under the [Apache 2.0 license](LICENSE).
