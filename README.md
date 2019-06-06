# Accel


## Introduction

**Accel** is a fast, high accuracy video segmentation system, initially described in a [CVPR 2019 paper](https://arxiv.org/abs/1807.06667). Accel is an extension of [Deep Feature Flow](https://github.com/msracver/Deep-Feature-Flow), a video recognition framework released by MSR Asia in 2017.

Some notes:

* Accel combines optical-flow based keyframe feature warping (Deep Feature Flow) with per-frame temporal correction (DeepLab) in a score fusion step to improve on the accuracy of each of its constituent sub-networks.
* Accel is trained end-to-end on the task of video semantic segmentation.
* Accel can be built with a range of feature sub-networks, from ResNet-18 to ResNet-101. Accel based on ResNet-18 (Accel-18) is fast and reasonably accurate. Accel based on ResNet-101 (Accel-101) exceeds state-of-the-accuracy.
* Accel can be evaluated on sparsely annotated video recognition datasets, such as Cityscapes and CamVid.

***Click image to watch our demo video***

[![Demo Video on YouTube](https://media.giphy.com/media/14erFWP6f5tDVe/giphy.gif)](http://www.youtube.com/watch?v=J0rMHE6ehGw)
[![Demo Video on YouTube](https://media.giphy.com/media/xwB5LVfIjLtS/giphy.gif)](http://www.youtube.com/watch?v=J0rMHE6ehGw)

## Comments

This is an official implementation for [Accel](https://arxiv.org/abs/1807.06667) in MXNet. It is worth noting that:

  * The code is tested on official [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60).
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) model and [Flying Chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) pre-trained [FlowNet](https://lmb.informatik.uni-freiburg.de/resources/binaries/dispflownet/dispflownet-release-1.2.tar.gz) model using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository used code from [MXNet rcnn example](https://github.com/dmlc/mxnet/tree/master/example/rcnn) and [mx-rfcn](https://github.com/giorking/mx-rfcn).



## License

© Microsoft, 2018. Licensed under the [MIT](LICENSE) License.


## Citing Accel

If you find Accel useful in your research, please consider citing:
```
@inproceedings{jain19,
    Author = {Samvit Jain, Xin Wang, Joseph E. Gonzalez},
    Title = {Accel: A Corrective Fusion Network for Efficient Semantic Segmentation on Video},
    Conference = {CVPR},
    Year = {2019}
}

@inproceedings{zhu17dff,
    Author = {Xizhou Zhu, Yuwen Xiong, Jifeng Dai, Lu Yuan, Yichen Wei},
    Title = {Deep Feature Flow for Video Recognition},
    Conference = {CVPR},
    Year = {2017}
}
```


## Main Results

|                                 | <sub>training data</sub>     | <sub>testing data</sub> | <sub>mIoU</sub> | <sub>time/image</br> (Tesla K80)</sub> |
|---------------------------------|-------------------|--------------|---------|---------|
| <sub>Deep Feature Flow</br>(DeepLab, ResNet-v1-101, FlowNet)</sub>                    | <sub>Cityscapes train</sub> | <sub>Cityscapes val</sub> | 68.7    | 0.25s    |
| <sub>Accel-18</br>(DeepLab, ResNet-v1-18, FlowNet)</sub>           | <sub>Cityscapes train</sub> | <sub>Cityscapes val</sub> | 72.1   |  0.44s    |
| <sub>Accel-50</br>(DeepLab, ResNet-v1-34, FlowNet)</sub>           | <sub>Cityscapes train</sub> | <sub>Cityscapes val</sub> | 74.2   |  0.67s    |
| <sub>Frame-by-frame baseline</br>(DeepLab, ResNet-v1-101)</sub>                    | <sub>Cityscapes train</sub> | <sub>Cityscapes val</sub> | 75.2    | 0.74s    |
| <sub>Accel-101</br>(DeepLab, ResNet-v1-101, FlowNet)</sub>           | <sub>Cityscapes train</sub> | <sub>Cityscapes val</sub> | 75.5   |  0.87s    |

*Running time is benchmarked on a single GPU (mini-batch size 1, key-frame duration length 5).*


## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not suppoort Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.

3. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install Cython
	pip install opencv-python==3.2.0.6
	pip install easydict==1.6
	```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.


## Requirements: Hardware

Any NVIDIA GPUs with at least 6GB memory should be OK


## Installation

1. Clone the Accel repository. Let ${Accel_ROOT} denote the cloned repository. 

~~~
git clone https://github.com/SamvitJ/Accel.git
~~~
2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:

	3.1 Clone MXNet and checkout to [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 62ecb60
	git submodule update
	```
	3.2 Copy operators in `$(Accel_ROOT)/dff_rfcn/operator_cxx` or `$(Accel_ROOT)/rfcn/operator_cxx` to `$(YOUR_MXNET_FOLDER)/src/operator/contrib` by
	```
	cp -r $(Accel_ROOT)/dff_rfcn/operator_cxx/* $(MXNET_ROOT)/src/operator/contrib/
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j4
	```
	3.4 Install the MXNet Python binding by
	
	***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
	```
	cd python
	sudo python setup.py install
	```
	3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/dff_rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.


## Demo

1. To run the demo with our trained models, please download the following models, and place them under folder `model/`:
	- FlowNet model -- manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMPLjGGCvAeciQflg) (for users from Mainland China, please try [Baidu Yun](https://pan.baidu.com/s/1nuPULnj))
	- Accel models -- manually from [Google Drive]() [TODO: ADD LINK]

	Make sure it looks like this:
	```
	./model/rfcn_dff_flownet_vid-0000.params
	./model/accel-18-0000.params
	./model/accel-34-0000.params
	./model/accel-50-0000.params
	./model/accel-101-0000.params
	```
2. Run (default: keyframe interval 1, num examples 10)
	```
	python ./dff_deeplab/demo.py
	```
	or run (custom: keyframe interval X, num examples Y)
	```
	python ./dff_deeplab/demo.py --interval 5 --num_ex 100
	```


## Preparation for Training & Testing [UPDATE]

1. Please download ILSVRC2015 DET and ILSVRC2015 VID dataset, and make sure it looks like this:

	```
	./data/ILSVRC2015/
	./data/ILSVRC2015/Annotations/DET
	./data/ILSVRC2015/Annotations/VID
	./data/ILSVRC2015/Data/DET
	./data/ILSVRC2015/Data/VID
	./data/ILSVRC2015/ImageSets
	```

2. Please download ImageNet pre-trained ResNet-v1-101 model and Flying-Chairs pre-trained FlowNet model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMOBdCBiNaKbcjPrA) (for users from Mainland China, please try [Baidu Yun](https://pan.baidu.com/s/1nuPULnj)), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	./model/pretrained_model/flownet-0000.params
	```


## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/{dff_deeplab}/cfgs`.

2. Two baseline config files are provided: DeepLab frame-by-frame and Deep Feature Flow. We use 4 GPUs to train models on Cityscapes and CamVid.

3. To perform experiments, run the python script with the corresponding config file as input. For example, to train and test Accel with DeepLab, use the following command
    ```
    python experiments/dff_deeplab/dff_deeplab_end2end_train_test.py --cfg experiments/dff_deeplab/cfgs/resnet_v1_101_flownet_cityscapes_deeplab_end2end_ohem.yaml
    ```
	A cache folder will be created automatically to save the model and the log under `output/dff_deeplab/cityscapes/`.
    
4. Please find more details in config files and in our code.


## Misc.

Code has been tested under:

- Ubuntu 14.04 with a Maxwell Titan X GPU and Intel Xeon CPU E5-2620 v2 @ 2.10GHz
- Windows Server 2012 R2 with 8 K40 GPUs and Intel Xeon CPU E5-2650 v2 @ 2.60GHz
- Windows Server 2012 R2 with 4 Pascal Titan X GPUs and Intel Xeon CPU E5-2650 v4 @ 2.30GHz


## FAQ

For common errors, please see the [Deep Feature Flow FAQ](https://github.com/msracver/Deep-Feature-Flow#FAQ).
