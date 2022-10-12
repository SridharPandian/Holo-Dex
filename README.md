# Holo-Dex: Teaching Dexterity with Immersive Mixed Reality

**Authors**: [Sridhar Pandian Arunachalam](https://sridharpandian.github.io), [Irmak Guzey](https://www.linkedin.com/in/irmak-guzey-6a9010175/), [Soumith Chintala](https://soumith.ch/), [Lerrel Pinto](https://lerrelpinto.com)

This repository contains the official implementation of [Holo-Dex](holo-dex.github.io) including the unity scripts for the VR application, robot demonstration collection pipeline, training scripts and deployment modules. The VR application's APK can be found [here](https://github.com/SridharPandian/Holo-Dex/releases/tag/VR).

## Robot Runs
<p align="center">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/planar-rotation.gif">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/object-flipping.gif">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/can-spinning.gif">
 </p>

 <p align="center">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/bottle-opening.gif">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/card-sliding.gif">
  <img width="30%" src="https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/main_task/postit-note-sliding.gif">
 </p>

 ## Method
![Holo-Dex](https://github.com/holo-dex/holo-dex.github.io/blob/website/mfiles/Intro.png)
Holo-Dex consists of two phases: demonstration colleciton, which is performed in real-time with visual feedback to VR Headset, and demonstration-based policy learning, which can learn to solve dexterous tasks from a limited number of demonstrations.

## Pipeline Installation and Setup
The pipeline requires [ROS](http://wiki.ros.org/noetic/Installation/Ubuntu) for Server-Robot communication. This Package uses the Allegro Hand and Kinova Arm controllers from [DIME-Controllers](https://github.com/NYU-robot-learning/DIME-Controllers). This implementation uses Realsense cameras and which require the [`librealsense`](https://github.com/IntelRealSense/librealsense#installation-guide) API. Also, [OpenCV](https://pypi.org/project/opencv-python/) is required for image compression and other purposes.

After installing all the prerequisites, you can install this pipeline as a package with pip:
```
pip install -e .
```

You can test if it has installed correctly by running `import holodex` from a python shell.

## Running the teleop
To use the Holo-Dex teleop module, open the VR Application in your Oculus Headset and enter the robot server's IP address (should be in the same network). Green Stream border indicates that the right hand keypoints are being streamed and blue indicates left. Red indicates that the stream is paused and the menu can be accessed only when paused.

On the robot server side, start the [controllers](https://github.com/NYU-robot-learning/DIME-Controllers) first followed by the following command to start the teleop:
```
python teleop.py
```
The Holo-Dex teleop configurations can be adjusted in `configs/tracker/oculus.yaml`. The robot camera configurations can be adjusted in `configs/robot_camera.yaml`.

The package also contains an 30 Hz teleop implementation of [DIME](https://arxiv.org/abs/2203.13251) and you can run it using the following command:
```
python teleop.py tracker=mediapipe tracker/cam_serial_num=<realsense_camera_serial_number>
``` 

## Data
All our data can be found in this URL: [https://drive.google.com/](https://drive.google.com/)

To collect demonstrations using this framework, run the following command:
```
python record_data.py -n <demonstration_number>
```

To filter and process data from the raw demonstrations run the following command:
```
python extract_data.py
```
You can change the data extraction configurations in `configs/demo_extract.yaml`.

## Training Neural Networks
You can train encoders using Self-Supervised methods such as [BYOL](https://arxiv.org/abs/2006.07733), [VICReg](https://arxiv.org/abs/2105.04906), [SimCLR](https://arxiv.org/abs/2002.05709) and [MoCo](https://arxiv.org/abs/2104.02057). Use the following command to train a resnet encoder using the above mentioned SSL methods:
```
python train_ssl.py ssl_method=<byol|vicreg|simclr|mocov3>
```
The training configurations can be changed in `configs/train_ssl.yaml`. 

You can also train:
- Behavior Cloning:
```
python train_bc.py encoder_gradient=true
```
- [Behavior Cloning-Rep](https://arxiv.org/abs/2008.04899):
```
python train_bc.py encoder_gradient=false
```
The training configurations can be changed in `configs/train_bc.yaml`.

## Deploying Models
This implementation can deploy BC, BC-Rep and INN (all visual) on the robot. To deploy BC or BC-Rep, use the following command:
```
python deploy.py model=BC task/bc.model_weights=<bc_model-weights>
```

To deploy INN use the following command:
```
python deploy.py model=VINN task/vinn.encoder_weights_path=<ssl_encoder_weights_path>
```

You can set a control loop instead of pressing the `Enter` key to get actions using the following command:
```
python deploy.py model=<BC|VINN> run_loop=true loop_frequency=<action_loop_frequency>
```

## Customizing the VR Application
To use Holo-Dex's VR side source code `vr/Holo-Dex`, you need to install Unity along with other dependencies which can be download through Nuget:
- Oculus SDK
- NetMQ
- TextMeshPro

## Citation

If you use this repo in your research, please consider citing the paper as follows:
```
@article{arunachalam2022dime,
  title={Holo-Dex: Teaching Dexterity with Immersive Mixed Reality},
  author={Sridhar Pandian Arunachalam and Irmak Guzey and Soumith Chintala and Lerrel Pinto},
  journal={arXiv preprint arXiv:},
  year={2022}
}