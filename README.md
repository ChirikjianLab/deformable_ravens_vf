**NOTE**: this code is based heavily on [the Ravens code base from Google][6]
and retains the same license.

**NOTE**: this code is based heavily on [the orignal DeformableRavens repo](https://github.com/DanielTakeshi/deformable-ravens).

# DeformableRavens-VF

Code for training the Goal-Conditioned Transporter Network (GCTN) of the multi-modal action proposal of Transporters with Visual Foresight (TVF).
The main repository for TVF is [ravens_visual_foresight](https://github.com/ChirikjianLab/ravens_visual_foresight). 
It also contains the data and pretrained models.
This code has been tested on Ubuntu 20.04 and Python 3.8.
If you have any questions, please use the issue tracker.

[TVF repo](https://github.com/ChirikjianLab/ravens_visual_foresight) || [Paper Link](https://arxiv.org/abs/2202.10765) || [Project Website](https://chirikjianlab.github.io/tvf/)


## Installation
```shell
./install_python_ubuntu.sh
```

## Usage
To train the GCTN for multiple tasks
```shell
python train_multi_task.py --data_dir=ravens_visual_foresight/data_train --models_dir=ravens_visual_foresight/gctn_models --num_demos=10 --num_runs=1
```
`--data_dir` specifies the directory of the training data.
`--models_dir` specifies the directory to save the trained models.
`--num_demos` specifies the number of demos per training task used for training.
`--num_runs` specifies the number of training runs.

## Related Work

- [Learning to Rearrange Deformable Cables, Fabrics, and Bags with Goal-Conditioned Transporter Networks](https://arxiv.org/abs/2012.03385)

- [Transporter Networks: Rearranging the Visual World for Robotic Manipulation](https://arxiv.org/abs/2010.14406)

[1]:https://www.tensorflow.org/hub/installation
[2]:https://github.com/tensorflow/addons/issues/1132
[3]:https://partner-code.googlesource.com/project-reach/+/75459a560ea9ae4b9d7283ef39d4a4d99598ab81
[4]:https://stackoverflow.com/a/56537286/3287820
[5]:https://berkeleyautomation.github.io/bags/
[6]:https://github.com/google-research/ravens
[7]:https://github.com/DanielTakeshi/deformable-ravens