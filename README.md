# MalConv-Pytorch
A Pytorch implementation of MalConv

---
## Desciprtion

This is the implementation of MalConv proposed in [Malware Detection by Eating a Whole EXE](https://arxiv.org/abs/1710.09435).

## Dependency

Please make sure each of them is installed with the correct version

- numpy
- pytorch (0.3.0.post4)
- pandas (0.20.3)


## Setup

#### Preparing data

For the training data, please place PE files under [`data/train/`](`data/train`) and build [the label table](data/example-train-label.csv) for training set with each row being

        <File Name>, <Label>

where label = 1 refers to malware. Validation set should be handled in the same way.

#### Training

Run the following command for training progress

        python3 train.py <config_file_path> <random_seed>
        Example : python3 train.py config/example.yaml 123

#### Training Log & Checkpoint

Log file, prediction on validation set & Model checkpoint will be stored at the path specified in config file.

## Parameters & Model Options

For parameters and options availible, please refer to [`config/example.yaml`](config/example.yaml).

# Improvement // 面向恶意代码对抗攻击的防御方法

原则：在防御的时候不会引入新的缺陷

## Positional Encoding

- block positional embedding // positional embedding // to avoid header injection
- noisy net/gan // 在head前添加noise，防止离群值的出现，有点类似于用GAN进行数据增强的意思了，但这里的gan只是用来进行防御的，gan可以帮助我们的检测器完善决策边界，但愿不会引入bias
// 对于比较离群的值，判定为阳性
- 问题在于，构造离群样本（image，纯code，纯random，都算离群样本，需要都判定为良性）,
- "是恶意代码还是良意代码" 转变为 "是不是恶意代码" // 从二分类的思维中跳出来，这是一个纯粹的检测任务
- 目的不是区分良意和恶意代码的特征区别，而是学习恶意代码的特征
- 恶意代码 or 良意代码 => 恶意+代码，其他任何东西（学到离群值）(良意代码，img, code, random, or their combination)

## 数据增强

- 为输入数据添加一个padding byte，对应的长度从1<<20到(1<<20)+500 or spp-net (消除padding的需要)
- randomly add [benign content/image/random/padding with one value] to malware overlay (avoid fuzzing) # force the detector to detect malicious block
- set None(257) for dos header & dos stub & section cave
