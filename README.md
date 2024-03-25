# STBMP

# Framework
![STBMP](https://github.com/JasonWang959/STPMP/blob/main/image/pipeline.png)

# Dependencies
![python = 3.8](https://img.shields.io/badge/python-3.8.13-green)
![torch = 1.10.0+cu111](https://img.shields.io/badge/torch-1.10.0%2Bcu111-yellowgreen)

## DataSet
[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```
[CMU mocap](http://mocap.cs.cmu.edu/) was obtained from the [repo](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics) of ConvSeq2Seq paper.
Directory structure: 
```shell script
cmu
|   |-- ...
|-- train
`-- test
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.
Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
# Training commands

All the running args are defined in opt.py. We use following commands to train on different datasets and representations.

+ Train on Human3.6M:

To train a model of short-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m_3d.py --t_input_size=66  --s_input_size=20 --input_n=10 --output_n=10 --dct_n=20  --is_load=False 
```
To train a model of long-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m_3d.py --t_input_size=66  --s_input_size=50 --input_n=25 --output_n=25 --dct_n=50  --is_load=False
```

+ Train on CMU-MoCap:
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_cmu_3d.py --t_input_size=75  --s_input_size=35 --input_n=10 --output_n=25 --dct_n=35  --is_load=False
```
  
+ Train on 3DPW:
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_3dpw_3d.py --t_input_size=69  --s_input_size=40 --input_n=10 --output_n=30 --dct_n=40  --is_load=False
```

### Acknowledgments
 
Some of our code was adapted from [SPGSN](https://github.com/MediaBrain-SJTU/SPGSN).

## Licence
This project is licensed under the terms of the MIT license.

This readme file is going to be further updated.
