# High-practicability Image Completion using Attention Mechanism and Joint Enhancive Discriminator

Tensorflow implementation Image Completion on public dataset.  

## Requirements

* Python 3
* Opencv 3.4
* Tensorflow 1.4.0
* Numpy 1.15

## Folder Setting

```
-data
    -img1.jpg
    -img2.jpg
    -...
-mask 
    -mask1.jpg
    -mask2.jpg
    -...
-testdata
    -img1.jpg
    -img2.jpg
    -...
```

## Train

To start train

```
$ python train.py
```

To continue training  

```
$ python train.py --continue_training=True
```

## Test  

To start test 

```
$ python test.py
```

## Notes

For the moment, we only upload the framework in network.py, neural network layers in architecture.py, parameter setting in config.py and other files. When this paper is accepted and published, we will uploaded the complete code including train.py and test.py.

### Acknowledgement

We benifit a lot from [NVIDIA-partialconv](https://github.com/NVIDIA/partialconv) and [GlobalLocalImageCompletion_TF](https://github.com/shinseung428/GlobalLocalImageCompletion_TF), thanks for their excellent work.

