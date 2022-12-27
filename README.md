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
where image_path is the path of the test images, test_path is the mask and result is the save-path of results.

### Acknowledgement

We benifit a lot from [NVIDIA-partialconv](https://github.com/NVIDIA/partialconv) and [GlobalLocalImageCompletion_TF](https://github.com/shinseung428/GlobalLocalImageCompletion_TF), thanks for their excellent work.

