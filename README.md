# LEDnet-keras
#### Unofficial implementation of [LEDnet](https://arxiv.org/pdf/1905.02423.pdf) in tensorflow keras

### Block Diagram
![alt text](https://github.com/anish9/LEDnet-keras/blob/master/logs/2-Figure1-1.png)
> Design
![alt_text](https://github.com/anish9/LEDnet-keras/blob/master/logs/3-Table1-1.png)

### Requirements
Python3.6
Tensorflow 1.13
opencv 
albumenations

## Custom Training
```
data_dir =  "/home/anish/data_dir/" #Root training and Val data dir 
train_image = "train_images"
val_images = "val_images"
train_masks = "train_masks"
val_masks = "val_masks"

#do not change below directory to avoid directory conflicts 
#original input dimension in paper : 1024x512

model_param = {"image_size" : (512,512),
			   "train_batch_size" : 2,
			   "val_batch_size" : 2,
			   "augument":False
			   }
```


> Todo
- [x] Implement encoder and decoder block
- [x] Update pre-trained weights on Cityscapes or Mapillary dataset
- [ ] Implement shuffle blocks as per paper
- [ ] Use Focal loss functions on unbalanced segmentations to test the performance.(Not so good at present)
- [ ] experiment different dilations apart from official parameters to evaluate segmentation.

> Author :
josh.anish1@gmail.com
