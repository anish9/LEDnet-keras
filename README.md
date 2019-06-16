# LEDnet-keras
#### Unofficial implementation of [LEDnet](https://arxiv.org/pdf/1905.02423.pdf) in tensorflow keras

### Block Diagram
![alt text](https://github.com/anish9/LEDnet-keras/blob/master/logs/2-Figure1-1.png)
> Design
![design](https://github.com/anish9/LEDnet-keras/blob/master/logs/3-Table1-1.png)

> Todo
- [x] Implement encoder and decoder block
- [x] Update pre-trained weights on Cityscapes or Mapillary dataset
- [ ] Implement shuffle blocks as per paper
- [ ] Use Focal loss functions on unbalanced segmentations to test the performance.(Not so good at present)
- [ ] experiment different dilations apart from official parameters to evaluate segmentation.

> Author
josh.anish1@gmail.com
