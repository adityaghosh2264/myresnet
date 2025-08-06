# myresnet
This is my implementation of ResNet from scratch in PyTorch and tested on the CIFAR-10 dataset. 

## ResNet-20
The following graph is without data augmentation and we can clearly see that the resnet overfits and the error stalls.
![image alt](https://github.com/adityaghosh2264/myresnet/blob/bb89165df1d345eefd696346fede40051cbd13c5/data/resnet20_error_curve(no_augmentation).png)


The following plot has been done with data augmentation as mentoned in the [paper](https://arxiv.org/abs/1512.03385).
![image alt](https://github.com/adityaghosh2264/myresnet/blob/bb89165df1d345eefd696346fede40051cbd13c5/data/resnet20_error_curve.png)


## ResNet-32
Below is the graph for ResNet-32 with data augmentation (The counterpart with no data augmentation was not trained because  similar results as that with less number of layers was expected)
![image alt](https://github.com/adityaghosh2264/myresnet/blob/bb89165df1d345eefd696346fede40051cbd13c5/data/resnet32_error_curve.png)
