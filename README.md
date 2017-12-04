## Project: Follow Me
____

The objective of this project was to train a fully convolutional network to perform semantic segmentation and allow for target tracking on a quadcopter.
Final average IoU obtained after 22 epochs of training is 0.475.
#### Structure of the project:

All the files and their content is described below:
- [*writeup.md*](./writeup.md): Notebook containing the network architecture and used for training.

- [*code/model_training.ipnb*](./code/model_training.ipynb): Notebook containing the network architecture and used for training.

- [*code/data_augmentation.ipnb*](./code/data_augmentation.ipynb): Contains code used to perform horizontal flipping on the images and double the dataset size.

- [*code/keras_viz_dependencies.txt*](./code/keras_viz_dependencies.txt): Requirement for keras.utils.vis_utils.plot_model.
- [*docs/misc/model_architecture.png*](./docs/misc/model_architecture.png): Picture of the network architecture.
____



##### Network Architecture

We will define blocks using the following nomenclature: BLOCK(kernelWxkernelH, Stride, depth) when relevant.
Our network is composed of an assembly of 5 main building blocks:

- encoder blocks : __ENCODER(depth)__=[ SEPARABLE_CONV2D(3x3, 1) -> RELU -> BATCHNORM ]

- decoder blocks : __DECODER(concat, depth)__=[ BILINEAR_UPSAMPLE(2x2) -> CONCATENATE(concat) -> [ SEPARABLE_CONV2D(3x3, 1) -> RELU -> BATCHNORM ] * 2 ]

- downsampling blocks : __DOWNSAMPLE__=[MAXPOOL(2x2, 2)] OR [ENCODER(2x2, 2)]

- dropout blocks : __DROP__=[SPATIAL_DROPOUT2D]

- 1x1 convolutional block : __1x1CONV2D(depth)__=[CONV2D(1x1, 1) -> RELU -> BATCHNORM]


The picture of the final architecture is available [*here*](./docs/misc/model_architecture.png). This architecture is inspired from SegNet.

The full architecture of the final model using the previous notation is as follow:

[INPUT] -> [ENCODER(32) -> DOWNSAMPLE -> DROP] -> [ENCODER(64) -> DOWNSAMPLE -> DROP] -> [ENCODER(128) -> DOWNSAMPLE -> DROP] -> [1x1CONV2D(256)] -> [DECODER(encoder128, 128) -> DROP] -> [DECODER(encoder64, 64) -> DROP] -> [DECODER(encoder32, 32) -> DROP] -> [OUTPUT] 

Note that we used MAXPOOL for all the final model downsampling.


    Total params: 142,814
    Trainable params: 140,958
    Non-trainable params: 1,856

##### Results:
Evaluation Set size:



<table class="tg">
  <tr>
    <th class="tg"></th>
    <th class="tg">following</th>
    <th class="tg">not visible</th>
    <th class="tg">far away</th>
  </tr>
  <tr>
    <td class="tg">number of sample</td>
    <td class="tg">542</td>
    <td class="tg">270</td>
    <td class="tg">322</td>
  </tr>
</table>


Average IoU:



<table class="tg">
  <tr>
    <th class="tg">aIoU for \ situation</th>
    <th class="tg">following</th>
    <th class="tg">not visible</th>
    <th class="tg">far away</th>
  </tr>
  <tr>
    <td class="tg">background</td>
    <td class="tg">0.996124</td>
    <td class="tg">0.989919</td>
    <td class="tg">0.997138</td>
  </tr>
  <tr>
    <td class="tg">people</td>
    <td class="tg">0.423597</td>
    <td class="tg">0.793352</td>
    <td class="tg">0.51175</td>
  </tr>
  <tr>
    <td class="tg">hero</td>
    <td class="tg">0.92578</td>
    <td class="tg">0</td>
    <td class="tg">0.310087</td>
  </tr>
</table>


Confusion Table:


<table class="tg">
  <tr>
    <th class="tg">confusion \<br>  situation</th>
    <th class="tg">following</th>
    <th class="tg">not visible</th>
    <th class="tg">far away</th>
  </tr>
  <tr>
    <td class="tg">true positive</td>
    <td class="tg">539</td>
    <td class="tg">0</td>
    <td class="tg">155</td>
  </tr>
  <tr>
    <td class="tg">false positive</td>
    <td class="tg">0</td>
    <td class="tg">60</td>
    <td class="tg">2</td>
  </tr>
  <tr>
    <td class="tg">false negatve</td>
    <td class="tg">0</td>
    <td class="tg">0</td>
    <td class="tg">146</td>
  </tr>
</table>


Example predictions:


![](./docs/misc/example_1.png)
![](./docs/misc/example_2.png)
![](./docs/misc/example_3.png)
_left_: input image, _middle_: target mask, _right_: output mask.

