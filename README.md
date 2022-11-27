## NeuralArt

Implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Tensorflow.

### Requirements
 - [Tensorflow](http://www.tensorflow.org/)
 - [VGG 19 model](https://drive.google.com/file/d/0B8QJdgMvQDrVU2cyZjFKU1RrLUU/view?usp=sharing)

### Examples

<p>
Content: <br/>
<img src="https://github.com/ckmarkoh/neuralart_tensorflow/blob/master/images/Taipei101.jpg?raw=true" width="50%"/> <br/>
Style: <br/>
<img src="https://github.com/ckmarkoh/neuralart_tensorflow/blob/master/images/StarryNight.jpg?raw=true" width="50%"/> <br/>
Output: <br/>
<img src="https://github.com/ckmarkoh/neuralart_tensorflow/blob/master/images/Taipei101_StarryNight.jpg?raw=true" width="50%"/> <br/>
</p>

### install

1. Download Python2

2. Download tensorflow <https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.10.0/py27/CPU/avx2/tensorflow-1.10.0-cp27-cp27m-win_amd64.whl>

3. Download imagenet-vgg-verydeep-19.mat <https://drive.google.com/file/d/0B8QJdgMvQDrVU2cyZjFKU1RrLUU/view?usp=sharing> and put imagenet-vgg-verydeep-19.mat in the same dir with main.py

4. run

```sh
pip install keras
pip uninstall scipy
pip install scipy==1.2.1
pip install imageio==2.6.1
```
