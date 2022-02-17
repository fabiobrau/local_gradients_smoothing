# Local Gradients Smoothing
Implements the Local Gradients Smoothing (LGS) technique for defense against adversarial patches as proposed in 
[Naseer, Muzammal, Salman Khan, and Fatih Porikli. "Local gradients smoothing: Defense against localized adversarial attacks." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.
](https://ieeexplore.ieee.org/iel7/8642793/8658235/08658401.pdf).

### Install 
```
$> git pull 
$> conda env create -n conda-env.yml 
```

### Usage
```
$> python3 -m "from local_gradients_smoothing import get_lgs_mask"
```
or 
```angular2html
>>> import sys
>>> sys.add('<path-of-local_gradients_smoothing>')
>>> from lgs import LocalGradientSmoothing # Is a class and needs parameters
>>> from lgs import get_lgs_mask # Is a function with defaults parameters
```
The test will provide the following image
![](example_result.jpg)