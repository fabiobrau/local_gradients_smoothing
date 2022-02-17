# Local Gradients Smoothing
Implements the Local Gradients Smoothing (LGS) technique for defense against adversarial patches as proposed in 
[Naseer, Muzammal, Salman Khan, and Fatih Porikli. "Local gradients smoothing: Defense against localized adversarial attacks." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.
](https://ieeexplore.ieee.org/iel7/8642793/8658235/08658401.pdf)

### Install 
```
$> git pull 
$> conda env create -n conda-env.yml 
```

### Usage

Clone this repository
```
$> git clone https://github.com/fabiobrau/local_gradients_smoothing <folder>
```
Be sure that the repository directory is in your `PYTHONPATH`.
```angular2html
$> export PYTHONPATH="$PYTHONPATH:<folder>/local_gradients_smoothing"
```
### Example
The following lines generate a mask by using the parameters of the original paper
```angular2html
$> python
>>> from lgs import get_lgs_mask # Is a function with defaults parameters
>>> import torch
>>> img = torch.randn(1,3,1000,1000)
>>> mask = get_lgs_mask(img)
```
By running ```$> python test_lgs.py``` you will obtain the following result
![](example_result.jpg)