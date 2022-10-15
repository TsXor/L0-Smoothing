<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">Image Smoothing via L0 Gradient Minimization</h3>

  <p align="center">
    <br />
    <a
    href="http://www.nthere.in/2020/06/15/Image-Smoothing-using-L0-Gradient-Minimization/">Blog
    Post</a>
    |
    <a href="https://github.com/nrupatunga/L0-Smoothing/issues">Report Bug on Numpy Version</a>
    |
    <a href="https://github.com/TsXor/L0-Smoothing/issues">Report Bug on PyOpenCL Version</a>
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
	- [Installation](#installation)
		- [Install from pypi](#install-from-pypi)
		- [Setup with source code](#setup-with-source-code)
	- [Usage](#usage)
		- [Import in your script](#import-in-your-script)
		- [Execute from terminal](#execute-from-terminal)
* [Maybe FAQ](#maybe-faq)

<!--ABOUT THE PROJECT-->
## About the Project

This repository is the Python implementation of the paper:
[Image Smoothing via L0 Gradient Minimization](http://www.cse.cuhk.edu.hk/~leojia/papers/L0smooth_Siggraph_Asia2011.pdf)

|Flower     |
|-----------|
|![](https://github.com/nrupatunga/L0-Smoothing/blob/master/src/output/flower.png) |

| Rock
|-------------|
|![](https://github.com/nrupatunga/L0-Smoothing/blob/master/src/output/rock2.png) |

<!--GETTING STARTED-->
## Getting Started  

<!--INSTALLATION-->
### Installation  

<!--INSTALL FROM PYPI-->
#### Install from pypi  
```bash
pip install L0-Smoothing
```

<!--SETUP WITH SOURCE CODE-->
#### Setup with source code  
```bash
# Clone the repository
git clone https://github.com/TsXor/L0-Smoothing.git

# build package
# build.bat is batch script for Windows cmd, and will not work on linux.
# build.bat consist of mostly commands to move files, so it is easy to be rewritten into a bash script.
# However, I don't have a linux machine available now, hope someone can write one and open a PR.
cd builder
./build.bat

# install via pip
cd dist
pip install L0_Smoothing-*-py3-none-any.whl
```

<!--USAGE-->
### Usage  

<!--IMPORT IN YOUR SCRIPT-->
#### Import in your script  
```python
from L0_Smoothing import L0_Smoothing, L0_Smoothing_accel

# It is not recommended to use cv2.imread because it easily throws error
# just because you are missing some unimportant parameters and cannot
# read image from path with Chinese (and maybe other non-ascii) characters.
# Just read it with PIL and convert it to numpy array!
# Note that you need to convert image to BGR with cv2.cvtColor if you read with PIL.
import numpy as np
from PIL import Image
img = np.asarray(Image.open(r'/path/to/your/image'))

# Parameters:
# L0_Smoothing(img, asHSV=False, lambda_=2e-2, kappa=2.0, beta_max=1e5, mode='pyvkfft')
# L0_Smoothing_accel(img, asHSV=False, lambda_=2e-2, kappa=2.0, beta_max=1e5)
#     img: numpy array of the image to be smoothed
#     asHSV: This module does operation per channel, and you can choose to convert it to HSV
#            while operating by giving parameter asHSV=True.
#     lambda_, kappa, beta_max: read the paper
#     mode: the OpenCL FFT backend to use
smoothed = L0_Smoothing_accel(img)
```
If you are programming with `pyopencl`, you can use this module like this:  
```python
import pyopencl as cl
import numpy as np
import pyopencl.array as clArray

from L0_Smoothing import L0_Smoothing_CL

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

img = np.asarray(Image.open(r'/path/to/your/image'))
S = clArray.to_device(queue, img/255)
S_smoothed = L0_Smoothing_CL(S)
img = S.get()*255
img = np.clip(img, 0, 255).astype(np.uint8)
```
Hint: You can try to apply blur before doing smoothing if the smoothing effect is not ideal.  

<!--EXECUTE FROM TERMINAL-->
#### Execute from terminal  
Notes:  
- It support only jpg and png images now.  
- Input and output path should be both file or both folder.  
- You can give lambda_, kappa, beta_max via `--params`.  
- You can choose to use slower numpy version with switch `--noaccel`.  
- If you give `show` for output path, processed image with not be saved but showed.  
```bash
# get some help
L0-Smoothing --help

# process single image and save it somewhere
L0-Smoothing /path/to/your/image /path/you/want/to/save

# process all images in a folder and save it somewhere
L0-Smoothing /path/to/your/image/folder /path/you/want/to/save
```

<!--MAYBE FAQ-->
## Maybe FAQ
- When I use its command from terminal, it throws error!  
  Try adding switch `--noaccel` to use slower numpy version.  
- pip throws errors (on compiling) when I am installing pyvkfft!  
  Download pyvkfft package bundled with OpenCL SDK from release and install it with pip.  
  Or just download binary package from release and install it.  
- Binary packages have problem on my machine.  
  Use `reikna` as fft backend, it is pure python.
