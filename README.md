# Wrinkle-Detector

## Brief introduction

This is a Wrinkle Detector project which is mainly based on the Digital Image Processing course by using three classic low-pass filter

## Preparation for the project

Before the formal using, please check if the following function libraries are installed.

- `cv2`
- `numpy`
- `pillow`

If not, please follow the instruction as below.

```shell
pip install opencv-python
pip install numpy
pip install pillow
```

Fully believe that you, the intelligent one, can definitely complete these initial steps.

## File introduction

The files I uploaded are `final_gui.py`,`test_gui.py`,`test_autodetect.py` and a gallery of images prepared in advance for you.

`final_gui.py` is the final project, which you can execute directly to perform wrinkle filtering, while `test_gui.py` and `test_autodetect.py` are meant to help everyone understand the entire project step by step.

- `test_gui.py`
  This file aims to design the whole frame of GUI. What' more, it also contains the three kinds of classical low-pass filters. You can run it and call the image file to test it.

- `test_autodetect.py`
  This file is mainly used to design the automatic wrinkle detection algorithm.

- `final_gui.py`
  This is the combination of the two file I mentioned above. You can select both modes under this GUI interface

## Algorithm introduce

Maybe you are curious about the algorithmic principle. Actually, I utilized traditional sobel edge detection as well as haar classifier for automatic wrinkle recognition. 

## Evaluation

The overall image processing flow should be to first process the image in 'automatic' mode to achieve initial processing, and then switch to 'manual' mode to manually process some less obvious details.

Such a design is actually like PS, for the little white people (such as me) can directly use the magic wand to achieve automatic keying, while some professional for some people can manually key their own to optimize the effect.

The images I've provided are basically the initial filtering that can be achieved through 'Automatic' mode. However, for some images that are not clear, the processing effect of 'Automatic' mode will become much worse