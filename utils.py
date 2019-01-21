import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_imgs_Ycbcr_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float) Ycbcr
    return scipy.misc.imread(path + file_name, mode='YCbCr')

def get_imgs_Y_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float) Y
    # the format of return [width, height, 1]
    return scipy.misc.imread(path + file_name, mode='YCbCr')[:,:,:1]


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=192, hrg=192, is_random=is_random)
    return x


def downsample_fn(x):
    ## We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[48, 48], interp='bicubic', mode=None)
    return x

def downsample(x, factor=4):
    ## this function is for validation 
    w,h,_ = [int(i/factor) for i in np.shape(x)]    
    x = imresize(x, size=[w, h], interp='bicubic', mode=None)
    return x


def rescale_0_1_fn(x):
    ## rescale to 0-1
	x = x / 255. 
	return x


def rescale_fn(x):
    ## rescale to -1~1
	x = x / (255. / 2.)
	x = x - 1
	return x


def set_image_alignment(image, alignment=4):
    ## here alignment = factor
    ## To alignment for facor
	alignment = int(alignment)
	width, height = image.shape[1], image.shape[0]
	width = (width // alignment) * alignment
	height = (height // alignment) * alignment

	if image.shape[1] != width or image.shape[0] != height:
		image = image[:height, :width, :]

	if len(image.shape) >= 3 and image.shape[2] >= 4:
		image = image[:, :, 0:3]

	return image


def convert_rgb_to_y(image):
    ## this function will return [width, height, 1]
	if len(image.shape) <= 2 or image.shape[2] == 1:
		return image

	xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
	y_image = image.dot(xform.T) + 16.0

	return y_image