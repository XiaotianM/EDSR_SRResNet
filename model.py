import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def EDSR_baseline(t_image, B=16, F=64, reuse=False):
    ''' 
    ref paper:
        Enhanced Deep Residual Networks for Single Image Suepr-Resolution
        
    Note: this model is a baseline for EDSR, without residual scaling factor
        The code is defualt for upscaling x4
        For CNN model, there is no tanh activation in the end  
        Here do not use relu outside residual block
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope("EDSR_baseline_g", reuse=reuse):
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name="in")
        n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="conv1")
        temp = n

        for i in range(1,B+1):
            with tf.variable_scope("ResBlock_%s"%i):
                nn = Conv2d(n, F, (3,3), (1,1), act=tf.nn.relu, W_init=w_init, name="conv1")
                nn = Conv2d(nn, F, (3,3), (1,1), act=None, W_init=w_init, name="conv2")
                n = ElementwiseLayer([n,nn], tf.add, name="residual_add")
        
        n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="conv2")
        n = ElementwiseLayer([n,temp], tf.add, name="skip_add")

        with tf.variable_scope("upsample"):
            n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name='conv1')
            n = SubpixelConv2d(n, 2, name="shuffle1")
            n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="conv2")
            n = SubpixelConv2d(n, 2, name="shuffle2") 
        
        n = Conv2d(n, 3, (3,3), (1,1), act=None, W_init=w_init, name="Conv3")
        return n


def EDSR(t_image, B=32, F=256, res_scale=0.1 ,reuse=False):
    ''' 
    ref paper:
        Enhanced Deep Residual Networks for Single Image Suepr-Resolution

    Note: this model adopt residual scaling to gurantee the numerically unstable 
    The code is defualt for upscaling x4 
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope("EDSR_g", reuse=reuse):
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name="in")
        n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="CONV1")
        temp = n

        for i in range(1,B+1):
            with tf.variable_scope("ResBlock_%s"%i):
                nn = Conv2d(n, F, (3,3), (1,1), act=tf.nn.relu, W_init=w_init, name="conv1")
                nn = Conv2d(nn, F, (3,3), (1,1), act=None, W_init=w_init, name="conv2")
                nn.outputs = tf.scalar_mul(res_scale, nn.outputs)
                n = ElementwiseLayer([n,nn], tf.add, name="residual_add")
        
        n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="CONV2")
        n = ElementwiseLayer([n,temp], tf.add, name="skip_add")

        with tf.variable_scope("upsample"):
            n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name='conv1')
            n = SubpixelConv2d(n, 2, name="shuffle1")
            n = Conv2d(n, F, (3,3), (1,1), act=None, W_init=w_init, name="conv2")
            n = SubpixelConv2d(n, 2, name="shuffle2") 
        
        n = Conv2d(n, 3, (3,3), (1,1), act=None, W_init=w_init, name="CONV3")
        return n


def SRResNet(t_image, is_train=False, reuse=False):
    """ 
    ref code:  https://github.com/tensorlayer/srgan
    """
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse):
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name="in")
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, act=None, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=None, W_init=w_init, name='out')
        return n