import time, random
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
lr_init = 1e-4
beta1 = 0.9
beta2 = 0.999
n_epoch = 2000
lr_decay = 0.1
decay_every = int(n_epoch / 2)

train_hr_img_path = './DIV2K_train_HR/'
# train_lr_img_path = './DIV2K_train_LR_bicubic/X4/'
valid_hr_img_path = './Set5/'
# valid_lr_img_path = './DIV2K_valid_LR_bicubic/X4/'
checkpoint_dir = "checkpoint"

def EDSR(t_image, B=32, F=256, res_scale=0.1 ,reuse=False):
    ''' 
    ref paper:
        Enhanced Deep Residual Networks for Single Image Suepr-Resolution

    Note: this model adopt residual scaling to gurantee the numerically unstable 
    The code is defualt for upscaling x4 
    '''
    w_init = tf.random_normal_initializer(stddev=0.02)
    with tf.variable_scope("EDSR", reuse=reuse):
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
    with tf.variable_scope("EDSR_baseline", reuse=reuse):
        tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
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



if __name__ == "__main__":

    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_hr_img_list = sorted(tl.files.load_file_list(path=train_hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=train_lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=valid_hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=valid_lr_img_path, regx='.*.png', printable=False))
    
    ## Read All valid dataSet into memory to accelerate evaluation
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=valid_hr_img_path)

    t_image = tf.placeholder('float32', [batch_size, 48, 48, 3], name="t_image_LR_input")
    t_target_image = tf.placeholder('float32', [batch_size, 192, 192, 3], name="t_target_image_HR_output")
    t_valid_image = tf.placeholder('float32', [1, None, None, 3], name="valid_image_LR_input")

    net = EDSR_baseline(t_image, reuse=False)
    net_test = EDSR_baseline(t_valid_image, reuse=True)

    ## mse loss
    # loss = tf.reduce_mean(tf.square(net.outputs - t_target_image), name="mse")
    ## L1 loss
    loss = tf.reduce_mean(tf.abs(net.outputs - t_target_image), name="L1_loss")
    loss_summary = tf.summary.scalar("L1_loss", loss)

    lr_v = tf.Variable(lr_init, trainable=False)
    net_vars = tl.layers.get_variables_with_name('EDSR_baseline', True, True)
    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(loss, var_list=net_vars)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(lr_v, lr_init))
    summary_writer = tf.summary.FileWriter("edsr_log", sess.graph)

    
    print("Start training...")
    for epoch in range(0, n_epoch+1):
        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        random.shuffle(train_hr_img_list)
        for idx in range(0, len(train_hr_img_list), batch_size):
            step_time = time.time()
            cur_hr_imgs_list = train_hr_img_list[idx:idx+batch_size]
            # cur_hr_imgs = tl.prepro.threading_data(cur_hr_imgs_list, fn=get_imgs_fn, path=train_hr_img_path)
            cur_hr_imgs = tl.vis.read_images(cur_hr_imgs_list, path=train_hr_img_path, n_threads=batch_size)
            ## smaple HR/LR image path pairs
            sample_hr = tl.prepro.threading_data(cur_hr_imgs, fn=crop_sub_imgs_fn, is_random=True)
            sample_lr = tl.prepro.threading_data(sample_hr, fn=downsample_fn)
            ## scale image to 0-1
            sample_hr = tl.prepro.threading_data(sample_hr, fn=rescale_0_1_fn)
            sample_lr = tl.prepro.threading_data(sample_lr, fn=rescale_0_1_fn)

            err, l1_loss_summary, _ = sess.run([loss, loss_summary, optim], feed_dict={t_image: sample_lr, t_target_image: sample_hr}) 
            print("Epoch [%2d/%2d] %2d time: %2.2fs, mse: %.8f " % (epoch, n_epoch, n_iter, time.time() - step_time, err))
            total_loss += err
            n_iter +=1

            summary_writer.add_summary(l1_loss_summary, epoch*len(train_hr_img_list)/batch_size+n_iter)
        
        log = "[*] Epoch: [%2d/%2d] time: %2.2fs, loss: %.8f " % (epoch, n_epoch, time.time() - epoch_time, total_loss / n_iter)
        print(log)

        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net.all_params, name=checkpoint_dir+'/edsrNet.npz')

        if(epoch % 5 == 0):
            num = len(valid_hr_img_list)
            total_psnr = 0
            for i in range(num):
                ## alignment for upscaling factor
                valid_hr_imgs[i] = set_image_alignment(valid_hr_imgs[i])
                valid_lr_img = rescale_0_1_fn(downsample(valid_hr_imgs[i]))[np.newaxis,:,:,:]
                out = sess.run(net_test.outputs, feed_dict={t_valid_image: valid_lr_img})
                ## rescale image to 0-255
                sr_img = np.uint8(out[0]*255)
                sr_img_y = convert_rgb_to_y(sr_img)
                hr_img_y = convert_rgb_to_y(valid_hr_imgs[i])

                psnr_val = tf.image.psnr(sr_img_y, hr_img_y, max_val=255)
                total_psnr += sess.run(psnr_val)

            psnr_summary = tf.Summary(value=[tf.Summary.Value(tag="psnr", simple_value=total_psnr/num)]) 
            summary_writer.add_summary(psnr_summary, epoch/5)
            log = "[*] Epoch: [%2d/%2d]  psnr: %.6fdB " % (epoch, n_epoch, total_psnr/num)
            print(log)