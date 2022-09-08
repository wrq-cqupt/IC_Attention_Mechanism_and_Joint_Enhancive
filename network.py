import tensorflow as tf
#import cv2
import numpy as np
import math
import ops
import architecture
import Cloud_Class



class network():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim 
        self.k = args.k
        self.input_width = args.input_width
        self.input_height = args.input_height

        self.local_width, self.local_height = args.local_input_width, args.local_input_height

        self.m = args.margin

        self.alpha = args.alpha
        
        self.muti_count = tf.placeholder(tf.float32, (args.k,))

        self.real_img, self.perturbed_img, self.mask, self.data_count, self.mask_count = ops.load_train_data(args)
        self.attention_o = -(self.mask - 1)

        self.cloud_x = tf.placeholder(tf.float32, (args.batch_size, args.input_height*args.input_width))
        self.cloud_y = tf.placeholder(tf.float32, (args.batch_size, args.N_range))

        self.single_orig = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))
        self.single_test = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))
        self.single_mask = tf.placeholder(tf.float32, (args.batch_size, args.input_height, args.input_width, 3))

        self.build_model(args)
        self.build_loss()

        self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss)
        self.c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)
        self.p_loss_sum = tf.summary.scalar("p_loss", self.p_loss)
        self.all_loss_sum = tf.summary.scalar("all_loss", self.all_loss)
        
        self.input_img_sum = tf.summary.image("input_img", self.perturbed_img, max_outputs=5)
        self.real_img_sum = tf.summary.image("real_img", self.real_img, max_outputs=5)        
        self.generate_img_sum = tf.summary.image("generate_img", self.generate_img, max_outputs=5)
        self.recon_img_sum = tf.summary.image("recon_img", self.recon_img, max_outputs=5)
        self.mask_sum = tf.summary.image("mask", self.mask, max_outputs=5)

    def build_model(self,args):
        def rand_crop(img, coord, pads):
          cropped = tf.image.resize_images(tf.image.crop_to_bounding_box(img, coord[0]-self.m, coord[1]-self.m, pads[0]+self.m*2, pads[1]+self.m*2), (self.local_height, self.local_width))
          return cropped

        self.generate_img, self.perceptual_g, self.A, self.g_nets = self.generator(args, self.perturbed_img, self.attention_o, name="generator")          
        self.recon_img = self.mask*self.generate_img + (1-self.mask)*self.real_img

        self.cal_gray()
        self.cal_fre()

        self.test_g_imgs, _, _, _ = self.generator(args,self.single_test, self.single_mask, name="generator", reuse=True)
        self.test_res_imgs = (1-self.single_mask)*self.test_g_imgs + self.single_mask*self.single_orig

        self.content_f = tf.zeros([self.batch_size,1], dtype=tf.float32)
        self.content_r = tf.ones([self.batch_size,1], dtype=tf.float32)
        
        
        print(self.generate_img,self.real_img)
        self.fake_c_v, self.global_fake_d_net = self.content_discriminator(self.generate_img, name="content_discriminator")
        self.real_c_v, self.global_real_d_net = self.content_discriminator(self.real_img, name="content_discriminator", reuse=True)       

        self.fake_p_v, _, self.global_fake_p_net = self.partial_discriminator(self.generate_img, self.mask, name="partial_discriminator")        
        self.real_p_v, _, self.global_real_p_net = self.partial_discriminator(self.generate_img, self.mask, name="partial_discriminator", reuse=True)

        trainable_vars = tf.trainable_variables()
        self.g_vars = []
        self.c_vars = []
        self.p_vars = []
        

        for var in trainable_vars:
            if "generator" in var.name:
                self.g_vars.append(var)
            elif "content_discriminator" in var.name:
                self.c_vars.append(var)
            elif "partial_discriminator" in var.name:
                self.p_vars.append(var)
        
        print('vars',len(self.g_vars), len(self.c_vars), len(self.p_vars))
                
        self.c_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.c_vars]
        self.p_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.p_vars]        

    def build_loss(self):
        
        self.c_loss = tf.losses.mean_squared_error(self.fake_c_v, self.content_f) + tf.losses.mean_squared_error(self.real_c_v, self.content_r)
        
        self.p_loss = tf.losses.mean_squared_error(self.fake_p_v, self.content_f) + tf.losses.mean_squared_error(self.real_p_v, self.content_r)
        
        self.recon_loss = tf.losses.mean_squared_error(self.real_img, self.generate_img)# + self.perceptual_loss

        self.all_loss = tf.losses.mean_squared_error(self.fake_c_v, self.content_r) + 15*tf.losses.mean_squared_error(self.fake_p_v, self.content_r) + self.recon_loss #tf.reduce_mean(tf.nn.l2_loss(self.fake_p_logits-self.pixel_l_true)) + tf.reduce_mean(tf.nn.l2_loss(self.fake_c_v - self.content_r)) + self.recon_loss * 0.001

        self.muti_loss = self.recon_loss + self.muti_count[1]#

    def cal_gray(self):
        self.gray_R = self.generate_img[:,:,:,0:1]
        self.gray_G = self.generate_img[:,:,:,1:2]
        self.gray_B = self.generate_img[:,:,:,2:]        

        self.gray_x_R = tf.pad(self.gray_R,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_x_R = self.gray_x_R[:,2:,1:-1]
        self.gray_y_R = tf.pad(self.gray_R,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_y_R = self.gray_y_R[:,1:-1,2:]
        self.gray_h_R = tf.pad(self.gray_R,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_h_R = self.gray_h_R[:,2:,2:]
        
        self.gray_x_R = tf.abs(self.gray_R - self.gray_x_R)
        self.gray_y_R = tf.abs(self.gray_R - self.gray_y_R)
        self.gray_h_R = tf.abs(self.gray_R - self.gray_h_R)
        
        self.img_n_R = self.gray_x_R + self.gray_y_R + self.gray_h_R
        self.img_n_R = tf.reshape(self.img_n_R,[self.batch_size,-1])

        self.gray_x_G = tf.pad(self.gray_G,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_x_G = self.gray_x_G[:,2:,1:-1]
        self.gray_y_G = tf.pad(self.gray_G,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_y_G = self.gray_y_G[:,1:-1,2:]
        self.gray_h_G = tf.pad(self.gray_G,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_h_G = self.gray_h_G[:,2:,2:]
        
        self.gray_x_G = tf.abs(self.gray_G - self.gray_x_G)
        self.gray_y_G = tf.abs(self.gray_G - self.gray_y_G)
        self.gray_h_G = tf.abs(self.gray_G - self.gray_h_G)
        
        self.img_n_G = self.gray_x_G + self.gray_y_G + self.gray_h_G
        self.img_n_G = tf.reshape(self.img_n_G,[self.batch_size,-1])

        self.gray_x_B = tf.pad(self.gray_B,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_x_B = self.gray_x_B[:,2:,1:-1]
        self.gray_y_B = tf.pad(self.gray_B,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_y_B = self.gray_y_B[:,1:-1,2:]
        self.gray_h_B = tf.pad(self.gray_B,[[0,0],[1,1],[1,1],[0,0]],'SYMMETRIC')
        self.gray_h_B = self.gray_h_B[:,2:,2:]
        
        self.gray_x_B = tf.abs(self.gray_B - self.gray_x_B)
        self.gray_y_B = tf.abs(self.gray_B - self.gray_y_B)
        self.gray_h_B = tf.abs(self.gray_B - self.gray_h_B)
        
        self.img_n_B = self.gray_x_B + self.gray_y_B + self.gray_h_B
        self.img_n_B = tf.reshape(self.img_n_B,[self.batch_size,-1])
        
        self.distance = tf.multiply(self.img_n_R,self.img_n_R) + tf.multiply(self.img_n_G,self.img_n_G) + tf.multiply(self.img_n_B,self.img_n_B)
        self.distance = tf.sqrt(self.distance)
    
    def cal_fre(self):
        self.y = tf.cast(tf.range(0, 255, 1), tf.float64)
        max_n_distance = tf.reduce_max(self.distance,reduction_indices=[1])
        min_n_distance = tf.reduce_min(self.distance,reduction_indices=[1])
        
        max_n_distance = tf.tile(max_n_distance, multiples=[self.input_width*self.input_height])
        max_n_distance = tf.reshape(max_n_distance,[self.batch_size,-1])
        min_n_distance = tf.tile(min_n_distance, multiples=[self.input_width*self.input_height])
        min_n_distance = tf.reshape(min_n_distance,[self.batch_size,-1])        
        
        self.distance = (self.distance - min_n_distance) / (max_n_distance - min_n_distance) 
            
        self.img_cn_distance = self.distance * 255
        self.img_cn_distance = tf.cast(self.img_cn_distance, dtype=tf.int32)
        for cn in range(self.img_cn_distance.shape[0]):
            if cn == 0:
                self.frequn_distance = tf.bincount(self.img_cn_distance[0:1,:], minlength=255, maxlength=255)
            elif cn == self.img_cn_distance.shape[0] - 1:
                self.frequn_distance = tf.concat([self.frequn_distance, tf.bincount(self.img_cn_distance[cn:,:], minlength=255, maxlength=255)],0)
            else:
                self.frequn_distance = tf.concat([self.frequn_distance, tf.bincount(self.img_cn_distance[-1:,:], minlength=255, maxlength=255)],0)
        self.frequn_distance = tf.cast(tf.reshape(self.frequn_distance, [self.batch_size,-1]), tf.float64)
        self.frequn_distance = self.frequn_distance / (self.input_width*self.input_height)        
        self.img_cn_distance = tf.cast(self.img_cn_distance, dtype=tf.float64)
        self.img_rn_distance = tf.reshape(self.img_cn_distance,[self.batch_size,self.input_width,self.input_height])

    def generator(self, args, input, attention_o, name="generator", reuse=False):
        
        input_shape = input.get_shape().as_list()
        
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:  

            conv1, atten1 = architecture.att_conv2d(input, attention_o, 64,
                          kernel=5,
                          stride=2,
                          padding="SAME",
                          name="conv1"
                          )
            conv1 = architecture.batch_norm(conv1, name="conv_bn1")
            conv1 = tf.nn.relu(conv1)
            print(conv1)

            block1, b_atten1 = architecture.att_conv2d_residual_block(conv1, atten1, 64, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block1")
            print(block1)

            conv2, atten2 = architecture.att_conv2d(block1, b_atten1, 128,
                          kernel=3,
                          stride=2,
                          padding="SAME",
                          name="conv2"
                          )
            conv2 = architecture.batch_norm(conv2, name="conv_bn2")
            conv2 = tf.nn.relu(conv2)
            print(conv2)

            block2, b_atten2 = architecture.att_conv2d_residual_block(conv2, atten2, 128, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block2")
            print(block2)

            grain1_dilate_conv1, grain1_mask1 = architecture.att_dilate_conv2d(block2, b_atten2, rate=1, name="grain1_1")
            grain1_dilate_conv2, grain1_mask1 = architecture.att_dilate_conv2d(grain1_dilate_conv1, grain1_mask1, rate=1, name="grain1_2")
            grain2_dilate_conv1, grain1_mask2 = architecture.att_dilate_conv2d(block2, b_atten2, rate=2, name="grain2_1")
            grain2_dilate_conv2, grain1_mask2 = architecture.att_dilate_conv2d(grain2_dilate_conv1, grain1_mask2, rate=2, name="grain2_2")
            grain3_dilate_conv1, grain1_mask3 = architecture.att_dilate_conv2d(block2, b_atten2, rate=3, name="grain3_1")
            grain3_dilate_conv2, grain1_mask3 = architecture.att_dilate_conv2d(grain3_dilate_conv1, grain1_mask3, rate=3, name="grain3_2")
            grain4_dilate_conv1, grain1_mask4 = architecture.att_dilate_conv2d(block2, b_atten2, rate=4, name="grain4_1")
            grain4_dilate_conv2, grain1_mask4 = architecture.att_dilate_conv2d(grain4_dilate_conv1, grain1_mask4, rate=4, name="grain4_2")

            multi_grain = tf.concat([block2*b_atten2,
                                     grain1_dilate_conv2*grain1_mask1,
                                     grain2_dilate_conv2*grain1_mask2,
                                     grain3_dilate_conv2*grain1_mask3,
                                     grain4_dilate_conv2*grain1_mask4],3)
            print(multi_grain)
            
            deconv1 = architecture.deconv2d(multi_grain, [self.batch_size, int(args.input_width/2), int(args.input_height/2), 512], name="deconv1")
            deconv1 = architecture.batch_norm(deconv1, name="deconv_bn1")
            deconv1 = tf.nn.relu(deconv1)
            print(deconv1)

            conv4 = architecture.conv2d(deconv1, 256,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv4"
                          )
            conv4 = architecture.batch_norm(conv4, name="conv_bn4")
            conv4 = tf.nn.relu(conv4)
            print(conv4)
            
            block4 = architecture.conv2d_residual_block(conv4, 256, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block4")
            print(block4)
            
            deconv2 = architecture.deconv2d(block4, [self.batch_size, args.input_width, args.input_height, 256], name="deconv2")
            deconv2 = architecture.batch_norm(deconv2, name="deconv_bn2")
            deconv2 = tf.nn.relu(deconv2)
            print(deconv2)
            
            conv5 = architecture.conv2d(deconv2, 128,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv5"
                          )
            conv5 = architecture.batch_norm(conv5, name="conv_bn5")
            conv5 = tf.nn.relu(conv5)
            print(conv5)
            
            block5 = architecture.conv2d_residual_block(conv5, 128, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block5")
            print(block5)
            
            conv6 = architecture.conv2d(block5, 64,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv6"
                          )
            conv6 = architecture.batch_norm(conv6, name="conv_bn6")
            conv6 = tf.nn.relu(conv6)
            print(conv6)
            
            block6 = architecture.conv2d_residual_block(conv6, 64, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block6")
            print(block6)
            
            conv7 = architecture.conv2d(block6, 32,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv7"
                          )
            conv7 = architecture.batch_norm(conv7, name="conv_bn7")
            conv7 = tf.nn.relu(conv7)
            print(conv7)
            
            block7 = architecture.conv2d_residual_block(conv7, 32, num_layers=3, padding="SAME", kernel=3, stride=1, name="conv_block7")
            print(block7)
            
            conv8 = architecture.conv2d(block7, 3,
                          kernel=3,
                          stride=1,
                          padding="SAME",
                          name="conv8"
                          )
            conv8 = architecture.batch_norm(conv8, name="conv_bn8")
            conv8 = tf.nn.tanh(conv8)
            print(conv8)
            
        At = attention_o
        
        return conv8, multi_grain, At, nets

    def content_discriminator(self, input, name="content_discriminator", reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            conv1 = tf.contrib.layers.conv2d(input, 10, 5, 1,
                                     padding="SAME",
                                     activation_fn=None,
                                     scope="conv1")

            conv1 = architecture.max_pool(conv1, 'VALID', 2, 2, 2, name="mp1")
            conv1 = tf.nn.relu(conv1)
            nets.append(conv1)
            print(conv1)
            conv2 = tf.contrib.layers.conv2d(conv1, 30, 5, 1,
                                     padding="SAME",
                                     activation_fn=None,
                                     scope="conv2")
            conv2 = architecture.max_pool(conv2, 'VALID', 2, 2, 2, name="mp2")
            conv2 = tf.nn.relu(conv2)
            nets.append(conv2)
            print(conv2)
            conv3 = tf.contrib.layers.conv2d(conv2, 64, 5, 1,
                                     padding="SAME",
                                     activation_fn=None,
                                     scope="conv3")
            conv3 = architecture.max_pool(conv3, 'VALID', 2, 2, 2, name="mp1")
            conv3 = tf.nn.relu(conv3)
            nets.append(conv3)
            print(conv3)

            flatten = tf.contrib.layers.flatten(conv3)
            
            output1 = architecture.linear(flatten, 64, name="linear1")
            
            output2 = architecture.linear(output1, 1, name="linear2")

            return output2, nets
        
    def partial_discriminator(self, input, mask, name="partial_discriminator", reuse=False):
        nets = []
        with tf.variable_scope(name, reuse=reuse) as scope:
            Pconv1, Pmask1 = architecture.Pconv2d(input, mask, out_filter=10, padding="VALID", kernel=5, stride=1, name='Pconv1')
            Pconv1 = architecture.max_pool(Pconv1, 'VALID', 2, 2, 2, name="Imp1")
            Pmask1 = architecture.max_pool(Pmask1, 'VALID', 2, 2, 2, name="Mmp1")
            Pconv1 = tf.nn.relu(Pconv1)
            Pmask1 = tf.nn.tanh(Pmask1)
            
            nets.append(Pconv1)
            
            Pconv2, Pmask2 = architecture.Pconv2d(Pconv1, Pmask1, out_filter=30, padding="VALID", kernel=5, stride=1, name='Pconv2')
            Pconv2 = architecture.max_pool(Pconv2, 'VALID', 2, 2, 2, name="Imp2")
            Pmask2 = architecture.max_pool(Pmask2, 'VALID', 2, 2, 2, name="Mmp2")
            Pconv2 = tf.nn.relu(Pconv2)
            Pmask2 = tf.nn.tanh(Pmask2)
            nets.append(Pconv2)
            
            Pconv3, Pmask3 = architecture.Pconv2d(Pconv2, Pmask2, out_filter=64, padding="VALID", kernel=5, stride=1, name='Pconv3')
            Pconv3 = tf.nn.relu(Pconv3)
            nets.append(Pconv3)
            
            flatten = tf.contrib.layers.flatten(Pconv3)
            
            output1 = architecture.linear(flatten, 64, name="linear1")
            
            output2 = architecture.linear(output1, 1, name="linear2")  
            
            PMask = Pmask2

            return output2, PMask, nets




