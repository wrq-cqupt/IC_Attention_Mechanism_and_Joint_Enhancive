import tensorflow as tf


def batch_norm(input, name="batch_norm"):
	with tf.variable_scope(name) as scope:
		input = tf.identity(input)
		channels = input.get_shape()[3]

		offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

		mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)

		normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)

		return normalized_batch 


def linear(input, output_size, name="linear"):
	shape = input.get_shape().as_list()

	with tf.variable_scope(name) as scope:
		matrix = tf.get_variable("W", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
		bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

		return tf.matmul(input, matrix) + bias
    
def max_pool(input, padding, height, width, stride, name="max_pool"):
	input_shape = input.get_shape().as_list()
	with tf.variable_scope(name) as scope:
		
		pool = tf.nn.max_pool(input,[1,height,width,1],[1,stride,stride,1],padding)      
		
		return pool


def conv2d(input, out_filter, padding, kernel=5, stride=2, name="conv2d"):
	input_shape = input.get_shape().as_list()
	with tf.variable_scope(name) as scope:
		w = tf.get_variable("w", [kernel, kernel, input_shape[-1], out_filter], initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("b", [out_filter], initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input, w, 
							strides=[1, stride, stride, 1],
							padding=padding
							)

		conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

		return conv

def conv2d_mask(input, out_filter, padding, kernel=5, stride=2, name="conv2d"):
	input_shape = input.get_shape().as_list()
	with tf.variable_scope(name) as scope:
		w = tf.ones([kernel, kernel, input_shape[-1], out_filter])		
		conv = tf.nn.conv2d(input, w, 
							strides=[1, stride, stride, 1],
							padding=padding
							)

		return conv
   
def Pconv2d(input, mask, out_filter, padding, kernel=5, stride=2, name="Pconv2d"):
	input_shape = input.get_shape().as_list()
	normalization = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(mask,2),1),1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1)
	normalization_T = tf.keras.backend.repeat_elements(normalization,input_shape[1],1)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[2],2)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[3],3)
	img_conv = conv2d(input*mask/normalization_T, out_filter, padding, kernel, stride, name=name + "_I_mask")
	mask_conv = conv2d_mask(mask, out_filter, padding, kernel, stride, name=name + "_M_mask")

	return img_conv, mask_conv

def att_conv2d(input, mask, out_filter, padding, kernel=5, stride=2, name="Att_conv2d"):
	input_shape = input.get_shape().as_list()

	normalization = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(mask,2),1),1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1)

	normalization_T = tf.keras.backend.repeat_elements(normalization,input_shape[1],1)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[2],2)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[3],3)

	img_conv = conv2d(input*mask/normalization_T, out_filter, padding, kernel, stride, name=name + "_A_img")
	pad_num = int((kernel-1)/2)   
	mask = tf.pad(mask,[[0,0],[pad_num,pad_num],[pad_num,pad_num],[0,0]],mode="REFLECT")    
	mask_conv = conv2d_mask(mask, out_filter, "VALID", kernel, stride, name=name + "_A_mask")

	temp_max = tf.reduce_max(mask_conv,0)    
	temp_max = tf.reduce_max(temp_max,0) 
	temp_max = tf.reduce_max(temp_max,0)
    
	temp_min = tf.reduce_min(mask_conv,0)    
	temp_min = tf.reduce_min(temp_min,0) 
	temp_min = tf.reduce_min(temp_min,0)
    
	mask_conv = (mask_conv - temp_min) / (temp_max - temp_min)

	return img_conv, mask_conv


def deconv2d(input, out_shape, name="deconv2d"):
	input_shape = input.get_shape().as_list()
	with tf.variable_scope(name) as scope:
		w = tf.get_variable("w", [4, 4, out_shape[-1], input_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.nn.conv2d_transpose(input, w, 
										output_shape=out_shape,
										strides=[1, 2, 2, 1],
										padding="SAME")
		deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

		return deconv

def dilate_conv2d(input, out_shape, rate, name="dilate_conv2d"):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [3, 3, input_shape[-1], out_shape[-1]])
        b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))
        dilate_conv = tf.nn.atrous_conv2d(input, w,
		                                  rate=rate,
		                                  padding="SAME"
		                                  )
        dilate_conv = tf.reshape(tf.nn.bias_add(dilate_conv, b), dilate_conv.get_shape())

    return dilate_conv

def att_dilate_conv2d(input, mask, rate, name="Att_dilate_conv2d"):
	input_shape = input.get_shape().as_list()

	normalization = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(mask,2),1),1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1) 
	normalization = tf.expand_dims(normalization,1)

	normalization_T = tf.keras.backend.repeat_elements(normalization,input_shape[1],1)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[2],2)
	normalization_T = tf.keras.backend.repeat_elements(normalization_T,input_shape[3],3)    

	img_conv = dilate_conv2d(input*mask/normalization_T,[input.get_shape()[0], input.get_shape()[1], input.get_shape()[2], 128], rate, name=name+"_att_dc")
  
	mask_conv = dilate_conv2d(mask,[mask.get_shape()[0], mask.get_shape()[1], mask.get_shape()[2], 128], rate, name=name+"_att_dc_mask")

	temp_max = tf.reduce_max(mask_conv,0)    
	temp_max = tf.reduce_max(temp_max,0) 
	temp_max = tf.reduce_max(temp_max,0)
    
	temp_min = tf.reduce_min(mask_conv,0)    
	temp_min = tf.reduce_min(temp_min,0) 
	temp_min = tf.reduce_min(temp_min,0)
    
	mask_conv = (mask_conv - temp_min) / (temp_max - temp_min)

	return img_conv, mask_conv

def conv2d_residual_block(input, out_filter, num_layers, padding, kernel=3, stride=1, name="conv2d_residual_block"):
    input_shape = input.get_shape().as_list()
    block_name = name
    for i in range(num_layers):
        if i == 0:
            resi_conv = conv2d(input, out_filter, padding, kernel, stride, name=block_name + "resi_conv" + str(i+1))       
            resi_conv = batch_norm(resi_conv, name=block_name + "resi_conv_bn" + str(i+1))
            resi_conv = tf.nn.relu(resi_conv)
        else:
            resi_conv = conv2d(resi_conv, out_filter, padding, kernel, stride, name=block_name + "resi_conv" + str(i+1))        
            resi_conv = batch_norm(resi_conv, name=block_name + "resi_conv_bn" + str(i+1))
            resi_conv = tf.nn.relu(resi_conv)
    
    conv_block = resi_conv + input
    
    return conv_block

def att_conv2d_residual_block(input, mask, out_filter, num_layers, padding, kernel=3, stride=1, name="att_conv2d_residual_block"):
    input_shape = input.get_shape().as_list()
    block_name = name
    for i in range(num_layers):
        if i == 0:
            resi_conv, resi_att = att_conv2d(input, mask, out_filter, padding, kernel, stride, name=block_name + "att_resi_conv" + str(i+1))       
            resi_conv = batch_norm(resi_conv, name=block_name + "att_resi_conv_bn" + str(i+1))
            resi_conv = tf.nn.relu(resi_conv)
        else:
            resi_conv, resi_att = att_conv2d(resi_conv, resi_att, out_filter, padding, kernel, stride, name=block_name + "att_resi_conv" + str(i+1))        
            resi_conv = batch_norm(resi_conv, name=block_name + "att_resi_conv_bn" + str(i+1))
            resi_conv = tf.nn.relu(resi_conv)
    
    conv_block = resi_conv + input
    
    return conv_block, resi_att