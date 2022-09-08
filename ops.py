
from glob import glob 
import os
import tensorflow as tf
import numpy as np
import cv2 

def block_patch(input, margin=0):
    shape = input.get_shape().as_list()

    pad_size = tf.random_uniform([2], minval=15, maxval=25, dtype=tf.int32)
    patch = tf.zeros([pad_size[0], pad_size[1], shape[-1]], dtype=tf.float32)
 	
    h_ = tf.random_uniform([1], minval=margin, maxval=shape[0]-pad_size[0]-margin, dtype=tf.int32)[0]
    w_ = tf.random_uniform([1], minval=margin, maxval=shape[1]-pad_size[1]-margin, dtype=tf.int32)[0]
	
    padding = [[h_, shape[0]-h_-pad_size[0]], [w_, shape[1]-w_-pad_size[1]], [0, 0]]
    padded = tf.pad(patch, padding, "CONSTANT", constant_values=1)

    coord = h_, w_
    res = tf.multiply(input, padded)

    pad_x_1 = 1
    pad_x_2 = 1
    pad_y_1 = 1
    pad_y_2 = 1
    if h_ == 0:
        pad_x_1 = 0
    if h_ == shape[0]-pad_size[0]-margin:
        pad_x_2 = 0
    if w_ == 0:
        pad_y_1 = 0
    if w_ == shape[1]-pad_size[1]-margin:
        pad_y_2 = 0
        
    grow_padded = tf.pad(patch, [[pad_x_1,pad_x_2],[pad_y_1,pad_y_2],[0,0]], "CONSTANT", constant_values=1)
    grow_padding = [[h_-pad_x_1, shape[0]-h_-pad_size[0]-pad_x_2], [w_-pad_y_1, shape[1]-w_-pad_size[1]-pad_y_2], [0, 0]]
    
    mask_grow = tf.pad(grow_padded, grow_padding, "CONSTANT", constant_values=0)
    mask_grow = tf.multiply(input, mask_grow)
    
    return res, padded, coord, pad_size, mask_grow

def load_train_data(args):
    paths = os.path.join(args.data, "*.jpg")
    m_paths = os.path.join(args.mask, "*.jpg")

    data_count = len(glob(paths))
    mask_count = len(glob(m_paths))
            
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))
    m_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(m_paths))
    
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    images = tf.image.decode_jpeg(image_file, channels=3)
	
    m_image_reader = tf.WholeFileReader()
    _, m_image_file = m_image_reader.read(m_filename_queue)
    masks = tf.image.decode_jpeg(m_image_file, channels=3)

    images = tf.image.resize_images(images ,[args.input_height, args.input_width])
    masks = tf.image.resize_images(masks ,[args.input_height, args.input_width])

    images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
    masks = tf.image.convert_image_dtype(masks, dtype=tf.float32) / 255.0
	
    orig_images = images
    
    masks = masks + 0.6
    masks = tf.cast(masks, dtype=tf.int32)
    masks = tf.cast(masks, dtype=tf.float32)

    mask = -(masks - 1)
    images = images * mask + mask

    orig_imgs, perturbed_imgs, mask = tf.train.shuffle_batch([orig_images, images, masks],
																			  batch_size=args.batch_size,
																			  capacity=args.batch_size*2,
																			  min_after_dequeue=args.batch_size
																			 )
    return orig_imgs, perturbed_imgs, mask, data_count, mask_count


def load_test_data(args):
    paths = os.path.join(args.testdata, "*.jpg")
    data_count = len(paths)

    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(paths))

    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    images = tf.image.decode_jpeg(image_file, channels=3)

    images = tf.image.resize_images(images ,[args.input_height, args.input_width])
    images = tf.image.convert_image_dtype(images, dtype=tf.float32) / 127.5 - 1
	
    orig_images = images
    images, mask, coord, pad_size, _ = block_patch(images, margin=args.margin)
    mask = tf.reshape(mask, [args.input_height, args.input_height, 3])

    mask = -(mask - 1)
    images += mask

    orig_imgs, mask, test_imgs = tf.train.batch([orig_images, mask, images],
												batch_size=args.batch_size,
												capacity=args.batch_size,
											    )


    return orig_imgs, test_imgs, mask, data_count

