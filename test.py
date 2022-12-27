import tensorflow as tf
import numpy as np
import config
import network
import cv2
import skimage
from skimage import io
import csv
import test_region_grow
import sys
import os
import time


drawing = False # true if mouse is pressed
ix,iy = -1,-1
color = (255,255,255)
size = 5
test_size = 128

def erase_img(args, img):

    # mouse callback function
    def erase_rect(event,x,y,flags,param):
        global ix,iy,drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(region,(x-size,y-size),(x+size,y+size),color,-1)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                # cv2.circle(img,(x,y),10,(255,255,255),-1)
                cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
                cv2.rectangle(region,(x-size,y-size),(x+size,y+size),color,-1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.rectangle(img,(x-size,y-size),(x+size,y+size),color,-1)
            cv2.rectangle(mask,(x-size,y-size),(x+size,y+size),color,-1)
            cv2.rectangle(region,(x-size,y-size),(x+size,y+size),color,-1)


    cv2.namedWindow('image')
    cv2.setMouseCallback('image',erase_rect)
    #cv2.namedWindow('mask')
    cv2.setMouseCallback('mask',erase_rect)
    cv2.setMouseCallback('region',erase_rect)
    
    mask = np.zeros(img.shape)
    region = np.zeros(img.shape)
    
    while(1):
        img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image',img_show)
        k = cv2.waitKey(100) & 0xFF
        if k == 13:
            break
        
        
    
    test_img = cv2.resize(img, (args.input_height, args.input_width))/127.5 - 1
    test_mask = cv2.resize(mask, (args.input_height, args.input_width))/255.0   
    test_region = cv2.resize(region, (args.input_height, args.input_width))/255


    region_begin = test_region_grow.region_find(test_region)
    region_begin = -region_begin + 1
    region_grow = test_region_grow.region_grow(test_img,region_begin)
    region_grow = region_grow[...,np.newaxis]
    test_img = (test_img * (1-test_mask)) + test_mask

    
    cv2.destroyAllWindows()
    return np.tile(test_img[np.newaxis,...], [args.batch_size,1,1,1]),np.tile(test_mask[np.newaxis,...], [args.batch_size,1,1,1]),np.tile(region_grow[np.newaxis,...], [args.batch_size,1,1,1])

def some_image(args, path, test_img):
    mask = cv2.imread(path)
    mask = cv2.resize(mask, (test_size, test_size))/255.0
    
    real_mask = np.array(mask)
    real_mask = real_mask + 0.6
    real_mask = real_mask.astype(int)
    real_mask = real_mask.astype(float)

    test_img = test_img * real_mask + real_mask

    
    return np.tile(test_img[np.newaxis,...], [args.batch_size,1,1,1]),np.tile(real_mask[np.newaxis,...], [args.batch_size,1,1,1])

def test(args, sess, model):
    #saver  
    saver = tf.train.Saver()        
    last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
    saver.restore(sess, last_ckpt)
    ckpt_name = str(last_ckpt)
    print ("Loaded model file from " + ckpt_name)   

    '''
    img = cv2.imread('./test//test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    test_img, mask, region_grow = erase_img(args, img)   
    print(mask.shape)
    cv2.imwrite('./mask3.jpg',mask[0]*255)'''
    
    times = 0
    time_count = 0
    
    
    
    for i in range(4):
        if i == 0:            
            test_path = './test_mask//mask.jpg'
            result = './result//recon//'
        elif i == 1:
            
            test_path = './test_mask//mask1.jpg'
            result = './result//recon1//'
        elif i == 2:
            
            test_path = './test_mask//mask2.jpg'
            result = './result//recon2//'
        elif i == 3:
            
            test_path = './test_mask//mask6.jpg'
            result = './result//recon3//'
        elif i == 4:
            
            test_path = './test_mask//mask4.jpg'
            result = './result//recon4//'
        elif i == 5:
            
            test_path = './test_mask//mask5.jpg'
            result = './result//recon5//'
        
        for root, dirs, files in os.walk('./data//PL//'):
            for j in range(len(files)):
                result_path = result + files[j]
                files[j] = root + files[j]
        
                img = cv2.imread(files[j])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
                orig_test = cv2.resize(img, (test_size, test_size))/127.5 - 1
                #orig_test = cv2.resize(img, (256, 256))/127.5 - 1
                orig_test = np.tile(orig_test[np.newaxis,...],[args.batch_size,1,1,1])
                orig_test = orig_test.astype(np.float32)
                orig_w, orig_h = img.shape[0], img.shape[1]
        
                t0 = time.time()
                
                test_img, mask = some_image(args, test_path, orig_test[0])
                mask = mask * 0 + 1
                test_img = test_img.astype(np.float32)
                #region_grow = region_grow.astype(np.float32)
                
                print ("Testing situation ",i,"picture ",j)
                res_img = sess.run(model.test_g_imgs, feed_dict={model.single_orig:orig_test,
                                                                   model.single_test:test_img,
                                                                   model.single_mask:mask})
                
                t1 = time.time()
                times += t1 - t0
                time_count += 1

                orig = cv2.resize((orig_test[0]+1)/2, (int(orig_h/2), int(orig_w/2)))
                test = cv2.resize((test_img[0]+1)/2, (int(orig_h/2), int(orig_w/2)))
                recon = cv2.resize((res_img[0]+1)/2, (128, 128))
                generate = cv2.resize((res_img[0]+1)/2, (int(orig_h/4), int(orig_w/4)))
            

                res = cv2.cvtColor(recon, cv2.COLOR_BGR2RGB)
                generate = cv2.cvtColor(generate, cv2.COLOR_BGR2RGB)
                pretur_image = cv2.cvtColor(test_img[0], cv2.COLOR_BGR2RGB)
                
                cv2.imwrite(result_path,res*255)

    print(times/time_count)
    print("Done.")


def main(args):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    
    with tf.Session(config=run_config) as sess:
        model = network.network(args)

        print ('Start Testing...')
        test(args, sess, model)

t0 = time.time()        
main(config.args)
t1 = time.time()
print(t1-t0)
