import tensorflow as tf
import numpy as np
import config 
import skimage
from sklearn.preprocessing import normalize
from skimage import io
import network
import os
import csv
import sys
import cv2
import Cloud_model


def train(args, sess, model):

    c_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_C").minimize(model.c_loss, var_list=model.c_vars)
    g_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_G").minimize(model.recon_loss, var_list=model.g_vars)
    p_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_P").minimize(model.p_loss, var_list=model.p_vars)
    m_optimizer = tf.train.AdamOptimizer(args.learning_rate, beta1=args.momentum, name="AdamOptimizer_M").minimize(model.muti_loss, var_list=model.g_vars)

    epoch = 0
    step = 0
    global_step = 0

    saver = tf.train.Saver()        
    if args.continue_training:
        tf.local_variables_initializer().run()
        last_ckpt = tf.train.latest_checkpoint(args.checkpoints_path)
        saver.restore(sess, last_ckpt)
        ckpt_name = str(last_ckpt)
        print ("Loaded model file from " + ckpt_name)
        epoch = 263
    else:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    all_summary = tf.summary.merge([model.recon_loss_sum,
                                    model.c_loss_sum,
                                    model.p_loss_sum,
                                    model.input_img_sum, 
                                    model.real_img_sum,
                                    model.recon_img_sum,
                                    model.generate_img_sum,                                    
                                    model.all_loss_sum,
                                    model.mask_sum])
    writer = tf.summary.FileWriter(args.graph_path, sess.graph)

    while epoch < args.train_step:
        if epoch < args.Tc:
            
            summary, g_loss, _, = sess.run([all_summary, model.recon_loss, g_optimizer])
            if epoch%args.frequency == 0: 
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] G Loss: [%.4f]" % (epoch, step, g_loss))
        
        elif epoch >= args.Tc and (epoch%args.T_all == 0 or epoch%args.T_all == 1):
            
            summary, d_loss_c, _, _ = sess.run([all_summary, model.c_loss, c_optimizer, model.c_clip])
            if epoch%args.frequency == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] C Loss: [%.4f]" % (epoch, step, d_loss_c))
        
        elif epoch >= args.Tc and (epoch%args.T_all == 2 or epoch%args.T_all == 3):

            summary, d_loss_p, _, _ = sess.run([all_summary, model.p_loss, p_optimizer, model.p_clip])
            if epoch%args.frequency == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] P Loss: [%.4f]" % (epoch, step, d_loss_p))
            
        elif epoch >= args.Td and int(epoch/args.T_all)%2 == 1:
          
            if epoch%args.T_all == 4:
             
                img_rn_4, img_n_distance, frequn_distance, y = sess.run([model.img_rn_distance, model.img_cn_distance, model.frequn_distance, model.y])
                k = args.k
                target = 0.01
                muti_data = np.zeros(k)
                for i in range(args.batch_size):
                    if i == 0:
                        Ex_d,En_d,He_d,CD_d = Cloud_model.t_cloud_tran(img_n_distance[i],k,target,y,frequn_distance[i])
                        Ex_d[0] = Ex_d[0] / 2
                        
                        multi_sharp, muti_count = Cloud_model.t_cal_muti_all(Ex_d,En_d,He_d,img_rn_4[i],k)
                        muti_data += muti_count
                    else:
                        multi_sharp, muti_count = Cloud_model.t_cal_muti_all(Ex_d,En_d,He_d,img_rn_4[i],k)
                        muti_data += muti_count 
                
                muti_data = muti_data / (args.batch_size * args.input_height * args.input_width)
                print(muti_data)                                  
                
                
                summary, muti_loss, _ = sess.run([all_summary,model.muti_loss, m_optimizer],feed_dict={model.muti_count: muti_data})
                if epoch%args.frequency == 0:
                    writer.add_summary(summary, global_step)
                    global_step += 1
                print ("Epoch [%d] Step [%d] Muti Loss: [%.4f]" % (epoch, step, muti_loss))
            else:
                
                img_rn_distance = sess.run(model.img_rn_distance)
                k = args.k
                target = 0.01
                muti_data = np.zeros(k)
                for i in range(args.batch_size):
                    multi_sharp, muti_count = Cloud_model.t_cal_muti_all(Ex_d,En_d,He_d,img_rn_distance[i],k)
                    muti_data += muti_count
                    
                muti_data = muti_data / (args.batch_size * args.input_height * args.input_width)
                summary, muti_loss, _ = sess.run([all_summary,model.muti_loss, m_optimizer],feed_dict={model.muti_count: muti_data})

                if epoch%args.frequency == 0:
                    writer.add_summary(summary, global_step)
                    global_step += 1
                print ("Epoch [%d] Step [%d] Muti Loss: [%.4f]" % (epoch, step, muti_loss))
        else:
                        
            summary, all_loss, _ = sess.run([all_summary,model.all_loss, g_optimizer])
            if epoch%args.frequency == 0:
                writer.add_summary(summary, global_step)
                global_step += 1
            print ("Epoch [%d] Step [%d] ALl Loss: [%.4f]" % (epoch, step, all_loss))
        if step*args.batch_size >= model.data_count:
            saver.save(sess, args.checkpoints_path + "/model")
            with open('./epoch.txt','w') as f:
                f.writelines(str(epoch))
            step = 0
            epoch += 1
        step += 1

    coord.request_stop()
    coord.join(threads)
    sess.close()            
    print("Done.")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    run_config = tf.ConfigProto()
    run_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    run_config.gpu_options.allow_growth = True

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)
    if not os.path.exists(args.images_path):
        os.makedirs(args.images_path)

    with tf.Session(config=run_config) as sess:
        model = network.network(args)
        print ('Start Training...')
        train(args, sess, model)

main(config.args)
