import argparse
import math

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description='')

#Image setting
parser.add_argument('--input_width', dest='input_width', default=128, help='input image width')
parser.add_argument('--input_height', dest='input_height', default=128, help='input image height')
parser.add_argument('--input_channel', dest='input_channel', default=3, help='input image channel')
parser.add_argument('--k', dest='k', default=3, help='muti data')
parser.add_argument('--target', dest='target', default=0.01, help='target')
parser.add_argument('--Normalized range', dest='N_range', default=100, help='Normalized range')

parser.add_argument('--input_dim', dest='input_dim', default=100, help='input z size')
parser.add_argument('--pcnn_size', dest='pcnn_size', default=32, help='pcnn_size')

#Training Settings
parser.add_argument('--continue_training', dest='continue_training', default=True, type=str2bool, help='flag to continue training')

parser.add_argument('--data', dest='data', default='./data/', help='training image path')
parser.add_argument('--data', dest='data', default='./testdata/', help='test image path')
parser.add_argument('--mask', dest='mask', default='./mask', help='training mask image path')

parser.add_argument('--blank', dest='blank', default=2, help='blank size')

parser.add_argument('--batch_size', dest='batch_size', default=10, help='batch size')
parser.add_argument('--train_step', dest='train_step', default=600, help='total number of train_step')
parser.add_argument('--Tc', dest='Tc', default=100, help='Tc to train Completion Network')
parser.add_argument('--Td', dest='Td', default=900, help='Td to train Discriminator Network')
parser.add_argument('--T_all', dest='T_all', default=15, help='T_all to train Discriminator Network')
parser.add_argument('--frequency', dest='frequency', default=30, help='Recording frequency')


parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, help='learning rate of the optimizer')
parser.add_argument('--momentum', dest='momentum', default=0.5, help='momentum of the optimizer')

parser.add_argument('--alpha', dest='alpha', default=1.0, help='alpha')

parser.add_argument('--margin', dest='margin', default=5, help='margin')

parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./checkpoints/', help='saved model checkpoint path')
parser.add_argument('--graph_path', dest='graph_path', default='./graphs/', help='tensorboard graph')
parser.add_argument('--images_path', dest='images_path', default='./images/', help='result images path')



args = parser.parse_args()