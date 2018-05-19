'''Command line arguments'''
import argparse

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', default=8.0, type=float, help='norm length for (virtual) adversarial training')
    parser.add_argument('--num_power_iterations', default=1, type=float, help='the number of power iterations')
    parser.add_argument('--xi', default=1e-6, type=float, help='small constant for finite difference')

    parser.add_argument('--keep_prob_hidden', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--lrelu_a', default=0.1, type=float, help='lrelu slope')
    parser.add_argument('--top_bn', default=False, type=bool, help='')

    parser.add_argument('--training', default='supervised', type=str, help='training type')
    parser.add_argument('--numlabels', default=60000, type=int, help='number of labels')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    
    args = parser.parse_args()
    return args
