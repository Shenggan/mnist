'''Command line arguments'''
import argparse
import resnet

model_names = sorted(name for name in resnet.__dict__)

print(model_names)

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
    parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: ResNet18)')
    parser.add_argument('--save_log', dest='save_log',
                        help='The directory used to save the log',
                        default='logs/resnet18', type=str)
    parser.add_argument('--low_lr', default=5, type=int, help='decrease_lr')
    args = parser.parse_args()
    return args
