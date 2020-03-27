import re
import argparse
import logging 

from third_party.mean_teacher import cli as mt_cli

from src import datasets
from src import architectures


LOG = logging.getLogger('main')


def create_parser():
    parser = argparse.ArgumentParser(description='Dual Student SSL PyTorch Version')

    # global
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path of the resume checkpoint (default: '')')
    parser.add_argument('--validation', type=str2bool,
                        help='only validate the model on eval-subdir')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model (this argument is not used any more)')

    # data
    parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=datasets.__all__,
                        help='dataset: ' + ' | '.join(datasets.__all__) + ' (default: cifar10)')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--target-domain', default=None, type=str, 
                        help='target unlabeled domain for domain adaptation experiments if not None')

    # optimization
    parser.add_argument('--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    
    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for optimizer')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--validation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # archtecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cnn13', choices=architectures.__all__,
                        help='model architecture: ' + ' | '.join(architectures.__all__))

    # constraint
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE', choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-scale', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')

    parser.add_argument('--stable-threshold', default=0.0, type=float, metavar='THRESHOLD',
                        help='threshold for stable sample')
    parser.add_argument('--stabilization-scale', default=None, type=float, metavar='WEIGHT',
                        help='use stabilization loss with given weight (default: None)')
    parser.add_argument('--stabilization-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the stabilization loss ramp-up')

    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss　between '
                             'the logits with the given weight (default: only have one output)')

    # for Multiple Student
    parser.add_argument('--model-num', default=2, type=int, metavar='MS',
                        help='number of the student models during training, which is required by '
                             ' multiple_student.py [set it to 2 is equal to Dual Student] (default: 2)')

    return parser


def parser_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args: %s", " ".join(cmdline_args))

    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    return mt_cli.str2bool(v)


def str2epochs(v):
    return mt_cli.str2epochs(v)
