import sys
import logging

import torch

from src.cli import parse_dict_args
from src.run_context import RunContext
import dual_student


LOG = logging.getLogger('main')


def parameters():
    defaults = {
        # global
        'resume': './checkpoints/ds_cifar100_10000l_cnn13.300.ckpt',
        'validation': True,

        # data
        'dataset': 'cifar100',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',
        'workers': 2,

        # optimization
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,
        # 'exclude_unlabeled': True,

        'base_lr': 0.2,
        'nesterov': True,
        'weight_decay': 2e-4,

        'checkpoint_epochs': 20,

        # architecture
        'arch': 'cnn13',

        # constraint
        'consistency_scale': 10.0,
        'consistency_rampup': 5,

        'stable_threshold': 0.4,
        'stabilization_scale': 100.0,
        'stabilization_rampup': 5,

        'logit_distance_cost': 0.01,

    }

    # 10000 labels:
    for data_seed in range(10, 19):
        yield {
            **defaults,
            'title': 'ds_cifar100_10000l_cnn13.',
            'n_labels': 10000,
            'data_seed': data_seed,
            'epochs': 300,
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)
    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'third_party/data-local/labels/cifar100/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    fh = logging.FileHandler('{0}/log.txt'.format(context.result_dir))
    fh.setLevel(logging.INFO)
    LOG.addHandler(fh)
    dual_student.args = parse_dict_args(**adapted_args, **kwargs)
    dual_student.main(context)


if __name__ == '__main__':
    for run_params in parameters():
        run(**run_params)
