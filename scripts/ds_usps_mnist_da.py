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
        # 'resume': './ds_usps_mnist_da.100.ckpt',
        # 'validation': True,

        # data
        'dataset': 'usps',
        'train_subdir': 'train',
        'eval_subdir': 'test',
        'workers': 2,

        'target_domain': 'mnist',

        # optimization
        'base_batch_size': 256,
        'base_labeled_batch_size': 32,

        'base_lr': 0.1,
        'nesterov': True,
        'weight_decay': 1e-4,

        'checkpoint_epochs': 100,

        # architecture
        'arch': 'cnn3',

        # constraint
        'stable_threshold': 0.6,
        'stabilization_scale': 1.0,
        'stabilization_rampup': 5,

        'consistency_scale': 1.0,
        'consistency_rampup': 5,

        'logit_distance_cost': 0.01,
    }

    # 1000 labels:
    for data_seed in range(0, 1):
        yield {
            **defaults,
            'title': 'ds_usps_mnist_da',
            'n_labels': 'all',
            'data_seed': data_seed,
            'epochs': 100,
        }

            
def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)
    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'third_party/data-local/labels/usps/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    dual_student.args = parse_dict_args(**adapted_args, **kwargs)
    dual_student.main(context)


if __name__ == '__main__':
    for run_params in parameters():
        run(**run_params)
