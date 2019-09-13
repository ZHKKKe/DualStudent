import os

from third_party.mean_teacher.run_context import TrainLog as MT_TrainLog
from third_party.mean_teacher.run_context import RunContext as MT_RunContext


class RunContext(MT_RunContext):
    def __init__(self, runner_file, run_idx):
        super().__init__(runner_file, run_idx)
        self.tmp_dir = self.result_dir + '/tmp'

        os.makedirs(self.tmp_dir)
