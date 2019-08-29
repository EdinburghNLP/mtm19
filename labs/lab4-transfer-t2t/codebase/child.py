from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

_datasets = {
    'train': [["", ("train.src.TODO", "train.trg.TODO")]],
    'dev': [["", ("dev.src.TODO", "dev.trg.TODO")]],
}

@registry.register_problem
class childProblem(translate.TranslateProblem):
    @property
    def vocab_filename(self):
        return "vocab.TODO"

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _datasets['train'] if train else _datasets['dev']


