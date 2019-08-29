from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
import tensorflow as tf

_datasets = {
    'train': [["", ("train_parent.src", "train_parent.trg")]],
    'dev': [["", ("dev_parent.src", "dev_parent.trg")]],
    }
@registry.register_problem
class parentProblem(translate.TranslateProblem):
    @property
    def vocab_filename(self):
        return "vocab.cseten.wp"

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _datasets['train'] if train else _datasets['dev']

