#!/bin/python
import sys
import warnings
warnings.filterwarnings("ignore")
from tensor2tensor.data_generators import generator_utils

# Next check is not necessary, it is only to show that you should not use whole parent corus
print("Checking length of file")
with open("mixed.txt") as f:
    for i, l in enumerate(f):
        pass
    
    if i > 200000 or i < 100000:
        print("Your 'mixed.txt' does not contain 150k rows which means that you have not balanced all languages. In our toy example, this would not be a problem, however whenever you are dealing with high-resource parent and low-resource child. It can quickly happen, that most of the mixed corpora would contain only parent sentences and then the generated vocabulary would contain mainly parent subwords.")
        sys.exit(0)

def get_generator():
    with open("mixed.txt") as f:
        for line in f:
            yield line.strip()

gen = get_generator()
print("Generating vocabulary. It will take a moment. Please, read the next section of tutorial.\n\n")
generator_utils.get_or_generate_vocab_inner("t2t_data", "vocab.cseten.wp", 32000, gen)
