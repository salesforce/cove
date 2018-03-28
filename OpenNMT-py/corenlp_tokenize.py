import os, sys
import time
sys.path.append(os.getcwd())

from stanza.nlp.corenlp import CoreNLPClient
import json
from tqdm import tqdm
from collections import defaultdict
import re
import gzip
import argparse

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-input-fn', required=True,
                    help="Path to the input english data")
parser.add_argument('-output-fn', required=True,
                    help="Path to the output english data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

corenlp = CoreNLPClient(default_annotators=['tokenize', 'ssplit'])

def annotate_sentence(corenlp, gloss):
    try:
        parse = corenlp.annotate(gloss)
    except:
        time.sleep(10)
        parse = corenlp.annotate(gloss)
    token_str = ' '.join([token['word'] for sentence in parse.json['sentence'] for token in sentence['token'] ])
    #return parse.json['sentence'][0]['token']
    return token_str

with open(opt.input_fn, 'r') as f:
    with open(opt.output_fn, 'w') as f2:
        for sent in f.readlines():
            f2.write(annotate_sentence(corenlp, sent))
            f2.write('\n')
