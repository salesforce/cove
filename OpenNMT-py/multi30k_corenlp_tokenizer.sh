#!/bin/bash
python corenlp_tokenize.py -input-fn data/multi30k/train.en -output-fn data/multi30k/train.corenlp.en
python corenlp_tokenize.py -input-fn data/multi30k/val.en -output-fn data/multi30k/val.corenlp.en
python corenlp_tokenize.py -input-fn data/multi30k/test.en -output-fn data/multi30k/test.corenlp.en
