from os import path, environ, makedirs
import logging
import requests
from argparse import ArgumentParser

import torch



def cache_glove(glove_prefix):
    stoi = {}
    itos = []
    vectors = []
    fname = glove_prefix+'.txt'

    with open(fname, 'rb') as f:
        for l in f:
            l = l.strip().split(b' ')
            word, vector = l[0], l[1:]
            try:
                word = word.decode()
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            stoi[word] =  len(itos)
            itos.append(word)
            vectors.append([float(x) for x in vector])
    d = {'stoi': stoi, 'itos': itos, 'vectors': torch.FloatTensor(vectors)}
    torch.save(d, glove_prefix+'.pt')

def cache_chargrams():
    stoi = {}
    itos = []
    vectors = []
    fname = 'kazuma1.emb'

    with open(fname, 'rb') as f:
        for l in f:
            l = l.strip().split(b' ')
            word = l[0]
            vector = [float(n) for n in l[1:]]

            try:
                word = word.decode()
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue

            stoi[word] =  len(itos)
            itos.append(word)
            vectors.append(vector)

    d = {'stoi': stoi, 'itos': itos, 'vectors': torch.FloatTensor(vectors)}
    torch.save(d, 'kazuma.100d.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('embeddings')
    parser.add_argument('-glove_prefix', default='glove.840B.300d', type=str)
    args = parser.parse_args()
    
    if args.embeddings == 'glove':
        cache_glove(args.glove_prefix)
    else:
        cache_chargrams()

