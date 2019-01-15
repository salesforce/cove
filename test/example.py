from argparse import ArgumentParser
import numpy as np

import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from cove import MTLSTM


parser = ArgumentParser()
parser.add_argument('--device', default=0, help='Which device to run one; -1 for CPU', type=int)
parser.add_argument('--data', default='.data', help='where to store data')
parser.add_argument('--embeddings', default='.embeddings', help='where to store embeddings')
args = parser.parse_args()

inputs = data.Field(lower=True, include_lengths=True, batch_first=True)

print('Generating train, dev, test splits')
train, dev, test = datasets.IWSLT.splits(root=args.data, exts=['.en', '.de'], fields=[inputs, inputs])
train_iter, dev_iter, test_iter = data.Iterator.splits(
            (train, dev, test), batch_size=100, device=torch.device(args.device) if args.device >= 0 else None)

print('Building vocabulary')
inputs.build_vocab(train, dev, test)
inputs.vocab.load_vectors(vectors=GloVe(name='840B', dim=300, cache=args.embeddings))

outputs_last_layer_cove = MTLSTM(n_vocab=len(inputs.vocab), vectors=inputs.vocab.vectors, model_cache=args.embeddings)
outputs_both_layer_cove = MTLSTM(n_vocab=len(inputs.vocab), vectors=inputs.vocab.vectors, layer0=True, model_cache=args.embeddings)
outputs_both_layer_cove_with_glove = MTLSTM(n_vocab=len(inputs.vocab), vectors=inputs.vocab.vectors, layer0=True, residual_embeddings=True, model_cache=args.embeddings)

if args.device >= 0:
    outputs_last_layer_cove.cuda()
    outputs_both_layer_cove.cuda()
    outputs_both_layer_cove_with_glove.cuda()

train_iter.init_epoch()
print('Generating CoVe')
for batch_idx, batch in enumerate(train_iter):
    if batch_idx > 0:
        break
    last_layer_cove = outputs_last_layer_cove(*batch.src)
    print(last_layer_cove.size())
    first_then_last_layer_cove = outputs_both_layer_cove(*batch.src)
    print(first_then_last_layer_cove.size())
    glove_then_first_then_last_layer_cove = outputs_both_layer_cove_with_glove(*batch.src)
    print(glove_then_first_then_last_layer_cove.size())
    assert np.allclose(last_layer_cove, first_then_last_layer_cove[:, :, -600:])
    assert np.allclose(last_layer_cove, glove_then_first_then_last_layer_cove[:, :, -600:])
    assert np.allclose(first_then_last_layer_cove[:, :, :600], glove_then_first_then_last_layer_cove[:, :, 300:900])
    print(last_layer_cove[:, :, -10:])
    print(first_then_last_layer_cove[:, :, -10:])
    print(glove_then_first_then_last_layer_cove[:, :, -10:])
