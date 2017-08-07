import torch
from torchtext import data
from torchtext import datasets

from cove import MTLSTM


inputs = data.Field(lower=True, include_lengths=True, batch_first=True)
answers = data.Field(sequential=False)

print('Generating train, dev, test splits')
train, dev, test = datasets.SNLI.splits(inputs, answers)

print('Building vocabulary')
inputs.build_vocab(train, dev, test)
inputs.vocab.load_vectors(wv_type='glove.840B', wv_dim=300)
answers.build_vocab(train)

model = MTLSTM(n_vocab=len(inputs.vocab), vectors=inputs.vocab.vectors)
model.cuda(0)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=100, device=0)

train_iter.init_epoch()
print('Generating CoVe')
for batch_idx, batch in enumerate(train_iter):
    model.train()
    cove_premise = model(*batch.premise)
    cove_hypothesis = model(*batch.hypothesis)
