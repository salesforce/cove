import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo


model_urls = {
    'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth'
}

MODEL_CACHE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.torch')


class MTLSTM(nn.Module):

    def __init__(self, n_vocab=None, vectors=None, residual_embeddings=False, model_cache=MODEL_CACHE):
        """Initialize an MTLSTM.
         
        Arguments:
            n_vocab (bool): If not None, initialize MTLSTM with an embedding matrix with n_vocab vectors
            vectors (Float Tensor): If not None, initialize embedding matrix with specified vectors
            residual_embedding (bool): If True, concatenate the input embeddings with MTLSTM outputs during forward
        """
        super(MTLSTM, self).__init__()
        self.embed = False
        if n_vocab is not None:
            self.embed = True
            self.vectors = nn.Embedding(n_vocab, 300)
            if vectors is not None:
                self.vectors.weight.data = vectors
        self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn.load_state_dict(model_zoo.load_url(model_urls['wmt-lstm'], model_dir=model_cache))
        self.residual_embeddings = residual_embeddings

    def forward(self, inputs, lengths, hidden=None):
        """A pretrained MT-LSTM (McCann et. al. 2017). 
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.
     
        Arguments:
            inputs (Tensor): If MTLSTM handles embedding, a Long Tensor of size (batch_size, timesteps).
                             Otherwise, a Float Tensor of size (batch_size, timesteps, features).
            lengths (Long Tensor): (batch_size, lengths) lenghts of each sequence for handling padding
            hidden (Float Tensor): initial hidden state of the LSTM
        """
        if self.embed:
            inputs = self.vectors(inputs)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.Tensor(lengths).long()
            if inputs.is_cuda:
                with torch.cuda.device_of(inputs):
                    lengths.cuda(torch.cuda.current_device())
        lens, indices = torch.sort(lengths, 0, True)
        outputs, hidden_t = self.rnn(pack(inputs[indices], lens.tolist(), batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        if self.residual_embeddings:
            outputs = torch.cat([inputs, outputs], 2)
        return outputs
