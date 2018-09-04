import os

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo


model_urls = {
    'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-8f474287.pth'
}

MODEL_CACHE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.torch')


class MTLSTM(nn.Module):

    def __init__(self, n_vocab=None, vectors=None, residual_embeddings=False, layer0=False, layer1=True, trainable=False, model_cache=MODEL_CACHE):
        """Initialize an MTLSTM. If layer0 and layer1 are True, they are concatenated along the last dimension so that layer0 outputs
           contribute the first 600 entries and layer1 contributes the second 600 entries. If residual embeddings is also true, inputs
           are also concatenated along the last dimension with any outputs such that they form the first 300 entries.
         
        Arguments:
            n_vocab (int): If not None, initialize MTLSTM with an embedding matrix with n_vocab vectors
            vectors (Float Tensor): If not None, initialize embedding matrix with specified vectors (These should be 300d CommonCrawl GloVe vectors)
            residual_embedding (bool): If True, concatenate the input GloVe embeddings with contextualized word vectors as final output
            layer0 (bool): If True, return the outputs of the first layer of the MTLSTM
            layer1 (bool): If True, return the outputs of the second layer of the MTLSTM
            trainable (bool): If True, do not detach outputs; i.e. train the MTLSTM (recommended to leave False)
            model_cache (str): path to the model file for the MTLSTM to load pretrained weights (defaults to the best MTLSTM from (McCann et al. 2017) -- 
                               that MTLSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.
        """
        super(MTLSTM, self).__init__()
        self.layer0 = layer0
        self.layer1 = layer1
        self.residual_embeddings = residual_embeddings
        self.trainable = trainable
        self.embed = False
        if n_vocab is not None:
            self.embed = True
            self.vectors = nn.Embedding(n_vocab, 300)
            if vectors is not None:
                self.vectors.weight.data = vectors
        state_dict = model_zoo.load_url(model_urls['wmt-lstm'], model_dir=model_cache)
        if layer0:
            layer0_dict = {k: v for k, v in state_dict.items() if 'l0' in k}
            self.rnn0 = nn.LSTM(300, 300, num_layers=1, bidirectional=True, batch_first=True)
            self.rnn0.load_state_dict(layer0_dict)
            if layer1:
                layer1_dict = {k.replace('l1', 'l0'): v for k, v in state_dict.items() if 'l1' in k}
                self.rnn1 = nn.LSTM(600, 300, num_layers=1, bidirectional=True, batch_first=True)
                self.rnn1.load_state_dict(layer1_dict)
        elif layer1:
            self.rnn1 = nn.LSTM(300, 300, num_layers=2, bidirectional=True, batch_first=True)
            self.rnn1.load_state_dict(model_zoo.load_url(model_urls['wmt-lstm'], model_dir=model_cache))
        else:
            raise ValueError('At least one of layer0 and layer1 must be True.')
         

    def forward(self, inputs, lengths, hidden=None):
        """
        Arguments:
            inputs (Tensor): If MTLSTM handles embedding, a Long Tensor of size (batch_size, timesteps).
                             Otherwise, a Float Tensor of size (batch_size, timesteps, features).
            lengths (Long Tensor): lenghts of each sequence for handling padding
            hidden (Float Tensor): initial hidden state of the LSTM
        """
        if self.embed:
            inputs = self.vectors(inputs)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.Tensor(lengths).long()
            if inputs.is_cuda:
                with torch.cuda.device_of(inputs):
                    lengths = lengths.cuda(torch.cuda.current_device())
        lens, indices = torch.sort(lengths, 0, True)
        outputs = [inputs] if self.residual_embeddings else []
        len_list = lens.tolist()
        packed_inputs = pack(inputs[indices], len_list, batch_first=True)

        if self.layer0:
            outputs0, hidden_t0 = self.rnn0(packed_inputs, hidden)
            unpacked_outputs0 = unpack(outputs0, batch_first=True)[0]
            _, _indices = torch.sort(indices, 0)
            unpacked_outputs0 = unpacked_outputs0[_indices]
            outputs.append(unpacked_outputs0)
            packed_inputs = outputs0
        if self.layer1:
            outputs1, hidden_t1 = self.rnn1(packed_inputs, hidden)
            unpacked_outputs1 = unpack(outputs1, batch_first=True)[0]
            _, _indices = torch.sort(indices, 0)
            unpacked_outputs1 = unpacked_outputs1[_indices]
            outputs.append(unpacked_outputs1)

        outputs = torch.cat(outputs, 2)
        return outputs if self.trainable else outputs.detach()
