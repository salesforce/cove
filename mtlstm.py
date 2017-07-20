import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo


model_urls = {
    'wmt-lstm' : 'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth'
}


class MTLSTM(nn.Module):

    def __init__(self, n_vocab):
        super().__init__()
        self.embed = nn.Embedding(n_vocab, 300)
        self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True)
        self.rnn.load_state_dict(model_zoo.load_url(model_urls['wmt-lstm']))

    def forward(self, inputs, lengths, hidden=None):
        """A pretrained MT-LSTM (McCann et. al. 2017). 
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.
     
        Arguments:
            inputs (Float Tensor): (batch_size, timesteps, features) input sequences
            lengths (Long Tensor): (batch_size, lengths) lenghts of each sequence for handling padding
            hidden (Float Tensor): initial hidden state of the LSTM
        """
        inputs = self.embed(inputs.t()).t()
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices]
        outputs, hidden_t = self.rnn(pack(inputs, lens.tolist(), batch_first=True), hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        return outputs
