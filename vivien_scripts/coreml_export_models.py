import re
import torch
import torch.nn as nn



def decompose_bnn_lstm(encoder):

    state_dict = encoder.state_dict()

    input_size = encoder.input_size
    hidden_size = encoder.hidden_size
    num_layer = encoder.num_layers
    bidirectional = encoder.bidirectional

    model_list = []

    for layer_index in range(num_layer):

        coreml_model = SingleLayerLSTMForCoreMLExport(input_size, hidden_size)
        coreml_model_state_dict = coreml_model.state_dict()
        for coreml_key in coreml_model_state_dict:
            if 'rnn_cell' in coreml_key:
                rnn_key = re.sub(r'rnn_cell\.', '', coreml_key) + '_l' + '{}'.format(layer_index)
                coreml_model_state_dict[coreml_key] = state_dict[rnn_key]
        coreml_model.load_state_dict(coreml_model_state_dict)
        model_list.append(coreml_model)

        if bidirectional:

            coreml_model = SingleLayerLSTMForCoreMLExport(input_size, hidden_size)
            coreml_model_state_dict = coreml_model.state_dict()
            for coreml_key in coreml_model_state_dict:
                rnn_key = coreml_key
                if 'rnn_cell' in coreml_key:
                    rnn_key = re.sub(r'rnn_cell\.', '', coreml_key) + '_l' + '{}'.format(layer_index) + '_reverse'
                coreml_model_state_dict[coreml_key] = state_dict[rnn_key]
            coreml_model.load_state_dict(coreml_model_state_dict)
            model_list.append(coreml_model)

            input_size = 2 * hidden_size

    return model_list





class EncoderForCoreMLExport(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, decomposed_model_list=None):

        super(EncoderForCoreMLExport, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.decomposed_model_list = decomposed_model_list


    def init_hidden(self, batch_size):
        hidden_state_zeros = torch.zeros(batch_size, self.rnn_hidden_size)
        return hidden_state_zeros


    def forward(self, x, hidden=None):
        



        seq_length, batch_size, input_size = x.shape
        assert batch_size == 1
        h, c = self.rnn_cell(x, hidden if hidden else self.init_hidden(batch_size))
        return h, c






class SingleLayerLSTMForCoreMLExport(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SingleLayerLSTMForCoreMLExport, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def init_hidden(self, batch_size):
        hidden_state_zeros = torch.zeros(batch_size, self.rnn_hidden_size)
        return hidden_state_zeros

    def forward(self, x, hidden=None):
        batch_size, input_size = x.shape
        assert batch_size == 1
        h, c = self.rnn_cell(x, hidden if hidden else self.init_hidden(batch_size))
        return h, c

