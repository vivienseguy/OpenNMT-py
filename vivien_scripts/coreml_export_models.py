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

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.decomposed_model_list = decomposed_model_list


    def init_hidden_list(self, batch_size):
        hidden_input_list = []
        for _ in range(len(self.rnn_cell_list)):
            hidden_state_zeros = torch.zeros(batch_size, self.hidden_size)
            cell_state_zeros = torch.zeros(batch_size, self.hidden_size)
            if self.gpu:
                hidden_state_zeros = hidden_state_zeros.cuda()
                cell_state_zeros = cell_state_zeros.cuda()
            hidden_input_list.append((hidden_state_zeros, cell_state_zeros))
        return hidden_input_list


    def forward(self, x, hidden_input_list=None):

        seq_length, batch_size, input_size = x.shape

        hidden_input_list = hidden_input_list if hidden_input_list else self.init_hidden_list(batch_size)

        input_sequence = list(x.split(1))
        output_sequence = []
        output_sequence_reverse = []

        for layer_index in range(self.num_layers):

            rnn_cell = self.decomposed_model_list[layer_index * (1 + self.bidirectional)]

            for t in range(len(input_sequence)):
                h_t, c_t = rnn_cell(input_sequence[t].view([1, -1]), hidden_input_list[layer_index * (1 + self.bidirectional)] if t == 0 else (h_t, c_t))
                output_sequence.append(h_t)

            if self.bidirectional:

                rnn_cell = self.decomposed_model_list[layer_index * (1 + self.bidirectional) + 1]

                for t in range(len(input_sequence)-1, -1, -1):
                    h_t_reverse, c_t_reverse = rnn_cell(input_sequence[t].view([1, -1]), hidden_input_list[layer_index * (1 + self.bidirectional) + 1] if t == 0 else (h_t_reverse, c_t_reverse))
                    output_sequence_reverse.append(h_t_reverse)

                outputs_concat = []
                for t in range(len(input_sequence)):
                    outputs_concat.append(torch.cat([output_sequence[t], output_sequence_reverse[len(input_sequence)-1-t]], dim=1))

                input_sequence = outputs_concat

            else:

                input_sequence = output_sequence

        return torch.cat([h_t, h_t_reverse], dim=1), 0.












        # for layer_index in range(self.num_layers):
        #
        #     rnn_cell = self.decomposed_model_list[layer_index * (1 + self.bidirectional)]
        #     h, c = rnn_cell(x, hidden if hidden else self.init_hidden(batch_size))




        return h, c






class SingleLayerLSTMForCoreMLExport(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SingleLayerLSTMForCoreMLExport, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.rnn_hidden_size)
        c = torch.zeros(batch_size, self.rnn_hidden_size)
        return (h, c)

    def forward(self, x, hidden=None):
        batch_size, input_size = x.shape
        assert batch_size == 1
        h, c = self.rnn_cell(x, hidden if hidden else self.init_hidden(batch_size))
        return h, c

