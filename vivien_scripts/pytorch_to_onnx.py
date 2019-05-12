from __future__ import unicode_literals
from itertools import repeat
import os
import torch
import json
import onmt
from argparse import Namespace
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from vivien_scripts.coreml_export_models import decompose_bnn_lstm, EncoderForCoreMLExport


dataset_root_path = '/Volumes/Extreme SSD/JP_EN_Translation_Data/pairwise_sentences'
dataset_files = [file for file in os.listdir(dataset_root_path) if not file.startswith('.')]

model_file_folder = '/Volumes/Extreme SSD/OpenNMTModels'
model_file_name = 'demo_jp-en_20k-model_step_800000.pt'
model_file_path = os.path.join(model_file_folder, model_file_name)



def test_rnn_and_coreml_models_equality(rnn_model, coreml_model):

    input_size = rnn_model.input_size
    hidden_size = rnn_model.hidden_size
    num_layers = rnn_model.num_layers
    bidirectional = rnn_model.bidirectional

    input = torch.randn(1, 1, input_size)

    hidden_state = torch.randn(num_layers * (1 + bidirectional), 1, hidden_size)
    cell_state = torch.randn(num_layers * (1 + bidirectional), 1, hidden_size)

    hidden_list = [(hidden_state[0,:,:], cell_state[0,:,:]), (hidden_state[1,:,:], cell_state[1,:,:]), (hidden_state[2,:,:], cell_state[2,:,:]), (hidden_state[3,:,:], cell_state[3,:,:])]

    rnn_model.eval()
    coreml_model.eval()

    rnn_model_output, rnn_model_hidden = rnn_model(input, (hidden_state, cell_state))
    coreml_model_output, coreml_model_hidden = coreml_model(input, hidden_list)

    rnn_model_output = rnn_model_output.squeeze()
    coreml_model_output = coreml_model_output.squeeze()

    relative_diff = torch.sqrt(torch.sum((rnn_model_output - coreml_model_output)**2)).detach() / \
                    torch.sqrt(torch.sum(coreml_model_output**2)).detach()

    print('RNN / CoreML models relative output diff = {}'.format(relative_diff))

    assert relative_diff < 0.001




def pytorch_to_onnx(opt):

    # opt.log_file = logging_file_path
    opt.model = model_file_path
    opt.n_best = 1
    opt.beam_size = 5
    opt.report_bleu = False
    opt.report_rouge = False

    # logger = init_logger(opt.log_file)
    # translator = build_translator(opt, report_score=True)

    model = onmt.model_builder.load_test_model(opt)
    print(model)

    embeddings = model[1].encoder.embeddings
    embeddings_input = torch.zeros((1, 1)).long()
    input_names = ['input_index']
    output_names = ['embedding']
    torch.onnx.export(embeddings, embeddings_input,
                      os.path.join(model_file_folder, 'embeddings.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names)


    encoder = model[1].encoder.rnn
    encoder_models = decompose_bnn_lstm(encoder)
    coreml_encoder = EncoderForCoreMLExport(encoder.input_size, encoder.hidden_size, decomposed_model_list=encoder_models, num_layers=encoder.num_layers, bidirectional=encoder.bidirectional)
    test_rnn_and_coreml_models_equality(encoder, coreml_encoder)
    num_directions = 1 + encoder.bidirectional
    for layer_index in range(encoder.number_layer):
        for direction in range(num_directions):
            encoder_model_part = encoder_models[layer_index * num_directions + direction]
            encoder_input = (torch.randn(1, 1, encoder.input_size if layer_index == 0 else encoder.hidden_size * num_directions),
                             encoder_model_part.init_hidden())
            input_names = ['actual_input_1', 'hidden_input_1', 'cell_intput_1']
            output_names = ['h', 'c']
            torch.onnx.export(encoder_model_part, encoder_input,
                              os.path.join(model_file_folder, 'encoder_model_{}.onnx'.format(layer_index * num_directions + direction)),
                              verbose=True, input_names=input_names, output_names=output_names)

    decoder = model[1].decoder





    # torch.onnx.export(model, dummy_input, os.path.join('../training/checkpoints', checkpoint_folder, 'model.onnx'),
    #                   verbose=True, input_names=input_names, output_names=output_names)



def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    pytorch_to_onnx(opt)





