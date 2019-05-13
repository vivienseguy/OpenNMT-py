from __future__ import unicode_literals
from itertools import repeat
import os
import torch
import re
import json
import onmt
from argparse import Namespace
from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from vivien_scripts.coreml_export_models import decompose_bnn_lstm, EncoderForCoreMLExport, MultiLayerLSTMForCoreMLExport


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
    num_directions = 1 + bidirectional

    input = torch.randn(1, 1, input_size)

    hidden_state = torch.randn(num_layers * (1 + bidirectional), 1, hidden_size)
    cell_state = torch.randn(num_layers * (1 + bidirectional), 1, hidden_size)

    hidden_list = []
    for layer_index in range(num_layers):
        for direction in range(num_directions):
            hidden_list.append((hidden_state[layer_index * num_directions + direction,:,:], cell_state[layer_index * num_directions + direction,:,:]))

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


def translate_with_submodels(src_file, fields, src_embeddings, decomposed_encoders, tgt_embeddings, decoder_rnn, attn_linear_in,
                             attn_linear_out, generator):

    src_field = dict(fields)["src"].base_field
    src_vocab = src_field.vocab
    src_eos_idx = src_vocab.stoi[src_field.eos_token]
    src_pad_idx = src_vocab.stoi[src_field.pad_token]
    src_bos_idx = src_vocab.stoi[src_field.init_token]
    src_unk_idx = src_vocab.stoi[src_field.unk_token]
    src_vocab_len = len(src_vocab)

    tgt_field = dict(fields)["tgt"].base_field
    tgt_vocab = tgt_field.vocab
    tgt_eos_idx = tgt_vocab.stoi[tgt_field.eos_token]
    tgt_pad_idx = tgt_vocab.stoi[tgt_field.pad_token]
    tgt_bos_idx = tgt_vocab.stoi[tgt_field.init_token]
    tgt_unk_idx = tgt_vocab.stoi[tgt_field.unk_token]
    tgt_vocab_len = len(tgt_vocab)







    return



def pytorch_to_onnx(opt):

    opt.model = model_file_path
    opt.n_best = 1
    opt.beam_size = 5
    opt.report_bleu = False
    opt.report_rouge = False

    translator = build_translator(opt, report_score=True)
    translator.translate(src='src-test.txt', batch_size=1)

    model = onmt.model_builder.load_test_model(opt)
    print(model)


    src_embeddings = model[1].encoder.embeddings
    embeddings_input = torch.zeros((1, 1, 1)).long()
    input_names = ['input_index']
    output_names = ['embedding']
    torch.onnx.export(src_embeddings, embeddings_input,
                      os.path.join(model_file_folder, 'src_embeddings.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names
                      )


    encoder = model[1].encoder.rnn
    encoder_models = decompose_bnn_lstm(encoder)
    coreml_encoder = EncoderForCoreMLExport(encoder.input_size, encoder.hidden_size, decomposed_model_list=encoder_models, num_layers=encoder.num_layers, bidirectional=encoder.bidirectional)
    test_rnn_and_coreml_models_equality(encoder, coreml_encoder)
    num_directions = 1 + encoder.bidirectional
    for layer_index in range(encoder.num_layers):
        for direction in range(num_directions):
            encoder_model_part = encoder_models[layer_index * num_directions + direction]
            encoder_input = (torch.randn(1, encoder.input_size if layer_index == 0 else encoder.hidden_size * num_directions),
                             encoder_model_part.init_hidden(batch_size=1))
            input_names = ['input', 'h', 'c']
            output_names = ['h', 'c']
            torch.onnx.export(encoder_model_part, encoder_input,
                              os.path.join(model_file_folder, 'encoder_model_{}.onnx'.format(layer_index * num_directions + direction)),
                              verbose=True, input_names=input_names, output_names=output_names)


    tgt_embeddings = model[1].decoder.embeddings
    embeddings_input = torch.zeros((1, 1, 1)).long()
    input_names = ['input_index']
    output_names = ['embedding']
    torch.onnx.export(tgt_embeddings, embeddings_input,
                      os.path.join(model_file_folder, 'tgt_embeddings.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names
                      )


    decoder_rnn = model[1].decoder.rnn
    state_dict = decoder_rnn.state_dict()
    coreml_decoder_rnn = MultiLayerLSTMForCoreMLExport(decoder_rnn.input_size, decoder_rnn.hidden_size, num_layers=decoder_rnn.num_layers)
    coreml_model_state_dict = coreml_decoder_rnn.state_dict()
    for layer_index in range(decoder_rnn.num_layers):
        for coreml_key in coreml_model_state_dict:
            if 'rnn_cell' in coreml_key:
                rnn_key = re.sub(r'rnn_cell\d\.', '', coreml_key) + '_l' + '{}'.format(layer_index)
                coreml_model_state_dict[coreml_key] = state_dict[rnn_key]
        coreml_decoder_rnn.load_state_dict(coreml_model_state_dict)
    decoder_rnn_input = (torch.randn(1, decoder_rnn.input_size),
                         coreml_decoder_rnn.init_hidden(batch_size=1))
    input_names = ['input']
    output_names = []
    for layer_index in range(decoder_rnn.num_layers):
        input_names += ['h{}'.format(layer_index), 'c{}'.format(layer_index)]
        output_names += ['h{}'.format(layer_index), 'c{}'.format(layer_index)]
    torch.onnx.export(coreml_decoder_rnn, decoder_rnn_input,
                      os.path.join(model_file_folder, 'decoder_rnn_model.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names)


    attn = model[1].decoder.attn
    input_names = ['rnn_output']
    output_names = ['output']
    rnn_output = torch.rand(1, 1, decoder_rnn.hidden_size)
    torch.onnx.export(attn.linear_in, (rnn_output, ),
                      os.path.join(model_file_folder, 'attn_linear_in.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names
                      )
    input_names = ['input']
    output_names = ['output']
    input = torch.rand(1, 1, 2 * decoder_rnn.hidden_size)
    torch.onnx.export(attn.linear_out, (input, ),
                      os.path.join(model_file_folder, 'attn_linear_out.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names
                      )


    generator = model[1].generator
    input_names = ['input']
    output_names = ['output']
    input = torch.rand(1, decoder_rnn.hidden_size)
    torch.onnx.export(generator, (input, ),
                      os.path.join(model_file_folder, 'generator.onnx'),
                      verbose=True, input_names=input_names, output_names=output_names
                      )


    translate_with_submodels(src_file='src-test.txt', fields=translator.fields, src_embeddings=src_embeddings, decomposed_encoders=encoder_models,
                             tgt_embeddings=tgt_embeddings, decoder_rnn=coreml_decoder_rnn, attn_linear_in=attn.linear_in,
                             attn_linear_out=attn.linear_out, generator=generator)



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





