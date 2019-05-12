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
from vivien_scripts.coreml_export_models import decompose_bnn_lstm


dataset_root_path = '/Volumes/Extreme SSD/JP_EN_Translation_Data/pairwise_sentences'
dataset_files = [file for file in os.listdir(dataset_root_path) if not file.startswith('.')]
model_file_path = '/Volumes/Extreme SSD/OpenNMTModels/demo_jp-en_20k-model_step_800000.pt'


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

    encoder = model[1].encoder.rnn
    encoder_models = decompose_bnn_lstm(encoder)

    dummy_input = torch.randn(1, 1, 10, 10)

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
