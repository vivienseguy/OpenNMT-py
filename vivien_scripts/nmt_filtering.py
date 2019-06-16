#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
import os
import torch
import time
import random

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


dataset_root_path = '/Volumes/Extreme SSD/JP_EN_Translation_Data/all_sentences'
dataset_files = [file for file in os.listdir(dataset_root_path) if not file.startswith('.')]
model_file_path = '/Volumes/Extreme SSD/OpenNMTModels/demo_jp-en_20k-model_step_1500000.pt'
logging_file_path = '/Volumes/Extreme SSD/OpenNMTModels/per_dataset_eval_log_1500000.txt'

src_file = 'jp_sentences_val_spaced_filtered.txt'
tgt_file = 'en_sentences_val_spaced_filtered.txt'


def nmt_filter_dataset(opt):

    opt.src = os.path.join(dataset_root_path, src_file)
    opt.tgt = os.path.join(dataset_root_path, tgt_file)
    opt.shard_size = 1

    opt.log_file = logging_file_path
    opt.models = [model_file_path]
    opt.n_best = 1
    opt.beam_size = 1
    opt.report_bleu = False
    opt.report_rouge = False

    logger = init_logger(opt.log_file)
    translator = build_translator(opt, report_score=True)

    src_file_path = os.path.join(dataset_root_path, src_file)
    tgt_file_path = os.path.join(dataset_root_path, tgt_file)

    src_shards = split_corpus(src_file_path, opt.shard_size)
    tgt_shards = split_corpus(tgt_file_path, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    pred_scores = []

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):

        start_time = time.time()
        shard_pred_scores, shard_pred_sentences = translator.translate(src=src_shard,
                                                                      tgt=tgt_shard,
                                                                      src_dir=opt.src_dir,
                                                                      batch_size=opt.batch_size,
                                                                      attn_debug=opt.attn_debug
                                                                      )
        print("--- %s seconds ---" % (time.time() - start_time))

        pred_scores += [scores[0] for scores in shard_pred_scores]

    average_pred_score = torch.mean(torch.stack(pred_scores)).detach()

    return average_pred_score





def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    opt = parser.parse_args()
    nmt_filter_dataset(opt)
