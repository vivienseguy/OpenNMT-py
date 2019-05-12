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


dataset_root_path = '/Volumes/Extreme SSD/JP_EN_Translation_Data/pairwise_sentences'
dataset_files = [file for file in os.listdir(dataset_root_path) if not file.startswith('.')]
model_file_path = '/Volumes/Extreme SSD/OpenNMTModels/demo_jp-en_20k-model_step_800000.pt'
logging_file_path = '/Volumes/Extreme SSD/OpenNMTModels/per_dataset_eval_log_800000.txt'
sentences_per_dataset_max = 1000

dataset_files = ['twitter_jp_spaced.txt', 'twitter_en_spaced.txt', 'wat2017_jp_spaced.txt', 'wat2017_en_spaced.txt'] # test


def evaluate_translation_on_datasets(opt):

    opt.log_file = logging_file_path
    opt.models = [model_file_path]
    opt.n_best = 1
    opt.beam_size = 5
    opt.report_bleu = False
    opt.report_rouge = False

    logger = init_logger(opt.log_file)
    translator = build_translator(opt, report_score=True)

    for dataset_file in dataset_files:

        if '_jp_spaced.txt' in dataset_file:

            src_file_path = os.path.join(dataset_root_path, dataset_file)
            tgt_file_path = os.path.join(dataset_root_path, dataset_file[:-len('_jp_spaced.txt')] + '_en_spaced.txt')

            num_lines = sum(1 for line in open(tgt_file_path))

            if num_lines > sentences_per_dataset_max:

                src_tmp_file_path = src_file_path[:-4] + '_tmp.txt'
                tgt_tmp_file_path = tgt_file_path[:-4] + '_tmp.txt'

                with open(src_file_path, 'r') as src_file, open(tgt_file_path, 'r') as tgt_file:

                    src_lines = src_file.read().splitlines()
                    tgt_lines = tgt_file.read().splitlines()

                    pairs = list(zip(src_lines, tgt_lines))
                    random.shuffle(pairs)

                    pairs = pairs[:sentences_per_dataset_max]

                    with open(src_tmp_file_path, 'w') as src_tmp_file, open(tgt_tmp_file_path, 'w') as tgt_tmp_file:
                        for pair in pairs:
                            src_tmp_file.write(pair[0]+'\n')
                            tgt_tmp_file.write(pair[1]+'\n')

                src_file_path = src_tmp_file_path
                tgt_file_path = tgt_tmp_file_path

            opt.src = src_file_path
            opt.tgt = tgt_file_path

            ArgumentParser.validate_translate_opts(opt)

            average_pred_score = evaluate_translation(translator, opt, src_file_path, tgt_file_path)

            logger.info('{}: {}'.format(dataset_file, average_pred_score))

            if num_lines > sentences_per_dataset_max:
                os.remove(src_tmp_file_path)
                os.remove(tgt_tmp_file_path)




def evaluate_translation(translator, opt, src, tgt):

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size) if opt.tgt is not None else repeat(None)
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
    evaluate_translation_on_datasets(opt)
