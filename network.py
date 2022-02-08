#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Nested NER training and evaluation in TensorFlow."""
from enums import STRATEGY
from active_utils import *
from utils import *
from utils import *
from configs import *
from sklearn.model_selection import train_test_split
import os, psutil
from seqeval.metrics import classification_report, f1_score,precision_score,recall_score

from active_utils import *
import json
import os
import sys


import numpy as np
import  tensorflow as tf
from gensim.models import word2vec


import morpho_dataset
from configs import *
import shutil

class Arguments(object):
    def __init__(self, data_path = "./data/teprorary/"):

        self.batch_size=8
        self.bert_embeddings_dev=data_path + "dev_vectors.txt"
        self.bert_embeddings_test=data_path + "test_vectors.txt"
        self.bert_embeddings_train=data_path + "train_vectors.txt"
        self.dev_data = data_path + "dev.txt"
        self.beta_2=0.98
        self.corpus="CoNLL_en"
        self.cle_dim=128
        self.decoding="CRF"

        self.logpath="/mnt/nfs-storage/logs/active/00_loginfo.csv"
        self.dropout=0.5
        self.elmo_dev=None
        self.elmo_test=None
        self.elmo_train=None
        self.epochs="1:1e-3"

        self.flair_dev=None
        self.flair_test=None
        self.flair_train=None
        self.form_wes_model=None
        self.label_smoothing=0
        self.lemma_wes_model=None
        self.max_sentences=None
        self.name=None
        self.predict=None
        self.rnn_cell="LSTM"
        self.rnn_cell_dim=256
        self.rnn_layers=1
        self.test_data=data_path + "test.txt"
        self.train_data=data_path + "train.txt"
        self.threads=4
        self.we_dim=256
        self.word_dropout=0.2


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args,
                  num_tags,  pretrained_bert_dim,
                  predict_only):
        with self.session.graph.as_default():

            # Inputs
            self.sentence_lens =  tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.pretrained_bert_wes = tf.placeholder(tf.float32, [None, None, pretrained_bert_dim], name="bert_wes")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])
            # RNN Cell

            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell

            inputs = []

            # BERT form embeddings
            if pretrained_bert_dim:
                inputs.append(self.pretrained_bert_wes)


            # Concatenate inputs
            inputs = tf.concat(inputs, axis=2)

            # Dropout
            inputs_dropout = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)

            # Computation
            hidden_layer_dropout = inputs_dropout # first layer is input
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim), rnn_cell(args.rnn_cell_dim),
                    hidden_layer_dropout, sequence_length=self.sentence_lens, dtype=tf.float32,
                    scope="RNN-{}".format(i))
                hidden_layer = tf.concat([hidden_layer_fwd, hidden_layer_bwd], axis=2)
                if i == 0: hidden_layer_dropout = 0
                hidden_layer_dropout += tf.layers.dropout(hidden_layer, rate=args.dropout, training=self.is_training)

            # Decoders
            if args.decoding == "CRF": # conditional random fields
                output_layer = tf.layers.dense(hidden_layer_dropout, num_tags)
                weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    output_layer, self.tags, self.sentence_lens)
                loss = tf.reduce_mean(-log_likelihood)
                self.predictions, _ = tf.contrib.crf.crf_decode(
                    output_layer, transition_params, self.sentence_lens)

                seq_score, _ = tf.contrib.crf.crf_log_likelihood(
                    output_layer, self.predictions, self.sentence_lens, transition_params)
                self.viterbi_score = seq_score/tf.cast( self.sentence_lens, tf.float32)
                self.predictions_training = self.predictions


            # Saver
            self.saver = tf.train.Saver(max_to_keep=1)
            if predict_only: return

            # Training
            global_step = tf.train.create_global_step()
            self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=global_step)

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions_training, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics =  tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            self.metrics = {}
            self.metrics_summarize = {}
            for metric in ["precision", "recall", "F1"]:
                self.metrics[metric] = tf.placeholder(tf.float32, [], name=metric)
                self.metrics_summarize[metric] = {}
                with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    for dataset in ["dev", "test"]:
                        self.metrics_summarize[metric][dataset] = tf.contrib.summary.scalar(dataset + "/" + metric,
                                                                                            self.metrics[metric])

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    def train_epoch(self, train, learning_rate, args):
        while not train.epoch_finished():
            batch_dict = train.next_batch(args.batch_size)
            self.session.run(self.reset_metrics)
            feeds = {self.sentence_lens: batch_dict["sentence_lens"],
                     self.tags: batch_dict["word_ids"][train.TAGS],
                     self.is_training: True,
                     self.learning_rate: learning_rate}
            if args.bert_embeddings_train: # BERT embeddings
                feeds[self.pretrained_bert_wes] = batch_dict["batch_bert_wes"]

            self.session.run([self.training, self.summaries["train"]], feeds)


    def evaluate(self, dataset_name, dataset, args):
        with open("{}/{}_system_predictions.conll".format(args.logdir, dataset_name), "w", encoding="utf-8") as prediction_file:
            self.predict(dataset_name, dataset, args, prediction_file, evaluating=True)

        f1 = 0.0
        pr = 0.0
        re = 0.0
        if args.corpus in ["CoNLL_en", "CoNLL_de", "CoNLL_nl", "CoNLL_es"]:
            os.system("cd {} && ../../run_conlleval.sh {} {} {}_system_predictions.conll".format(args.logdir, dataset_name, args.__dict__[dataset_name + "_data"], dataset_name))

            with open("{}/{}.eval".format(args.logdir,dataset_name), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("accuracy:"):
                        f1 = float(line.split()[-1])
                        re = float(line.split()[-3][:-2])
                        pr = float(line.split()[-5][:-2])
                        self.session.run(self.metrics_summarize["F1"][dataset_name], {self.metrics["F1"]: f1})

            return pr, re, f1
        elif args.corpus in [ "ACE2004", "ACE2005", "GENIA" ]: # nested named entities evaluation
            os.system("cd {} && ../../run_eval_nested.sh {} {}".format(args.logdir, dataset_name, os.path.dirname(args.__dict__[dataset_name + "_data"])))

            with open("{}/{}.eval".format(args.logdir,dataset_name), "r", encoding="utf-8") as result_file:
                for line in result_file:
                    line = line.strip("\n")
                    if line.startswith("Recall:"):
                        recall = float(line.split(" ")[1])
                    if line.startswith("Precision:"):
                        precision = float(line.split(" ")[1])
                    if line.startswith("F1:"):
                        f1 = float(line.split(" ")[1])
                        for metric, value in [["precision", precision], ["recall", recall], ["F1", f1]]:
                            self.session.run(self.metrics_summarize[metric][dataset_name], {self.metrics[metric]: value})
            return precision, recall, f1
        else:
            raise ValueError("Unknown corpus {}".format(args.corpus))

    def get_tags(self, dataset, args):
        tags = []
        scors = []
        while not dataset.epoch_finished():
            batch_dict = dataset.next_batch(args.batch_size)
            targets = [self.predictions]
            scores = [self.viterbi_score]

            feeds = {self.sentence_lens: batch_dict["sentence_lens"],
                    self.is_training: False}

            if args.bert_embeddings_dev or args.bert_embeddings_test: # BERT embeddings
                feeds[self.pretrained_bert_wes] = batch_dict["batch_bert_wes"]

            tags.extend(self.session.run(targets, feeds)[0])
            scors.extend(self.session.run(scores, feeds)[0])

        tags_res = []
        scores_res = []
        forms = dataset.factors[dataset.FORMS].strings
        for s in range(len(forms)):
            j = 0
            sent = []
            score_sent = []
            for i in range(len(forms[s])):
                sent.append(dataset.factors[dataset.TAGS].words[tags[s][i]])
            tags_res.append(sent)
            scores_res.append(np.exp(scors[s]))
        return tags_res, scores_res

    def f1_score_span(self, labels, tags=None, isList=True):
        if isList:
           all_labels, all_tags = self.make_list_of_words(tags, labels)
        else:
            all_labels, all_tags  =  labels, tags
        new_pr, new_re, new_f1 = precision_score([all_labels], [all_tags]), recall_score([all_labels], [all_tags]), f1_score([all_labels], [all_tags])
        return new_pr, new_re, new_f1

    def make_list_of_words(self, tags, labels):
        all_tags = []
        all_labels = []

        for tag_n, label_n in zip(tags, labels):
            for tag, label in zip(tag_n, label_n):
                all_tags.append(tag)
                all_labels.append(label)
        return  all_labels, all_tags


    def predict(self, dataset_name, dataset, args, prediction_file, evaluating=False):
        if evaluating:
            self.session.run(self.reset_metrics)
        tags = []
        while not dataset.epoch_finished():
            batch_dict = dataset.next_batch(args.batch_size)
            targets = [self.predictions]

            feeds = {self.sentence_lens: batch_dict["sentence_lens"],
                    self.is_training: False}
            if evaluating:
                targets.extend([self.update_accuracy, self.update_loss])
                feeds[self.tags] = batch_dict["word_ids"][dataset.TAGS]
            if args.bert_embeddings_dev or args.bert_embeddings_test: # BERT embeddings
                feeds[self.pretrained_bert_wes] = batch_dict["batch_bert_wes"]

            tags.extend(self.session.run(targets, feeds)[0])

        if evaluating:
            self.session.run([self.current_accuracy, self.summaries[dataset_name]])

        forms = dataset.factors[dataset.FORMS].strings
        for s in range(len(forms)):
            j = 0
            for i in range(len(forms[s])):
                if args.decoding == "seq2seq": # collect all tags until <eow>
                    labels = []
                    while j < len(tags[s]) and dataset.factors[dataset.TAGS].words[tags[s][j]] != "<eow>":
                        labels.append(dataset.factors[dataset.TAGS].words[tags[s][j]])
                        j += 1
                    j += 1 # skip the "<eow>"
                    print("{} _ _ {}".format(forms[s][i], "|".join(labels)), file=prediction_file)
                else:
                    print("{} _ _ {}".format(forms[s][i], dataset.factors[dataset.TAGS].words[tags[s][i]]), file=prediction_file)
            print("", file=prediction_file)


def train_model(model_config):
    import argparse
    import datetime
    import os
    import re

    path_data = "data/teprorary" + str(model_config.number) + "/"
    args = Arguments(path_data)


    # Create logdir name
    logargs = dict(vars(args).items())
    logargs["form_wes_model"] = 1 if args.form_wes_model else 0
    logargs["lemma_wes_model"] = 1 if args.lemma_wes_model else 0
    del logargs["bert_embeddings_dev"]
    del logargs["bert_embeddings_test"]
    del logargs["bert_embeddings_train"]
    del logargs["beta_2"]
    del logargs["cle_dim"]
    del logargs["dev_data"]
    del logargs["dropout"]
    del logargs["elmo_dev"]
    del logargs["elmo_test"]
    del logargs["elmo_train"]
    del logargs["flair_dev"]
    del logargs["flair_test"]
    del logargs["flair_train"]
    del logargs["label_smoothing"]
    del logargs["max_sentences"]
    del logargs["rnn_cell_dim"]
    del logargs["test_data"]
    del logargs["threads"]
    del logargs["train_data"]
    del logargs["we_dim"]
    del logargs["word_dropout"]
    logargs["bert_embeddings"] = 1 if args.bert_embeddings_train else 0
    logargs["flair_embeddings"] = 1 if args.flair_train else 0
    logargs["elmo_embeddings"] = 1 if args.elmo_train else 0

    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                  for key, value in sorted(logargs.items())))
    )

    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself
    if not os.path.exists(args.logdir): os.mkdir(args.logdir)

    # Dump passed options to allow future prediction.
    with open("{}/options.json".format(args.logdir), mode="w") as options_file:
        json.dump(vars(args), options_file, sort_keys=True)

    # Postprocess args
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Load the data
    train = morpho_dataset.MorphoDataset(args.train_data, max_sentences=args.max_sentences, bert_embeddings_filename=args.bert_embeddings_train)
    if args.dev_data:
        dev = morpho_dataset.MorphoDataset(args.dev_data, train=train, shuffle_batches=False,  bert_embeddings_filename=args.bert_embeddings_dev)
    test = morpho_dataset.MorphoDataset(args.test_data, train=train, shuffle_batches=False,  bert_embeddings_filename=args.bert_embeddings_test)

    # Character-level embeddings
    args.including_charseqs = (args.cle_dim > 0)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args,
                      num_tags=len(train.factors[train.TAGS].words),

                      pretrained_bert_dim=train.bert_embeddings_dim(),

                      predict_only=args.predict)


    # Train
    keep_max, best_epoch, epoch = 0, 0, 0
    f1s = [-1]
    for epochs, learning_rate in args.epochs:
        while keep_max < model_config.stop_criteria_steps:
            epoch += 1
            network.train_epoch(train, learning_rate, args)
            dev_score = 0

            precision, recall, dev_score = network.evaluate("dev", dev, args)
            print("epoch {} devf1 {}".format(epoch, dev_score))
            keep_max += 1
            if max(f1s) < dev_score:
                keep_max = 0
                best_epoch = epoch
                network.saver.save(network.session, "{}/model".format(args.logdir), write_meta_graph=False)
            f1s.append(dev_score)
            stat_in_file(model_config.loginfo, ["   EndEpoch", epoch, "f1", dev_score, "precision", precision, "recall", recall,
                                        "memory", model_config.p.memory_info().rss / 1024 / 1024])

    # Save network
    network.saver.restore(network.session, "{}/model".format(args.logdir))

    precision, recall, test_score = network.evaluate("test", test, args)
    print("testf1 {}".format(test_score))

    shutil.rmtree(args.logdir)
    return network, args, train, precision, recall, test_score

