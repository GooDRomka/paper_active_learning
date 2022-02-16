import random
import logging
import copy
import time

from utils import *
from configs import *
import os, psutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import morpho_dataset
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
import numpy as np


class DataPool(object):
    def __init__(self, texts, labels, init_num):
        self.text_pool = np.array(texts)
        self.label_pool = np.array(labels)
        assert len(texts) == len(labels)
        self.pool_size = len(texts)
        # _l表示已标注数据集,_u表示未标注数据集
        self.selected_texts = None
        self.selected_labels = None
        self.unselected_texts = None
        self.unselected_labels = None
        self.selected_idx = sorted(set(random.sample(list(range(self.pool_size)), init_num)))
        self.unselected_idx = sorted(set(range(self.pool_size)) - set(self.selected_idx))
        self.update_pool()

    def update_pool(self):
        self.selected_texts = self.text_pool[self.selected_idx]
        self.selected_labels = self.label_pool[self.selected_idx]
        self.unselected_texts = self.text_pool[self.unselected_idx]
        self.unselected_labels = self.label_pool[self.unselected_idx]

    def update_idx(self, new_selected_idx):
        new_selected_idx = set(new_selected_idx)
        self.selected_idx = sorted(set(self.selected_idx) | new_selected_idx)
        self.unselected_idx = sorted(set(self.unselected_idx) - new_selected_idx)

    def translate_select_idx(self, source_idx):
        target_idx = [self.unselected_idx[idx] for idx in source_idx]
        return target_idx

    def update(self, unselected_idx):
        unselected_idx = self.translate_select_idx(unselected_idx)
        self.update_idx(unselected_idx)
        self.update_pool()

    def get_unselected_small(self, num):
        num = round(num)
        if num >= len(self.unselected_idx):
            return list(range(len(self.unselected_idx))), self.unselected_texts, self.unselected_labels
        idxs = list(range(len(self.unselected_idx)))
        small_unselected_idx = sorted(random.sample(idxs, num))
        small_unselected_labels = self.unselected_labels[small_unselected_idx]
        small_unselected_texts = self.unselected_texts[small_unselected_idx]
        return small_unselected_idx, small_unselected_texts, small_unselected_labels


    def get_selected(self):
        return self.selected_texts, self.selected_labels

    def get_selected_id(self):
        return self.selected_idx

    def get_unselected_id(self):
        return self.unselected_idx

    def get_unselected(self):
        return self.unselected_texts, self.unselected_labels

    def update_labels(self, to_be_selected_ids, tobe_small_selected_idxs, predicted_labels, model_config):
        perfect, changed = 0, 0
        price = 0
        ll = []
        TP,FP,FN = 0,0,0
        for id_to_be in to_be_selected_ids:
            trans_id = np.array(self.unselected_idx)[tobe_small_selected_idxs[id_to_be]]
            ll.append(trans_id)
            price+=len(self.label_pool[trans_id])

            true = bio_tags_to_spans(self.label_pool[trans_id])
            pred = bio_tags_to_spans(predicted_labels[id_to_be])
            for tag in true:
                if tag in pred:
                    TP+=1
                else:
                    FN+=1
            for tag in pred:
                if tag not in true:
                    FP+=1

            if self.label_pool[trans_id] == predicted_labels[id_to_be]:
                perfect+=1
            else:
                changed+=1
            self.label_pool[trans_id] = predicted_labels[id_to_be]

        return changed, perfect, (TP,FN,FP)

    def replace_label(self, new_text, new_label):
        for id in range(len(self.text_pool)):
            if self.text_pool[id] == new_text:
                self.label_pool[id] = new_label
                break


    def get_label_pool(self):
        return self.label_pool

    def get_text_pool(self):
        return self.text_pool

class ActiveStrategy(object):
    def __init__(self):
        pass

    def get_strategy(self, name):
        if name == 'lc':
            return self.least_confidence
        elif name == 'mnlp':
            return self.mnlp
        else:
            return self.random_sampling

    @classmethod
    def random_sampling(cls, texts, num):
        idxs = list(range(len(texts)))
        if num > len(texts):
            return idxs
        return random.sample(idxs, num)

    @classmethod
    def self_sampling(cls, model_config, viterbi_scores, texts):
        tobe_selected_idxs = []
        tobe_selected_scores = []
        idxs = list(range(len(texts)))
        # l,r = 0,0
        for log_score,id in zip(viterbi_scores,idxs):
            # r+=1
            if log_score>=model_config.self_threshold:
                tobe_selected_idxs.append(id)
                tobe_selected_scores.append(log_score)
                # l+=1
            # print(l,r,log_score)

        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def lc_sampling(cls, viterbi_scores, texts, select_num):
        """
        Least Confidence
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        seq_lens = np.array([len(text) for text in texts])
        scores = np.array(viterbi_scores)
        scores = scores/seq_lens
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def mc_sampling(cls, viterbi_scores, texts, select_num):
        """
        Least Confidence
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        seq_lens = np.array([len(text) for text in texts])
        scores = np.array(viterbi_scores)
        scores = scores/seq_lens
        tobe_selected_idxs = np.argsort(scores)[::-1][:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def mnlp_sampling(cls, mnlp_scores, texts, select_num):

        select_num = select_num if len(texts) >= select_num else len(texts)
        scores = np.array(mnlp_scores)
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        price = len(tobe_selected_idxs)
        return tobe_selected_idxs, tobe_selected_scores, price

    @classmethod
    def total_token_entropy(cls, prob):
        epsilon = 1e-9
        prob += epsilon
        tte = np.einsum('ij->', -np.log(prob) * prob)
        return tte

    @classmethod
    def tte_sampling(cls, probs, texts, select_num):
        """
        Total token entropy sampling.
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        tte_scores = np.array([cls.total_token_entropy(prob[:len(text), :])
                               for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(tte_scores)[-select_num:]
        tobe_selected_scores = tte_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def te_sampling(cls, probs, texts, select_num):
        select_num = select_num if len(texts) >= select_num else len(texts)
        te_scores = np.array([cls.total_token_entropy(prob[:len(text), :])/len(text)
                              for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(te_scores)[-select_num:]
        tobe_selected_scores = te_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    # 8) тут мы случайным образом выбираем часть датасета и после делаем ленивую разметку и отбрасываем по threshold данные
    # В одной из статей прочитал сравнение методов и оказалось, что рандом чуть ли не лучший алгоритм выбора данных,
    # там сравнивали всякие метрики уверенности и выбирали самые славбые примеры для разметки
    #  а в итоге рандом работал лучше
    @classmethod
    def random_sampling_precision(cls, scores, num, threshold):
        idxs = list(range(len(scores)))
        bad_examples = []
        if num > len(scores):
            samples = idxs
        else:
            samples = random.sample(idxs, num)
        res = []
        price = len(samples)
        perfect, not_perfect = 0,0
        for id in samples:
            if scores[id] > threshold:
                if scores[id] == 1:
                    perfect += 1
                else:
                    not_perfect += 1
                res.append(id)
            else:
                bad_examples.append(id)

        tobe_selected_idxs = res
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores, bad_examples, price, perfect, not_perfect

    @classmethod
    def sampling_precision(cls, tobe_selected_idxs, texts, scores, threshold, step):
        thrown_away = 0
        price = 0
        res = []
        scores_res = []
        perfect, not_perfect = 0, 0
        for id in tobe_selected_idxs:
            if len(texts[id]) + price < step:
                price += len(texts[id])
                if scores[id] >= threshold:
                    if scores[id] == 1:
                        perfect += len(texts[id])
                    else:
                        not_perfect += len(texts[id])
                    res.append(id)
                    scores_res.append(scores[id])
                else:
                    thrown_away += len(texts[id])
        return res, scores_res, thrown_away, perfect, not_perfect, price




def predict_precision_span(model, args, train_m, model_config,  texts, labels, embedings):
    dataset = create_temp_dataset(model_config, train_m,  texts, embedings, labels)
    tags, _ = model.get_tags(dataset, args)
    scores = []
    for label, text, tag in zip(labels, texts, tags):
        pr = 0
        if len(text) != len(label) or len(label) != len(tag):
            scores.append(pr)
        else:
            pr, re, f1 = model.f1_score_span([label], [tag])
            scores.append(pr)
    return np.array(scores),  np.array(tags)

def key_by_value(d,value):
    for key, val in d.items():
        if val == value:
            return key


def get_conll_file(file, model_config, sentences, embedings, labels):

    embed_path = "data/teprorary" + str(model_config.number)+"/" + file + "_vectors.txt"
    conll_path = "data/teprorary" + str(model_config.number)+"/" + file + ".txt"
    if not os.path.exists("data/teprorary"+str(model_config.number)+"/"):
        os.makedirs("data/teprorary"+str(model_config.number)+"/")

    if os.path.exists(conll_path):
        os.remove(conll_path)

    if os.path.exists(embed_path):
        os.remove(embed_path)

    if not os.path.exists(embed_path):
        with open(embed_path, 'w'):
            pass

    if not os.path.exists(conll_path):
        with open(conll_path, 'w'):
            pass

    file_object = open(conll_path, 'a')

    for sent,lable,embed in zip(sentences, labels, embedings):
        for s,l in zip(sent,lable):
            file_object.write("{}\t_\t_\t{}\n".format(s, l))
        file_object.write("\n")
    file_object.close()

    with open(embed_path, 'wb') as fp:
        pickle.dump(list(embedings), fp)

def create_temp_dataset(model_config, train_m,  texts, embedings, labels ):
    get_conll_file("temp", model_config, texts, embedings, labels)
    embed_path = "data/teprorary" + str(model_config.number)+"/" + "temp" + "_vectors.txt"
    conll_path = "data/teprorary" + str(model_config.number)+"/" + "temp" + ".txt"

    dataset = morpho_dataset.MorphoDataset(conll_path, train=train_m, shuffle_batches=False,  bert_embeddings_filename=embed_path)


    return dataset

def active_learing_sampling(model, dataPool, model_config, args,train_m, train, sum_prices, iterations_of_learning):
    unselected_ids = dataPool.get_unselected_id()
    small_unselected_ids, small_unselected_texts, small_unselected_labels = dataPool.get_unselected_small(model_config.step_budget)
    small_unselected_embedings, _ = get_embeding( np.array(unselected_ids)[small_unselected_ids], small_unselected_labels,
                                                            train['embed'])
    tobe_selected_idxs  = None

    if model_config.select_strategy == STRATEGY.LC:
        dataset = create_temp_dataset(model_config, train_m, small_unselected_texts, small_unselected_embedings, small_unselected_labels)
        tags, scores = model.get_tags(dataset, args)
        tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.lc_sampling(scores, small_unselected_embedings,
                                                                              model_config.step_budget)

    elif model_config.select_strategy == STRATEGY.MC:
        dataset = create_temp_dataset(model_config, train_m, small_unselected_texts, small_unselected_embedings, small_unselected_labels)
        tags, scores = model.get_tags(dataset, args)
        tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.mc_sampling(scores, small_unselected_embedings,
                                                                              model_config.step_budget)
    elif model_config.select_strategy == STRATEGY.RAND:
        tobe_selected_idxs = ActiveStrategy.random_sampling(small_unselected_embedings,
                                                            model_config.step_budget)

    elif model_config.select_strategy == STRATEGY.SELF:
        dataset = create_temp_dataset(model_config, train_m, small_unselected_texts, small_unselected_embedings, small_unselected_labels)
        tags, scores = model.get_tags(dataset, args)
        tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.self_sampling(model_config, scores, small_unselected_embedings)

    perfect, not_perfect, thrown_away = 0, 0, 0
    price = 0
    tpfnfp = (0,0,0)
    if model_config.label_strategy == STRATEGY.LAZY: #разметка проверяется оракулом, испольщуем PREDICT, а не GOLD
        scores, predicted_labels = predict_precision_span(model, args, train_m, model_config, small_unselected_texts, small_unselected_labels, small_unselected_embedings)
        tobe_selected_idxs, tobe_selected_scores, thrown_away, perfect, not_perfect, price = ActiveStrategy.sampling_precision(tobe_selected_idxs=tobe_selected_idxs, texts=small_unselected_texts, scores=scores, threshold=model_config.threshold, step=min(model_config.step_budget, model_config.budget - sum_prices))
        changed, not_changed, tpfnfp = dataPool.update_labels(tobe_selected_idxs, small_unselected_ids, predicted_labels, model_config)
        tobe_selected_idxs = np.array(small_unselected_ids)[tobe_selected_idxs]

    elif model_config.label_strategy == STRATEGY.NORMAL: #оракул размечает используем GOLD разметку
        tobe_selected_idxs_copy = tobe_selected_idxs.copy()
        tobe_selected_idxs = []
        for id in tobe_selected_idxs_copy:
            cost = len(small_unselected_embedings[id])
            if price + cost > min(model_config.step_budget, model_config.budget - sum_prices):
                end_marker = True
                # break
            else:
                tobe_selected_idxs.append(id)
                price += cost
        tobe_selected_idxs = np.array(small_unselected_ids)[tobe_selected_idxs]

    sum_prices += price
    dataPool.update_pool()
    dataPool.update(tobe_selected_idxs)
    selected_texts,selected_labels = dataPool.get_selected()
    stat_in_file(model_config.loginfo,
                 ["Selection", iterations_of_learning, "len(selected_texts):", len(selected_texts), "fullcost", compute_price(selected_labels),
                  "iter_spent_budget:", price, "not_porfect:", not_perfect, "thrown_away:", thrown_away, "perfect:", perfect, "total_spent_budget:", sum_prices,
                   "memory", model_config.p.memory_info().rss/1024/1024, "tpfnfp per iter", tpfnfp])
    return dataPool, price, perfect, not_perfect, sum_prices

def init_data(dataPool,model_config):
    budget_init = model_config.init_budget
    sum_price_init, price_init = 0, 0
    unselected_texts, unselected_labels = dataPool.get_unselected()

    while budget_init > 10 and len(unselected_texts) > 1:
        unselected_texts, unselected_labels = dataPool.get_unselected()
        tobe_selected_idxs = ActiveStrategy.random_sampling(unselected_texts, model_config.step_budget)
        tobe_selected_idxs, budget_init, price_init = choose_ids_by_price(tobe_selected_idxs, budget_init,
                                                                          unselected_texts)
        sum_price_init += price_init
        dataPool.update_pool()
        dataPool.update(tobe_selected_idxs)

    return dataPool

def stat_in_file(path, stats):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(stats)

def clear_old_model(path):
    try:
        os.remove(path)
    except Exception:
        pass

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x
