
from enums import STRATEGY
from active_utils import *
from utils import *
from utils import *
from configs import *
from sklearn.model_selection import train_test_split
import os, psutil

from tagger import *
from network import *
def start_active_learning(train, dev, test, model_config):
    set_seed(model_config.seed)
    print("\n\n\n\n Strating new exp  \"active\" with params:", 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                'threshold', model_config.threshold, 'seed', model_config.seed)

    #### набираем init данные
    dataPool = init_data(DataPool(train['texts'], train['labels'], init_num=0), model_config)
    selected_texts, selected_labels = dataPool.get_selected()
    selected_ids = dataPool.get_selected_id()
    stat_in_file(model_config.loginfo, ["initDist", init_distribution(selected_labels), "initbudget", model_config.init_budget,
                    "initSumPrices", compute_price(selected_labels), "memory", model_config.p.memory_info().rss/1024/1024])

    print("init_distribution", init_distribution(selected_labels),"init_budget", compute_price(selected_labels))


    embedings, labels = get_embeding( selected_ids, selected_labels, train['embed'])
    X_train, X_dev, y_train, y_dev = train_test_split(list(range(len(labels))), list(range(len(labels))), test_size=0.2, random_state=42)

    train_texts, train_embed, train_labels = [selected_texts[i] for i in X_train], [embedings[i] for i in X_train], [selected_labels[i] for i in X_train]
    dev_texts, dev_embed, dev_labels = [selected_texts[i] for i in X_dev], [embedings[i] for i in X_dev],[selected_labels[i] for i in X_dev]
    dev_init_ids = [selected_ids[i] for i in X_dev]
    get_conll_file("train", model_config, train_texts, train_embed, train_labels)
    get_conll_file("dev", model_config, dev_texts, dev_embed, dev_labels)
    get_conll_file("test", model_config, dev['texts'], dev['embed'], dev['labels'])


    #### обучаем init модель
    network,  args, train_m, f1, precision, recall = train_model(model_config)
    print("init_model trained, budget", compute_price(selected_labels), "metrics ", f1, precision, recall)

    stat_in_file(model_config.loginfo,
                     ["TrainInitFinished", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget, "init_budget", compute_price(selected_labels),
                      "devprecision", precision, "devrecall", recall, "devf1", f1, "memory", model_config.p.memory_info().rss/1024/1024])

    ### активка цикл
    end_marker, iterations_of_learning, sum_prices, sum_perfect, sum_changed, sum_not_changed, sum_not_perfect, perfect, not_perfect, changed, not_changed, thrown_away, price = False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    while (selected_texts is None) or sum_prices < model_config.budget - 10 and not end_marker:
        iterations_of_learning += 1
        if model_config.label_strategy==STRATEGY.SELF:
            model_config.step_budget=20000000
            model_config.budget=20000000
        ### выбрать несколько примеров с помощью активки и разметить их
        dataPool, price, perfect, not_perfect, sum_prices = active_learing_sampling(network, dataPool, model_config, args, train_m, train, sum_prices, iterations_of_learning)
        selected_texts, selected_labels = dataPool.get_selected()
        selected_ids = dataPool.get_selected_id()
        if model_config.select_strategy == STRATEGY.SELF and price == 0:
            end_marker = True
            break

        embedings, labels = get_embeding(selected_ids, selected_labels, train['embed'])

        if model_config.label_strategy != STRATEGY.SELF:
            X_train, X_dev, y_train, y_dev = train_test_split(list(range(len(labels))), list(range(len(labels))),
                                                          test_size=0.2, random_state=42)
            train_texts,train_embed,train_labels = [selected_texts[i] for i in X_train], [embedings[i] for i in X_train], [selected_labels[i] for i in X_train]
            dev_texts,dev_embed,dev_labels = [selected_texts[i] for i in X_dev], [embedings[i] for i in X_dev],[selected_labels[i] for i in X_dev]
        else:
            X_train = []
            for i in range(len(labels)):
                if selected_labels[i] not in dev_init_ids:
                    X_train.append(i)
            train_texts,train_embed,train_labels = [selected_texts[i] for i in X_train], [embedings[i] for i in X_train], [selected_labels[i] for i in X_train]

        get_conll_file("train", model_config, train_texts, train_embed, train_labels)
        get_conll_file("dev", model_config, dev_texts, dev_embed, dev_labels)
        get_conll_file("test", model_config, dev['texts'], dev['embed'], dev['labels'])

        #### обучаем init модель

        network,  args, train_m, f1, precision, recall = train_model(model_config)
        #### сохранить результаты
        print("memory after training", model_config.p.memory_info().rss/1024/1024)
        print("iter ", iterations_of_learning, "finished, metrics dev", f1, precision, recall)
        stat_in_file(model_config.loginfo,
                 ["IterFinished", iterations_of_learning, "len(selected_texts):", len(selected_texts), "fullcost", compute_price(selected_labels),
                  "iter_spent_budget:", price, "not_porfect:", not_perfect, "thrown_away:", thrown_away, "perfect:", perfect, "total_spent_budget:", sum_prices,
                  "devprecision", precision, "devrecall", recall, "devf1", f1, "memory", model_config.p.memory_info().rss/1024/1024])


    get_conll_file("test", model_config, test['texts'], test['embed'], test['labels'])

    network, args, train_m, testf1, testprecision, testrecall = train_model(model_config)

    stat_in_file(model_config.loginfo,
                 ["result", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget, "Init_budget:", model_config.init_budget,
                  "testprecision", testprecision, "testrecall", testrecall, "testf1", testf1, "devprecision", precision, "devrecall", recall, "devf1", f1])

    print("result", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget, "Init_budget:", model_config.init_budget,
                  "testprecision", testprecision, "testrecall", testrecall, "testf1", testf1, "devprecision", precision, "devrecall", recall, "devf1", f1)
