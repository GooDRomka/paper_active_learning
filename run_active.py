from active_utils import *
from configs import *
from active_learning import start_active_learning
import sys

try:
    processN = sys.argv[2]
    total_num_of_process = sys.argv[1]
except Exception:
    processN = "1"
    total_num_of_process = "1"

try:
    exp_type = sys.argv[3]
except Exception:
    exp_type = "1"
exp_type = int(exp_type)
process = int(processN)

model_config = ModelConfig()
train_file = '/mnt/nfs-storage/data/english/train.txt'
test_file = '/mnt/nfs-storage/data/english/test.txt'
dev_file = '/mnt/nfs-storage/data/english/valid.txt'
train_vectors = "/mnt/nfs-storage/data/english/embeding/train_vectors_lists.txt"
test_vectors = "/mnt/nfs-storage/data/english/embeding/test_vectors_lists.txt"
dev_vectors = "/mnt/nfs-storage/data/english/embeding/dev_vectors_lists.txt"
vocab = '/mnt/nfs-storage/data/english/vocab.txt'

train = load_data(train_file, train_vectors)
dev = load_data(dev_file, dev_vectors)
test = load_data(test_file, test_vectors)

os.makedirs("/mnt/nfs-storage/logs/active"+str(exp_type)+"/", exist_ok=True)
number = find_new_number("/mnt/nfs-storage/logs/active"+str(exp_type)+"/")
model_config.loginfo = "/mnt/nfs-storage/logs/active"+str(exp_type)+"/" + number + "_loginfo.csv"

model_config.number = number
model_config.save_model_path = "saved_models/active_model.pth"
model_config.process = int(total_num_of_process)



seed = 0
if exp_type== 1:
    # lazy active(LC) learning
    params = [[STRATEGY.LC, STRATEGY.LAZY, 800, 8000, 0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 1000, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 2000, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 3000, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 2400, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5],
          [STRATEGY.LC, STRATEGY.LAZY, 1600, 8000,  0.5],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)
elif exp_type == 2:
    # self learning
    params = [[STRATEGY.LC, STRATEGY.LAZY, 800, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 1000, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 2000, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 3000, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 2400, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000, 0],
          [STRATEGY.LC, STRATEGY.LAZY, 1600, 8000, 0],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])
                    start_active_learning(train, dev, test, model_config)
elif exp_type == 3:
    # active leaning
    params = [[STRATEGY.LC, STRATEGY.NORMAL, 800, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 1000, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 2000, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 3000, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 2400, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 4000, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 1200, 8000, 0],
          [STRATEGY.LC, STRATEGY.NORMAL, 1600, 8000, 0],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])
                    start_active_learning(train, dev, test, model_config)
elif exp_type == 4:
    # lazy active(LC) learning
    params = [[STRATEGY.LC, STRATEGY.LAZY, 800, 8000, 0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 1000, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 2000, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 3000, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 2400, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.25],
          [STRATEGY.LC, STRATEGY.LAZY, 1600, 8000,  0.25],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)

elif exp_type == 5:
    # lazy active(LC) learning
    params = [[STRATEGY.LC, STRATEGY.LAZY, 800, 8000, 0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 1000, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 2000, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 3000, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 2400, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.75],
          [STRATEGY.LC, STRATEGY.LAZY, 1600, 8000,  0.75],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)
elif exp_type == 6:
    # lazy active learning rand 0.75
    params = [[STRATEGY.RAND, STRATEGY.LAZY, 800, 8000, 0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 1000, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 2000, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 3000, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 2400, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 4000, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 1200, 8000,  0.75],
          [STRATEGY.RAND, STRATEGY.LAZY, 1600, 8000,  0.75],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)

elif exp_type == 7:
    # self learning rand
    params = [[STRATEGY.RAND, STRATEGY.LAZY, 800, 8000, 0],
          [STRATEGY.RAND, STRATEGY.LAZY, 1000, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 2000, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 3000, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 2400, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 4000, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 1200, 8000,  0],
          [STRATEGY.RAND, STRATEGY.LAZY, 1600, 8000,  0],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)

elif exp_type == 8:
    # changing step_budget lazy active(LC) learning
    params = [
          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5, 250],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5, 250],

          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5, 750],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5, 750],

          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5, 1000],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5, 1000],


          [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5, 2000],
          [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5, 2000],

          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold, model_config.step_budget = param
                    model_config.seed = seed

                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)

elif exp_type == 9:
    # self learning rand
    params = [[STRATEGY.SELF, STRATEGY.LAZY, 800, 8000, 0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 1000, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 2000, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 3000, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 2400, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 4000, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 1200, 8000,  0, 0.9975],
          [STRATEGY.SELF, STRATEGY.LAZY, 1600, 8000,  0, 0.9975],
          ]
    for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
                    model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
                    model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold, model_config.self_threshold = param
                    model_config.seed = seed
                    if model_config.select_strategy==STRATEGY.SELF:
                        model_config.step_budget=20000000
                        model_config.budget=20000000
                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                    'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])

                    start_active_learning(train, dev, test, model_config)

# if exp_type == 10:
#     # new metric lazy active(LC) learning
#     params = [[STRATEGY.LC, STRATEGY.LAZY, 800, 8000, 0.5, STRATEGY.WORD ],
#           [STRATEGY.LC, STRATEGY.LAZY, 1000, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 2000, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 3000, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 2400, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 4000, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 1200, 8000,  0.5, STRATEGY.WORD],
#           [STRATEGY.LC, STRATEGY.LAZY, 1600, 8000,  0.5, STRATEGY.WORD],
#           ]
#     for i in range(5):
#         for param in params:
#             for j in range(2):
#                 seed += 1
#                 if seed<process*((10*len(params))//model_config.process) and seed>=(process-1)*((10*len(params))//model_config.process):
#                     model_config.save_model_path = "saved_models/active_model"+str(process)+".pth"
#                     model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold, model_config.oracle_metric = param
#                     model_config.seed = seed
#
#                     stat_in_file(model_config.loginfo, ["\n\n"])
#                     stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
#                                     'threshold', model_config.threshold,  "lr", model_config.learning_rate,"batch_size", model_config.batch_size, 'seed', model_config.seed ])
#
#                     start_active_learning(train, dev, test, model_config)

print("Ya vse")

