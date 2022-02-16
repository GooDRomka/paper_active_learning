import pandas as pd
import codecs
import os
import shutil
import csv
import json
import random
import pylab
from configs import ModelConfig, ActiveConfig
import matplotlib.pyplot as plt
from show_simple import read_file_simple

def read_from_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        stats = []
        for line in reader:
            stats.append(line)
        return stats


def read_file_active(path, scale, num=1):
    experiments = []
    loginfo = read_from_csv(path)
    for line in loginfo:
        if len(line) > 1:
            if line[0] == "BEGIN":
                stat = {"strategy": line[2], "label_strategy": line[4],"budget":line[6],"init_budget":line[8],"step_budget":line[10],
                        "threshold": line[12], "seed": line[14], "active_iteration": [], "epoch_iter": []}
                budget = float(line[6])
                init_budget = float(line[8])
                step_budget= float(line[10])
                if line[2] == "STRATEGY.SELF":
                    step_budget = 500
                spent_budget = init_budget
                fullcost = init_budget

            if line[0] == "TrainInitFinished":
                stat['active_iteration'].append({"bestf1dev": float(line[12]), "bestprecisiondev": float(line[8]), "bestrecalldev": float(line[10]),"budget": budget, "init_budget": init_budget, "step_budget":step_budget ,"spent_budget":spent_budget})

            if line[0] == "Selection":
                if num>=9:
                    S = 200000
                else:
                    S = 1
                added_price = (float(line[5]) - fullcost)/S
                fullcost = float(line[5])
                spentprice = float(line[15])
                spent_budget = spent_budget+step_budget*scale


            if line[0] == "IterFinished":
                stat['active_iteration'].append({"bestf1dev": float(line[21]), "bestprecisiondev": float(line[17]), "bestrecalldev": float(line[19]),"added_price": added_price,"budget": budget, "init_budget": init_budget, "step_budget":step_budget ,"spent_budget":spent_budget})

            if line[0] == "result":
                stat['active_iteration'] = stat['active_iteration'][:-2]
                stat.update({"f1test": float(line[12]), "precisiontest": float(line[8]), "recalltest": float(line[10]), "cost_of_train": fullcost,
                             "devf1": float(line[18]), "devprecision": float(line[14]), "devrecall": float(line[16]), "spentprice": spentprice})
                experiments.append(stat)

    return pd.DataFrame(experiments)

def find_new_number(directory):
    result = 0
    for filename in os.listdir(directory):
        try:
            num = int(filename[:2])
            result = num if num > result else result
        except Exception:
            pass

    if result+1<10:
        result = "0"+str(result+1)
    else:
        result = str(result+1)
    return result

def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

if __name__ == '__main__':
    directory_report = "report/active_b/"
    shutil.rmtree(directory_report)
    Title = {"1":"active,LC",
             "2":"lazy,LC,0.75",
             "3":"lazy,RAND,0.75",
             "4":"lazy,MC,0.75",
             "5":"lazy,LC,0.5",
             "6":"lazy,LC,0.25",
             "7":"self,LC,0",
             "8":"self,RAND,0",
             "9":"self,MC,0",
             "10":"self paper 0,5",
             "11":"self paper 0,995",
             "12":"self paper 0,999",
             "13":"self paper 0,8",
             "14":"<>"}
    scale = 1
    i = 2000
    # added_price_i = False
    scales, nums, metrics, exname= [1,0.5,0.2,0.1],['1','2','3','4','5','6','7','8','9'],['bestprecisiondev', 'bestf1dev', 'bestrecalldev'], "allapproches"#all approaches
    # scales,nums, exname = [1],['1','2','5','6'],['bestprecisiondev', 'bestf1dev', 'bestrecalldev'], "lazythreshold" #lazyactive different threshold
    # scales,nums, exname = [1],['1','2','3','4,['bestprecisiondev', 'bestf1dev', 'bestrecalldev'] #lazyactive different strategies
    # scales,nums, exname = [1],['1','7','8','9'],['bestprecisiondev', 'bestf1dev', 'bestrecalldev'] "selfstrategies" #selfkearning different strategies
    # scales,nums, exname = [1],['10','11','12','13'],['bestprecisiondev', 'bestf1dev', 'bestrecalldev'] #selfpaper
    for metric in metrics:
        for added_price_i in [False, True]:
            for scale in scales:
                for i in [800,1200,2000,4000]:
                    plt.style.use('ggplot')
                    plt.figure(figsize=(22,16),frameon=False)
                    j=0
                    for num in nums:
                        model_config = ModelConfig()
                        path_active = "logs/clusterDialog/log_exp_" + num + ".txt"
                        if not os.path.exists(directory_report):
                            os.makedirs(directory_report)
                        new_plot_num = find_new_number(directory_report)

                        path_simple = "logs/simple/paper_simple_learning_dev.csv"
                        experiments = read_file_simple(path_simple)

                        experiments_simple = experiments.groupby('budget', as_index=False).agg({'f1': ['mean', 'std']})

                        colors = [[0, 0.4470, 0.7410],[0, 0, 1],[0.8500, 0.3250, 0.0980],[0, 0.5, 0],[1, 0, 0],[0.4940, 0.1840, 0.5560],[0, 0.75, 0.75],
                        [0.4660, 0.6740, 0.1880],[0.75, 0, 0.75],[0.3010, 0.7450, 0.9330],[0.75, 0.75, 0],[0.6350, 0.0780, 0.1840],[0.25, 0.25, 0.25],[0, 0, 0.5],[0, 0.5, 0]]
                        if num=="1":
                            scale1 = 1
                        else:
                            scale1 = scale
                        experiments = read_file_active(path_active, scale1)
                        iterations_c = experiments["active_iteration"]
                        iterations = []
                        for lis in iterations_c:
                            for it in lis:
                                iterations.append(it)
                        iterations = pd.DataFrame(iterations)

                        iterations = iterations.groupby(['budget', 'init_budget','step_budget', 'spent_budget'],as_index=False).agg({'added_price': ['mean','std'],'bestf1dev': ['mean','std'], 'bestprecisiondev': ['mean','std'], 'bestrecalldev': ['mean','std']})

                        experiments = experiments.groupby(['budget','init_budget','step_budget']).agg(
                            {'devf1': ['mean', 'std'], 'devprecision': ['mean', 'std'], 'devrecall': ['mean', 'std']})
                        init_budget, budget, step_budget = pd.unique(iterations['init_budget']),pd.unique(iterations['budget']),pd.unique(iterations['step_budget'])

                        if j==0 and not added_price_i:
                            filt = max(budget)*scale1+max(init_budget)+1000
                            experiments_simple_filt = experiments_simple[experiments_simple['budget']<=filt]
                            plt.plot(experiments_simple_filt['budget'], experiments_simple_filt[('f1','mean')],label="simple", marker="o", color="black")
                            plt.fill_between(experiments_simple_filt['budget'],experiments_simple_filt[('f1','mean')]+experiments_simple_filt[('f1','std')],experiments_simple_filt[('f1','mean')]- experiments_simple_filt[('f1','std')],alpha=.2)


                        df = iterations[iterations['init_budget']==i]

                        if added_price_i:
                            plt.plot(df['spent_budget'], df[('added_price', 'mean')], label=Title[num], marker="o", color=colors[j])
                            plt.fill_between(df['spent_budget'],df[('added_price','mean')]+df[('added_price','std')],df[('added_price','mean')]- df[('added_price','std')],alpha=.2)
                        else:
                            plt.plot(df['spent_budget'],df[(metric,'mean')], label=Title[num], marker="o", color=colors[j])
                            plt.fill_between(df['spent_budget'],df[(metric,'mean')]+df[(metric,'std')],df[(metric,'mean')]- df[(metric,'std')],alpha=.2)
                        j+=1

                        # plt.errorbar(df['spent_budget'],df[('bestf1dev','mean')], df[('bestf1dev','std')], linestyle='None', marker='^')





                    plt.xlabel('spent_budget')
                    plt.ylabel(metric)
                    plt.legend(loc='best')
                    plt.title("learning with budget = "+str(i))
                    plt.savefig(directory_report+str(added_price_i)+"_"+metric+"_"+exname+"_"+str(i)+"_"+str(scale) +'.png')







