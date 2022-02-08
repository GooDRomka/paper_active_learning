
import os

#
# for exp in ["1","2","3"]:
#     direct_path = "res_from_cluster/logs/active"+exp+"/"
#     output = "./loginfo_exp_" + exp + ".txt"
#     os.remove(output)
#     for filename in os.listdir(direct_path):
#         input = direct_path+filename
#         print(input)
#         with open(output, "a") as fw, open(input,"r") as fr: fw.writelines(l for l in fr)

def bad_files(num,name):
    # if name[0]=="0":
    #     name = float(name[1])
    # else:
    #     name = float(name)
    # if num=="3" and name in [5,6,7,8,9]:
    #     return True
    # if num=="9" and name<=30:
    #     return True
    # if num=="10" and name<=15:
    #     return True
    # if num =='11' and name<=6:
    #     return True
    # if num =='12' and name<=4:
    #     return True
    return False

direct_path = "res_from_cluster_dialog/logsDialog/active"
output = "./logs/clusterDialog/log_exp_"
try:
    os.remove(direct_path)
except Exception:
    pass


for num in ['1']:
    for filename in os.listdir(direct_path+num+"/"):
        if not bad_files(num,filename[:2]):
            input = direct_path+num+"/"+filename
            print(" ", input)
            with open(output+num+".txt", "a") as fw, open(input,"r") as fr: fw.writelines(l for l in fr)
