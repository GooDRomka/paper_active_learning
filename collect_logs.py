
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
    if name[0]=="0":
        name = float(name[1])
    else:
        name = float(name)
    if num=="3" and name in [5,6,7,8,9]:
        return True
    if num=="9" and name<=30:
        return True
    if num=="10" and name<=15:
        return True
    if num =='11' and name<=6:
        return True
    if num =='12' and name<=4:
        return True
    return False

direct_path = "res_from_cluster/logs/active"
output = "./logs/cluster/log_exp_"
try:
    os.remove(direct_path)
except Exception:
    pass

# Title = {"1":"lazy active learning, LC, threshold 0.5","2":"self learning,LC, threshold 0","3":"active learning, LC",
#          "4":"lazy active learning, LC, threshold 0.25","5":"lazy active learning, LC, threshold 0.75","6":"lazy active learning,RAND, threshold 0.75",
#          "7":"self learning, RAND, threshold 0", "8":"lazy active learning, LC, threshold 0.5, dif step", "9":"self learning paper version 0,9975",
#          "10":"self learning paper version 0,99", "11":"self learning paper version 0,995",
#          "12":"self learning paper version 0,95","13":"self learning paper version 0,8",
#          "14":"self learning paper version 0,5"}
for num in ['','2','3','4','5','6','7','8','9','10','11','12','13','14']:
    for filename in os.listdir(direct_path+num+"/"):
        if num == "":
            num1="1"
        else:
            num1 = num
        if not bad_files(num1,filename[:2]):
            input = direct_path+num+"/"+filename
            print(" ", input)
            with open(output+num1+".txt", "a") as fw, open(input,"r") as fr: fw.writelines(l for l in fr)
