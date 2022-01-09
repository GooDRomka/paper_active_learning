
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


direct_path = "res_from_cluster/logs/active"
output = "./logs/cluster/log_exp_"
try:
    os.remove(direct_path)
except Exception:
    pass


for num in ['','2','3','4','5','6','7']:
    for filename in os.listdir(direct_path+num+"/"):
        if num == "":
            num1="1"
        else:
            num1 = num
        input = direct_path+num+"/"+filename
        print(input)
        with open(output+num1+".txt", "a") as fw, open(input,"r") as fr: fw.writelines(l for l in fr)
