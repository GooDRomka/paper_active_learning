
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


direct_path = "res_from_cluster/logs/active/"
output = "./logs/cluster/log_exp_1.txt"
try:
    os.remove(output)
except Exception:
    pass

with open(output, 'a') as f:
    pass
for filename in os.listdir(direct_path):
    input = direct_path+filename
    print(input)
    with open(output, "a") as fw, open(input,"r") as fr: fw.writelines(l for l in fr)
