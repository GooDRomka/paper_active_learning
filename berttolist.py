import pickle
import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
type = "dev"
# for type in ['dev','test','train']:
#     old = "./data/english/embeding/"+ type+ "_vectors.txt"
#     new = "./data/english/embeding/"+ type+ "_vectors_new.txt"
#     with open(old, 'rb') as fp:
#         bert_embeddings = list(pickle.load(fp).values())
#     res = []
#     for sent in bert_embeddings:
#         res.append([])
#         for word in sent:
#             res[-1].append(word.numpy())
#         res[-1] = np.array(res[-1], dtype=np.float32)
#
#     with open(new, 'wb') as fp:
#         pickle.dump(res, fp)
#
#     print("done")
#
# for type in ['valid','test','train']:
#     # with open("data/english/"+ type+ ".txt") as f:
#     #     print(f.read().replace(' ','\t'))
#     with open("data/english/" + type+ ".txt") as f:
#         text = ''.join([line.replace(' ','\t') for line in f.readlines()])
#         f.seek(0)
#         f.write(text)
#     print("done")

for type in ['valid','test','train']:
    file = open("data/english/" + type+ ".txt", "r")
    replacement = ""
    # using the for loop
    for line in file:
        line = line.strip()
        changes = line.replace(' ','\t')
        replacement = replacement + changes + "\n"

    file.close()
    # opening the file in write mode
    fout = open("data/english/" + type+ ".txt", "w")
    fout.write(replacement)
    fout.close()
    print("done")
