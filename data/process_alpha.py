import numpy as np
import csv

# Set random seed
seed = 0
np.random.seed(seed)

# Load data
print("Loading dataset")
# load adjacency matrix and the ground_truth
with open('alpha_network.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    net = [[d[0], d[1]] for d in data]

productID = {}
userID = {}
for key_word in net:
    if key_word[0] not in userID:
        userID[key_word[0]] = len(userID.keys())
    if key_word[1] not in productID:
        productID[key_word[1]] = len(productID.keys())

G = {}
for key_word in net:
    if (userID[key_word[0]], len(userID)+productID[key_word[1]]) not in G.keys():
        G[(userID[key_word[0]], len(userID)+productID[key_word[1]])] = 1
    else:
        G[(userID[key_word[0]], len(userID)+productID[key_word[1]])] += 1

f = open('alpha.txt','a')
for key in G.keys():
    f.write(str(key[0])+' '+str(key[1])+'\n')
f.close()

label = []
tmp_id = []
with open('alpha_gt.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    for d in data:
        if (d[0] in userID.keys()) and (d[0] not in tmp_id):
            tmp_id.append(d[0])
            if d[1] == '-1':
                label.append([userID[d[0]], 1])
            else:
                label.append([userID[d[0]], 0])
np.savetxt('alpha_label.txt', label, fmt="%d")