import pickle
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying/retailrocket')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()


def resolve_dataset_dir(dataset_name):
    candidates = [
        dataset_name,
        dataset_name.lower(),
        dataset_name.capitalize(),
        'RetailRocket',
        'retailrocket',
    ]
    for name in candidates:
        path = os.path.join('datasets', name)
        if os.path.isdir(path):
            return name
    raise FileNotFoundError('Dataset directory not found for {}'.format(dataset_name))


def infer_num_node(sequences):
    max_item_id = 0
    for seq in sequences:
        if not seq:
            continue
        seq_max = max(seq)
        if seq_max > max_item_id:
            max_item_id = seq_max
    return max_item_id + 1


dataset = resolve_dataset_dir(opt.dataset)
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == 'diginetica':
    num = 43098
elif dataset == 'Tmall':
    num = 40728
elif dataset == 'Nowplaying':
    num = 60417
else:
    num = infer_num_node(seq)

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for data in seq:
    for k in range(1, 4):
        for j in range(len(data) - k):
            src = data[j]
            dst = data[j + k]
            adj1[src][dst] = adj1[src].get(dst, 0) + 1
            adj1[dst][src] = adj1[dst].get(src, 0) + 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
