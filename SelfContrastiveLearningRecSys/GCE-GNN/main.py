import time
import argparse
import pickle
import logging
import os
import sys
from pathlib import Path
from model import *
from utils import *

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from ncs_logging import format_best_k_summary, write_run_summary
DATA_ROOT = REPO_ROOT / 'Data' / 'GCE-GNN'


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall/retailrocket')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--max_seq_len', type=int, default=None, help='truncate/pad session length for memory safety')
# 
parser.add_argument('--saved_models_path', type=str, default='output/test')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--item_cl_loss_weight', type=float, default=0.1)
parser.add_argument('--sampled_item_size', type=int, default=5000)
parser.add_argument('--use_item_cl_loss', action='store_true', default=False)
parser.add_argument('--max_train_samples', type=int, default=None, help='limit training samples for quick smoke tests')

opt = parser.parse_args()


def resolve_dataset_dir(dataset_name):
    path = DATA_ROOT / dataset_name.lower()
    if path.is_dir():
        return path
    raise FileNotFoundError('Dataset directory not found: {}'.format(path))


def infer_num_node(all_train_seq):
    max_item_id = 0
    for seq in all_train_seq:
        if not seq:
            continue
        seq_max = max(seq)
        if seq_max > max_item_id:
            max_item_id = seq_max
    return max_item_id + 1


def build_graph_from_sequences(seq, num_node, sample_num):
    adj1 = [dict() for _ in range(num_node)]
    adj = [[] for _ in range(num_node)]

    for data in seq:
        for k in range(1, 4):
            for j in range(len(data) - k):
                src = data[j]
                dst = data[j + k]
                adj1[src][dst] = adj1[src].get(dst, 0) + 1
                adj1[dst][src] = adj1[dst].get(src, 0) + 1

    weight = [[] for _ in range(num_node)]
    for t in range(num_node):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda y: y[1])]
        adj[t] = [v[0] for v in x][:sample_num]
        weight[t] = [v[1] for v in x][:sample_num]
    return adj, weight


def ensure_graph_files(dataset_dir, n_sample_all, num_node):
    adj_path = dataset_dir / 'adj_{}.pkl'.format(n_sample_all)
    num_path = dataset_dir / 'num_{}.pkl'.format(n_sample_all)
    if adj_path.exists() and num_path.exists():
        return adj_path, num_path
    all_train_seq = pickle.load(open(dataset_dir / 'all_train_seq.txt', 'rb'))
    adj, weight = build_graph_from_sequences(all_train_seq, num_node, n_sample_all)
    pickle.dump(adj, open(adj_path, 'wb'))
    pickle.dump(weight, open(num_path, 'wb'))
    return adj_path, num_path


def recall_and_mrr(predictions, ground_truth, k=20):
    recall, mrr = 0.0, 0.0
    if not ground_truth:
        return recall, mrr
    for gt, pred in zip(ground_truth, predictions):
        if gt in pred[:k]:
            recall += 1
            mrr += 1 / (pred[:k].index(gt) + 1)
    recall /= len(ground_truth)
    mrr /= len(ground_truth)
    return recall, mrr


def main():
    if not os.path.exists(opt.saved_models_path):
        os.makedirs(opt.saved_models_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[ %(asctime)s ] [%(levelname)s] %(message)s', datefmt='%H:%M:%S %a %b %d %Y')

    sHandler = logging.StreamHandler()
    sHandler.setLevel(logging.INFO)
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    fHandler = logging.FileHandler(os.path.join(opt.saved_models_path, 'output.log'), mode='w')
    fHandler.setLevel(logging.INFO)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    logger.info(opt)

    init_seed(2020)

    dataset_dir = resolve_dataset_dir(opt.dataset)

    if opt.dataset.lower() == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset.lower() == 'nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset.lower() == 'tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    else:
        all_train_seq = pickle.load(open(dataset_dir / 'all_train_seq.txt', 'rb'))
        num_node = infer_num_node(all_train_seq)
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0

    graph_adj_path, graph_num_path = ensure_graph_files(dataset_dir, opt.n_sample_all, num_node)

    train_data = pickle.load(open(dataset_dir / 'train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(dataset_dir / 'test.txt', 'rb'))

    if opt.max_train_samples is not None:
        n = opt.max_train_samples
        train_data = (train_data[0][:n], train_data[1][:n])

    adj = pickle.load(open(graph_adj_path, 'rb'))
    num = pickle.load(open(graph_num_path, 'rb'))
    seq_len_cap = opt.max_seq_len
    if seq_len_cap is None and opt.dataset.lower() == 'retailrocket':
        seq_len_cap = 50  # 90th percentile of retailrocket session lengths is ~55

    train_data = Data(train_data, train_len=seq_len_cap)
    test_data = Data(test_data, train_len=seq_len_cap)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))
    cl_loss_function = nn.CrossEntropyLoss()  # 

    start = time.time()
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info(f'Epoch {epoch + 1}/{opt.epoch}')

        result = train_test(
            model,
            train_data,
            test_data,
            cl_loss_function,
            use_item_cl_loss=opt.use_item_cl_loss,
            sampled_item_size=opt.sampled_item_size,
            temperature=opt.temperature,
            item_cl_loss_weight=opt.item_cl_loss_weight,
            saved_models_path=opt.saved_models_path,
        )

        flag = 0
        for K in top_K:
            hit = result['hit%d' % K]
            mrr = result['mrr%d' % K]
            if best_results['metric%d' % K][0] < hit:
                best_results['metric%d' % K][0] = hit
                best_results['epoch%d' % K][0] = epoch + 1
                flag = 1
            if best_results['metric%d' % K][1] < mrr:
                best_results['metric%d' % K][1] = mrr
                best_results['epoch%d' % K][1] = epoch + 1
                flag = 1

        logger.info(result)
        for K in top_K:
            logger.info(
                'Best Recall@%d: %.4f\tBest MRR@%d: %.4f\tEpoch: %d, %d'
                % (
                    K,
                    best_results['metric%d' % K][0],
                    K,
                    best_results['metric%d' % K][1],
                    best_results['epoch%d' % K][0],
                    best_results['epoch%d' % K][1],
                )
            )

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            logger.info('Early stopping triggered at epoch %d', epoch + 1)
            break
    logger.info('-------------------------------------------------------')
    end = time.time()
    logger.info('Run time: %f s' % (end - start))

    write_run_summary(
        'SelfContrastiveLearningRecSys',
        opt.dataset.lower(),
        format_best_k_summary(opt.dataset, best_results, top_K, header=f"model=GCE-GNN | {opt}"),
    )


if __name__ == '__main__':
    main()
