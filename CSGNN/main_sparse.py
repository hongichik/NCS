import argparse
import pickle
import time
from pathlib import Path

from torch.backends import cudnn

from util import Data
from model import *
import os

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/nowplaying')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.02, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--gpu_id', type=int, default=0)

opt = parser.parse_args()
print(opt)
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True


setup_seed(2021)


def main():
    dataset_root = str(REPO_ROOT / 'Data' / 'CSGNN')
    # 选择不同过滤规则的数据集
    train_data = pickle.load(open(os.path.join(dataset_root, opt.dataset, 'filter20', 'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(dataset_root, opt.dataset, 'filter20', 'test.txt'), 'rb'))
    train_cate = pickle.load(open(os.path.join(dataset_root, opt.dataset, 'filter20', 'category_train.txt'), 'rb'))
    test_cate = pickle.load(open(os.path.join(dataset_root, opt.dataset, 'filter20', 'category_test.txt'), 'rb'))
    print('-----train length: %d ----' % len(train_data[0]))
    print('-----test length: %d ----' % len(test_data[0]))

    max_item_session = max(
        max((max(session) for session in train_data[0] if len(session) > 0), default=0),
        max((max(session) for session in test_data[0] if len(session) > 0), default=0)
    )
    max_item_target = max(int(np.max(train_data[1])), int(np.max(test_data[1])))
    n_node = max(max_item_session, max_item_target)

    max_cat_session = max(
        max((max(category) for category in train_cate[0] if len(category) > 0), default=0),
        max((max(category) for category in test_cate[0] if len(category) > 0), default=0)
    )
    max_cat_target = 0
    if len(train_cate) > 1 and len(test_cate) > 1:
        max_cat_target = max(int(np.max(train_cate[1])), int(np.max(test_cate[1])))
    c_node = max(max_cat_session, max_cat_target)
    train_data = Data(train_data, train_cate, shuffle=True, n_node=n_node, c_node=c_node)
    test_data = Data(test_data, test_cate, shuffle=False, n_node=n_node, c_node=c_node, build_adjacency=False)
    n_node, c_node = train_data.n_node, train_data.c_node
    # embedding_matrix = get_embedding(opt.dataset, n_node + c_node, opt.embSize)
    # 不使用预训练的结果
    embedding_matrix = None
    model = trans_to_cuda(
        DHCN(adjacency=train_data.adjacency, n_node=n_node, c_node=c_node, lr=opt.lr, l2=opt.l2, beta=opt.beta,
             layers=opt.layer,
             emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset, embedding=embedding_matrix))

    top_K = [1, 3, 5, 10, 15, 20, 25, 30]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
