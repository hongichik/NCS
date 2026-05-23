import argparse
import datetime
import pickle
import time
import sys
import torch
from pathlib import Path
from util import Data, split_validation
from model import *
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from ncs_logging import format_best_k_summary, write_run_summary
from ncs_paths import log_file


class TeeLogger:
    """Ghi đồng thời ra stdout/stderr và file log."""
    def __init__(self, log_path, stream):
        self._stream = stream
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._file = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self._stream.write(message)
        self._file.write(message)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/Nowplaying/sample')
parser.add_argument('--data_path', default=None,
                    help='đường dẫn thư mục chứa dataset (ghi đè --dataset). '
                         'Thư mục phải chứa train.txt và test.txt')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--n_node', type=int, default=None,
                help='số lượng item/node. Nếu không truyền sẽ tự suy ra từ dữ liệu')
parser.add_argument('--log', default=None,
                    help='đường dẫn file log (vd: logs/run.log). '
                         'Nếu không truyền thì chỉ in ra terminal.')
parser.add_argument('--topk', default='5,10,20',
                help='danh sách K để đánh giá, dạng csv. Ví dụ: 20 hoặc 5,10,20')

opt = parser.parse_args()

if not opt.log:
    opt.log = str(log_file('DHCN', opt.dataset.lower()))

# Thiết lập logging
if opt.log:
    _tee_out = TeeLogger(opt.log, sys.stdout)
    _tee_err = TeeLogger(opt.log, sys.stderr)
    sys.stdout = _tee_out
    sys.stderr = _tee_err

print(opt)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# torch.cuda.set_device(1)


def parse_topk(topk_csv):
    values = []
    for x in str(topk_csv).split(','):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k <= 0:
            raise ValueError('topk must contain positive integers')
        values.append(k)
    if not values:
        raise ValueError('topk is empty')
    return sorted(set(values))

def main():
    # Xác định đường dẫn dataset
    if opt.data_path:
        base_path = opt.data_path
    else:
        base_path = str(REPO_ROOT / 'Data' / 'DHCN' / opt.dataset.lower())

    train_data = pickle.load(open(os.path.join(base_path, 'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(base_path, 'test.txt'), 'rb'))

    if opt.n_node is not None:
        n_node = opt.n_node
    elif opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'Tmall':
        n_node = 40727
    elif opt.dataset == 'Nowplaying':
        n_node = 60416
    else:
        # Suy ra n_node từ item id lớn nhất trong sessions + targets của train/test.
        train_sessions, train_targets = train_data
        test_sessions, test_targets = test_data
        max_session_id = max(
            max((max(session) if len(session) > 0 else 0) for session in train_sessions),
            max((max(session) if len(session) > 0 else 0) for session in test_sessions),
        )
        max_target_id = max(np.max(train_targets), np.max(test_targets))
        n_node = int(max(max_session_id, max_target_id))
        print('Auto inferred n_node =', n_node)
    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=True, n_node=n_node)
    model = trans_to_cuda(DHCN(adjacency=train_data.adjacency,n_node=n_node,lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print('Using device =', next(model.parameters()).device)

    top_K = parse_topk(opt.topk)
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data, top_k=top_K)
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

    write_run_summary(
        'DHCN',
        opt.dataset.lower(),
        format_best_k_summary(opt.dataset, best_results, top_K, header=str(opt)),
    )


if __name__ == '__main__':
    main()
