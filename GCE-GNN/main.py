import time
import argparse
import pickle
import os
import sys
import logging
from datetime import datetime
from model import *
from utils import *


def setup_logging(log_dir, dataset_name):
    """Setup logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{dataset_name}_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_file


def calculate_num_node(train_data):
    """Auto-calculate num_node from training data"""
    all_items = set()
    for sequence in train_data[0]:
        all_items.update(sequence)
    # Return max item id + 1 (assuming 0-based indexing)
    num_node = max(all_items) + 1 if all_items else 1
    return num_node


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall/retailrocket')
parser.add_argument('--data_path', default='datasets', help='Path to datasets directory')
parser.add_argument('--log_dir', default='logs', help='Directory to save log files')
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
parser.add_argument('--auto_num_node', action='store_true', help='Auto-calculate num_node from data')

opt = parser.parse_args()


def main():
    # Setup logging
    log_file = setup_logging(opt.log_dir, opt.dataset)
    logger = logging.getLogger()
    logger.info(f"Log file: {log_file}")
    
    init_seed(2020)
    
    # Build dataset path
    dataset_path = os.path.join(opt.data_path, opt.dataset)
    
    # Load training data first to calculate num_node if needed
    train_data_file = os.path.join(dataset_path, 'train.txt')
    train_data = pickle.load(open(train_data_file, 'rb'))
    
    # Calculate or set num_node
    if opt.auto_num_node or opt.dataset == 'retailrocket':
        num_node = calculate_num_node(train_data)
        logger.info(f"Auto-calculated num_node: {num_node}")
    else:
        if opt.dataset == 'diginetica':
            num_node = 43098
            opt.n_iter = 2
            opt.dropout_gcn = 0.2
            opt.dropout_local = 0.0
        elif opt.dataset == 'Nowplaying':
            num_node = 60417
            opt.n_iter = 1
            opt.dropout_gcn = 0.0
            opt.dropout_local = 0.0
        elif opt.dataset == 'Tmall':
            num_node = 40728
            opt.n_iter = 1
            opt.dropout_gcn = 0.6
            opt.dropout_local = 0.5
        else:
            num_node = calculate_num_node(train_data)
            logger.info(f"Unknown dataset, auto-calculated num_node: {num_node}")
    
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data_file = os.path.join(dataset_path, 'test.txt')
        test_data = pickle.load(open(test_data_file, 'rb'))

    adj_file = os.path.join(dataset_path, f'adj_{opt.n_sample_all}.pkl')
    num_file = os.path.join(dataset_path, f'num_{opt.n_sample_all}.pkl')
    
    adj = pickle.load(open(adj_file, 'rb'))
    num = pickle.load(open(num_file, 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    logger.info(str(opt))
    logger.info(f"num_node: {num_node}")
    logger.info("Training started...")
    
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        logger.info('-------------------------------------------------------')
        logger.info(f'epoch: {epoch}')
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        logger.info('Current Result:')
        logger.info(f'\tRecall@20:\t{hit:.4f}\tMRR@20:\t{mrr:.4f}')
        logger.info('Best Result:')
        logger.info(f'\tRecall@20:\t{best_result[0]:.4f}\tMRR@20:\t{best_result[1]:.4f}\tEpoch:\t{best_epoch[0]},\t{best_epoch[1]}')
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    logger.info('-------------------------------------------------------')
    end = time.time()
    logger.info(f"Run time: {end - start:.2f} s")
    logger.info("Training finished!")


if __name__ == '__main__':
    main()
