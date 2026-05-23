import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        # Sessions have variable lengths, so store as object array for compatibility
        # with newer NumPy versions (ragged arrays are no longer inferred).
        self.raw = np.asarray(data[0], dtype=object)
        H_T = data_masks(self.raw, n_node)
        row_sum_ht = np.asarray(H_T.sum(axis=1)).reshape(1, -1)
        row_sum_ht[row_sum_ht == 0] = 1.0
        BH_T = H_T.T.multiply(1.0/row_sum_ht)
        BH_T = BH_T.T
        H = H_T.T
        row_sum_h = np.asarray(H.sum(axis=1)).reshape(1, -1)
        row_sum_h[row_sum_h == 0] = 1.0
        DH = H.T.multiply(1.0/row_sum_h)
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)

        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        n_sessions = len(sessions)
        rows, cols = [], []
        item_to_col = {}

        for r, session in enumerate(sessions):
            unique_items = np.unique(session)
            unique_items = unique_items[unique_items != 0]
            for item in unique_items:
                item = int(item)
                c = item_to_col.get(item)
                if c is None:
                    c = len(item_to_col)
                    item_to_col[item] = c
                rows.append(r)
                cols.append(c)

        if item_to_col:
            data = np.ones(len(rows), dtype=np.float32)
            incidence = coo_matrix((data, (rows, cols)), shape=(n_sessions, len(item_to_col))).tocsr()
            intersection = (incidence @ incidence.T).toarray().astype(np.float32)
            cnt = np.asarray(incidence.sum(axis=1)).reshape(-1, 1)
            union = cnt + cnt.T - intersection
            matrix = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection, dtype=np.float32),
                where=union != 0,
            )
        else:
            matrix = np.zeros((n_sessions, n_sessions), dtype=np.float32)

        np.fill_diagonal(matrix, 1.0)
        degree = np.sum(matrix, 1)
        degree[degree == 0] = 1.0
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask

