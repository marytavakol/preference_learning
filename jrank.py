
import numpy as np
import ujson as json
import sys
from scipy.sparse import coo_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score



def attribute_kernel(a_features, b_matrix):
    return a_features.dot(b_matrix.transpose())


#@profile
def kernel(u, U, x, X, u_corr, x_corr):
    if u not in U:
        Ku = attribute_kernel(u_features[u], u_features[U])
    else:
        Cu = u_corr[u].toarray()
        Ku = attribute_kernel(u_features[u], u_features[U]) + Cu

    if x not in X:
        Kx = x_features[x, X]
    else:
        Cx = x_corr[x].toarray()
        Kx = x_features[x, X] + Cx

    return np.multiply(Ku, Kx)

def predict(u, x, alpha, theta, U, X, u_corr, x_corr):
    k = kernel(u, U, x, X, u_corr, x_corr)
    value = np.multiply(alpha[:, 0], k).sum()
    if value < theta:
        return 0
    return 1


def compute_std(data):
    number_of_1 = sum(data)
    len_data = len(data)
    number_of_0 = len_data - number_of_1
    key = (number_of_0, number_of_1)
    if key not in stds:
        mean = number_of_1 / len_data
        std = np.std(data)
        stds[key] = (std, mean)
    return stds[key]


#@profile
def ComputeCorrelation(ids, matrix, n):
    print("--------Compute correlation")
    sys.stdout.flush()
    uniq_ids = np.unique(ids)
    nu = len(uniq_ids)
    cols_dict = {j: [] for j in range(nu)}
    uniq_ids_rev = {value: idx for idx, value in enumerate(uniq_ids)}
    for idx, id in enumerate(ids):
        j = uniq_ids_rev[id]
        cols_dict[j].append(idx)
    correlation = dict()
    for u in uniq_ids:
        correlation[u] = dict()

    for i in range(nu):
        idx = cols_dict[i]
        u1 = uniq_ids[i]
        for item in idx:
            correlation[u1][item] = 2  # plus the identity kernel
        a_list = matrix[u1]
        a_list_data = list(a_list.values())
        if len(a_list_data) < 2:
            continue
        std_a, mu_a = compute_std(a_list_data)
        if std_a == 0:
            continue
        a_rates = list(a_list.keys())
        a_rates_set = set(a_rates)
        for j in range(i + 1, nu):
            u2 = uniq_ids[j]
            b_list = matrix[u2]
            b_list_data = list(b_list.values())
            if len(b_list_data) < 2:
                continue
            std_b, mu_b = compute_std(b_list_data)
            if std_b == 0:
                continue
            b_rates = list(b_list.keys())
            intersection = list(a_rates_set & set(b_rates))
            if not intersection:
                continue
            corr = [((a_list_data[a_rates.index(k)] - mu_a) / std_a) * ((b_list_data[b_rates.index(k)] - mu_b) / std_b)
                    for k in intersection]
            corr = np.mean(corr)
            corr += (corr * corr)
            cols = cols_dict[j]
            for item in cols:
                correlation[u1][item] = corr
            for item in idx:
                correlation[u2][item] = corr

    correlation2 = dict()
    for u in uniq_ids:
        correlation2[u] = coo_matrix((list(correlation[u].values()), (np.zeros(len(correlation[u])), list(correlation[u].keys()))), shape=(1, n), dtype=np.float32)
    return correlation2


#@profile
def train(u_matrix, x_matrix, data):
    num_iterations = 10
    num = len(data)
    print("num of train samples: ", num)
    sys.stdout.flush()
    alpha = np.zeros((num, 1))
    theta = 0

    u_corr = ComputeCorrelation(data[:, 0], u_matrix, num)
    x_corr = ComputeCorrelation(data[:, 1],  x_matrix, num)

    print("start training...")
    sys.stdout.flush()

    for itr in range(num_iterations):
        s = 0
        for u, x, r in data:
            r_ = predict(u, x, alpha, theta, data[:, 0], data[:, 1], u_corr, x_corr)
            if r_ > r:
                alpha[s] = alpha[s] + (r - r_)
                theta += 1
            elif r_ < r:
                alpha[s] = alpha[s] + (r - r_)
                theta -= 1
            s += 1
        print("itr "+ str(itr) + " theta: " + str(theta))
        sys.stdout.flush()

    return alpha, theta, u_corr, x_corr, data[:, 0], data[:, 1]


#@profile
def evaluate(test_data, alpha, theta, u_corr, x_corr, U, X):
    num = len(test_data)
    print("num of test samples: ", num)
    sys.stdout.flush()

    pred_r, true_r = [], []
    for u, x, r in test_data:
        true_r.append(r)
        r_ = predict(u, x, alpha, theta, U, X, u_corr, x_corr)
        pred_r.append(r_)

    auc = roc_auc_score(true_r, pred_r)
    return auc


#@profile
def cross_validation(name):

    print("Cross Validation")
    sys.stdout.flush()

    with open(name, 'r') as fd:
        all_data = fd.readlines()
    all_data = np.unique(np.array(all_data))

    aucs = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(all_data):
        X_train, X_test = all_data[train_index], all_data[test_index]
        u_matrix = dict()
        x_matrix = dict()
        data = []
        for line in X_train:
            line = json.loads(line)
            line = map(int, [line['user'].split('-')[1], line['loser'], line['winner'], line['context']])
            user, loser, winner, context = line
            u = uids[user]
            xp = winner
            xn = loser
            if u not in u_matrix:
                u_matrix[u] = dict()
            u_matrix[u][xp] = 1
            u_matrix[u][xn] = 0
            if xp not in x_matrix:
                x_matrix[xp] = dict()
            if xn not in x_matrix:
                x_matrix[xn] = dict()
            x_matrix[xp][u] = 1
            x_matrix[xn][u] = 0

            # data.append([u, xp, 1])
            # data.append([u, xn, 0])

            if (u, xp, 1) not in data:
                if (u, xp, 0) in data:
                    data.remove((u, xp, 0))
                data.append((u, xp, 1))

            if (u, xn, 0) not in data:
                if (u, xn, 1) in data:
                    data.remove((u, xn, 1))
                data.append((u, xn, 0))

        data = np.array(data)

        del X_train
        alpha, theta, u_corr, x_corr, U, X = train(u_matrix, x_matrix, data)
        del u_matrix
        del x_matrix

        test_data = []
        for line in X_test:
            line = json.loads(line)
            line = map(int, [line['user'].split('-')[1], line['loser'], line['winner'], line['context']])
            user, loser, winner, context = line
            u = uids[user]
            xp = winner
            xn = loser

            # test_data.append([u, xp, 1])
            # test_data.append([u, xn, 0])

            if (u, xp, 1) not in test_data:
                if (u, xp, 0) in test_data:
                    test_data.remove((u, xp, 0))
                test_data.append((u, xp, 1))

            if (u, xn, 0) not in test_data:
                if (u, xn, 1) in test_data:
                    test_data.remove((u, xn, 1))
                test_data.append((u, xn, 0))

        del X_test
        #test_data = np.unique(test_data, axis=0)
        test_data = np.array(test_data)
        res = evaluate(test_data, alpha, theta, u_corr, x_corr, U, X)
        del u_corr
        del x_corr
        del U
        del X
        del alpha
        print(res)
        aucs.append(res)

    print(aucs)
    print("AUC: ", np.mean(aucs))
    print("std/sqrt(n)", np.std(aucs) / np.sqrt(len(aucs)))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('enter the input (data+x_attr+userFeatures file names.')
        sys.exit(0)

    x_features = np.load(str(sys.argv[2]))
    nX = x_features.shape[0]
    print('num of items: ', nX)

    with open(str(sys.argv[3]), 'r') as fd:
        u_dict = json.load(fd)

    nU = len(u_dict)
    print('num of users: ', nU)
    sys.stdout.flush()

    uids = list(map(int, u_dict.keys()))
    sorted_uids = sorted(uids)
    uids = {value: idx for idx, value in enumerate(uids)}

    u_features = []
    for u in sorted_uids:
        u_features.append(u_dict[str(u)])

    del u_dict
    u_features = np.array(u_features, dtype=np.float32)

    stds = {}
    cross_validation(str(sys.argv[1]))

