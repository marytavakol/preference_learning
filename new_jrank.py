import numpy as np
import json
import sys
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


#@profile
def identity_kernel(a, B):
    idvec = np.zeros((1, B.shape[0]))
    idvec[0, np.where(B == a)] = 1
    return idvec

#@profile
def attribute_kernel(a_features, b_matrix):
    return a_features.dot(b_matrix.transpose())


#@profile
def kernel(u, U, x, X, u_corr, x_corr):
    if u not in U:
        Ku = attribute_kernel(u_features[u], u_features[U])
    else:
        Cu = u_corr[u].toarray()
        Cu += np.multiply(Cu, Cu)
        Ku = np.array(identity_kernel(u, U) + attribute_kernel(u_features[u], u_features[U]) + Cu)[0]
    if x not in X:
        Kx = x_features[x, X]
    else:
        Cx = x_corr[x].toarray()
        Cx += np.multiply(Cx, Cx)
        Kx = np.array(identity_kernel(x, X) + x_features[x, X] + Cx)[0]
    return np.multiply(Ku, Kx)

#@profile
def predict(u, x, alpha, theta, U, X, u_corr, x_corr):
    k = kernel(u, U, x, X, u_corr, x_corr)
    value = np.multiply(alpha[:, 0], k).sum()
    if value < theta:
        return 0
    return 1


#@profile
def ComputeCorrelation(ids, matrix, n):
    uniq_ids = np.unique(ids)
    nu = len(uniq_ids)
    correlation = dict()
    for u in uniq_ids:
        correlation[u] = lil_matrix((1, n), dtype=np.float32)
    for i in range(nu):
        idx = np.where(ids == uniq_ids[i])[0]
        u1 = uniq_ids[i]
        correlation[u1][0, idx] = 1
        a_list = matrix[u1]
        a_list_data = list(a_list.values())
        if len(a_list_data) < 2:
            continue
        std_a = np.std(a_list_data)
        if std_a == 0:
            continue
        mu_a = np.mean(a_list_data)
        a_rates = list(a_list.keys())
        for j in range(i + 1, nu):
            u2 = uniq_ids[j]
            b_list = matrix[u2]
            b_list_data = list(b_list.values())
            if len(b_list_data) < 2:
                continue
            std_b = np.std(b_list_data)
            if std_b == 0:
                continue
            b_rates = list(b_list.keys())
            intersection = list(set(a_rates) & set(b_rates))
            if intersection == []:
                continue
            mu_b = np.mean(b_list_data)
            corr = [((a_list_data[a_rates.index(k)] - mu_a) / std_a) * ((b_list_data[b_rates.index(k)] - mu_b) / std_b) for k in intersection]
            corr = np.mean(corr)
            cols = np.where(ids==u2)[0]
            correlation[u1][0, cols] = corr

    return correlation


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

    with open(name, 'r') as fd:
        all_data = fd.readlines()

    aucs = []
    all_data = np.unique(np.array(all_data))

    kf = KFold(n_splits=5, shuffle=True)#, random_state=7)
    for train_index, test_index in kf.split(all_data):
        X_train, X_test = all_data[train_index], all_data[test_index]
        u_matrix = dict()
        x_matrix = dict()
        data = []
        for line in X_train:
            line = json.loads(line)
            u = uids.index(int(line['user'].split('-')[1]))
            xp = int(line['winner'])
            xn = int(line['loser'])
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
            u = uids.index(int(line['user'].split('-')[1]))
            xp = int(line['winner'])
            xn = int(line['loser'])
            test_data.append([u, xp, 1])
            test_data.append([u, xn, 0])

        del X_test
        test_data = np.unique(test_data, axis=0)
        res = evaluate(test_data, alpha, theta, u_corr, x_corr, U, X)
        del u_corr
        del x_corr
        del test_data
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
    uids = sorted(uids)

    u_features = []
    for u in uids:
        u_features.append(u_dict[str(u)])

    del u_dict
    u_features = np.array(u_features, dtype=np.float32)

    cross_validation(str(sys.argv[1]))




