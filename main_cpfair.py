from preprocess import *
from glob import glob
import pandas as pd
from mip import Model, xsum, maximize
from tqdm import tqdm

no_item_groups = 2
no_user_groups = 2
topk = 50


def fairness_optimisation(fairness='N', uepsilon=0.000005, iepsilon = 0.0000005):
    print(f"Runing fairness optimisation on '{fairness}', {format(uepsilon, 'f')}, {format(iepsilon, 'f')}")
    # V1: No. of users
    # V2: No. of top items (topk)
    # V3: No. of user groups
    # V4: no. og item groups
    V1, V2, V3, V4 = set(range(int(dataset['user_size']))), set(range(topk)), set(range(no_user_groups)), set(range(no_item_groups))

    # initiate model
    model = Model()

    # W is a matrix (size: user * top items) to be learned by model
    #W = [[model.add_var(var_type=BINARY) for j in V2] for i in V1]
    W = [[model.add_var() for j in V2] for i in V1]
    user_dcg = [model.add_var() for i in V1]
    user_ndcg = [model.add_var() for i in V1]
    group_ndcg_v = [model.add_var() for k in V3]
    item_group = [model.add_var() for k in V4]

    user_precision=[model.add_var() for i in V1]
    group_precision=[model.add_var() for k in V3]

    user_recall=[model.add_var() for i in V1]
    group_recall= [model.add_var() for k in V3]

    if fairness == 'N':
        ### No Fairness ###
        model.objective = maximize(xsum((S[i][j] * W[i][j]) for i in V1 for j in V2))
    elif fairness == 'C':
        ### C-Fairness: NDCG_Best: group_ndcg_v[1] - group_ndcg_v[0] ###
        model.objective = maximize(xsum((S[i][j] * W[i][j]) for i in V1 for j in V2) - uepsilon * (group_ndcg_v[1] - group_ndcg_v[0]))
    elif fairness == 'P':
        model.objective = maximize(xsum((S[i][j] * W[i][j]) for i in V1 for j in V2) - iepsilon * (item_group[0] - item_group[1]))
    elif fairness == 'CP':
        model.objective = maximize(xsum((S[i][j] * W[i][j]) for i in V1 for j in V2) - uepsilon * (group_ndcg_v[1] - group_ndcg_v[0]) - iepsilon * (item_group[0] - item_group[1]))

    # first constraint: the number of 1 in W should be equal to top-k, recommending top-k best items
    k = 20
    for i in V1:
        model += xsum(W[i][j] for j in V2) == k

    for i in V1:
        # user_idcg_i = 7.137938133620551
        user_idcg_i = 7.040268381923512

        model += user_dcg[i] == xsum((W[i][j] * Ahelp[i][j]) for j in V2)
        model += user_ndcg[i] == user_dcg[i] / user_idcg_i

        model += user_precision[i]==xsum((W[i][j] * Ahelp[i][j]) for j in V2) / k
        model += user_recall[i]==xsum((W[i][j] * Ahelp[i][j]) for j in V2) / len(train_checkins[i])

    for k in V3:
        model += group_ndcg_v[k] == xsum(user_dcg[i] * U[i][k] for i in V1)
        model += group_precision[k] == xsum(user_precision[i] * U[i][k] for i in V1)
        model += group_recall[k] == xsum(user_recall[i] * U[i][k] for i in V1)

    for k in V4:
        model += item_group[k]== xsum(W[i][j] * Ihelp[i][j][k] for i in V1 for j in V2)

    for i in V1:
        for j in V2:
            model += W[i][j] <= 1
    # optimizing
    model.optimize()

    return W, item_group


def conv_mapping(mapping, x):
    for k, v in mapping.items():
        if v == x:
            return k


def to_mapping(mapping, x):
    for k, v in mapping.items():
        if k == x:
            return v


dataset_list = ['amazon_baby', 'facebook_books', 'ml-1m']
for dataset_name in dataset_list:
    dataset, index_F, index_M, genre_mask, popular_dict, vec_pop, long_tail, short_head, train_aplt, train_user_tail_list = preprocessing(dataset_name)
    train_checkins = {}
    for i, el in enumerate(dataset['train_user_list']):
        key, value = i, set(el)
        train_checkins[i] = value
    ground_truth = {}
    for i, el in enumerate(dataset['test_user_list']):
        key, value = i, set(el)
        ground_truth[i] = value
    shorthead_item_ids = set(short_head)
    longtail_item_ids = set(long_tail)
    recs = glob('arrays/BPRMF/*.npz')
    U = np.zeros([int(dataset['user_size']), no_user_groups])
    U[:, 1] = 1
    for rec in recs:
        if dataset_name in rec:
            # df = pd.read_csv(rec, sep='\t', names=['user', 'item', 'rate'])
            # df['user'] = df['user'].map(lambda x: to_mapping(dataset['user_mapping'], x))
            # df['item'] = df['item'].map(lambda x: to_mapping(dataset['item_mapping'], x))
            # S = np.zeros([int(dataset['user_size']), int(dataset['item_size'])])
            # X = df['user'].values
            # Y = df['item'].values
            # S[X, Y] = 1
            loaded = np.load(rec)
            S = loaded['arr_0']
            P = np.zeros([int(dataset['user_size']), topk])
            # i = 0
            #for j in range(0, P.shape[0]):
            #     P[j] = Y[i:i + topk]
            #    i += 50
            for uid in tqdm(range(int(dataset['user_size']))):
                P[uid] = np.array(list(reversed(S[uid].argsort()))[:topk])
            Ahelp = np.zeros([int(dataset['user_size']), topk])
            for uid in range(int(dataset['user_size'])):
                for j in range(topk):
                    # convert user_ids to user_idx
                    # convert item_ids to item_idx
                    if P[uid][j] in train_checkins[uid]:
                        Ahelp[uid][j] = 1

            Ihelp = np.zeros([int(dataset['user_size']), topk, no_item_groups])
            for uid in range(int(dataset['user_size'])):
                for lid in range(topk):
                    # convert item_ids to item_idx
                    if P[uid][lid] in shorthead_item_ids:
                        Ihelp[uid][lid][0] = 1
                    elif P[uid][lid] in longtail_item_ids:
                        Ihelp[uid][lid][1] = 1
            for fair_mode in ['P']:
                if fair_mode == 'N':
                    W, item_group = fairness_optimisation(fairness=fair_mode)
                elif fair_mode == 'C':
                    for user_eps in [0.003, 0.0005, 0.0001, 0.00005, 0.000005]:
                        W, item_group = fairness_optimisation(fairness=fair_mode, uepsilon=user_eps)
                elif fair_mode == 'P':
                    for item_eps in [0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 0.8, 1]:  # , 0.0005, 0.0001, 0.00005, 0.000005
                        W, item_group = fairness_optimisation(fairness=fair_mode, uepsilon=0, iepsilon=item_eps)
                        R = np.zeros([int(dataset['user_size']), topk])
                        for uid in range(int(dataset['user_size'])):
                            for j in range(topk):
                                R[uid][j] = W[uid][j].x
                        N_P = (P + 1) * R
                        N_X = np.zeros([int(dataset['user_size']) * 20, 2])
                        j = 0
                        for u in range(N_P.shape[0]):
                            for i in range(N_P.shape[1]):
                                if N_P[u][i] != 0:
                                    N_X[j][0] = int(u)
                                    N_X[j][1] = int(N_P[u][i])
                                    j += 1
                        N_X[:, 1] = N_X[:, 1] - 1
                        rec_elliot = pd.DataFrame(N_X, columns=['user', 'item'])
                        rec_elliot['user'] = rec_elliot['user'].map(lambda x: conv_mapping(dataset['user_mapping'], x))
                        rec_elliot['item'] = rec_elliot['item'].map(lambda x: conv_mapping(dataset['item_mapping'], x))
                        if 'BPRMF' in rec:
                            rec_elliot.to_csv(f'recs/BPRMF/BPRMF_PFAIR_{item_eps}_{dataset_name}.tsv',
                                              sep='\t', index=False, header=False)
                        elif 'LightGCN' in rec:
                            rec_elliot.to_csv(f'recs/LightGCN/LightGCN_PFAIR_{item_eps}_{dataset_name}.tsv',
                                              sep='\t', index=False, header=False)
                        # new_recs_string = rec.replace('None', f'PFAIR_{item_eps}').replace('npz', 'tsv')
                        # rec_elliot.to_csv(new_recs_string,
                        #                   sep='\t', index=False, header=False)
                elif fair_mode == 'CP':
                    for user_eps in [0.003, 0.0005, 0.0001, 0.00005, 0.000005]:
                        for item_eps in [0.003, 0.0005, 0.0001, 0.00005, 0.000005]:
                            W, item_group = fairness_optimisation(fairness=fair_mode, uepsilon=user_eps, iepsilon=item_eps)
