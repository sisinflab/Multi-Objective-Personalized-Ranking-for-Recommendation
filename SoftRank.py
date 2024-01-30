import torch
import torch.nn as nn
import math
from functools import partial



class SmoothRank(torch.nn.Module):

    def __init__(self, temp=1):
        super(SmoothRank, self).__init__()
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()

    @staticmethod
    def approximate_rank(temp, user_subscores, all_scores):
        sigmoid = torch.nn.Sigmoid()
        # x_0 = (user_subscores / temp).unsqueeze(dim=-1)
        # x_1 = (all_scores / temp).unsqueeze(dim=-2)
        # diff = sigmoid((all_scores / temp).type(torch.half).unsqueeze(dim=-2) - (user_subscores / temp).type(torch.half).unsqueeze(dim=-1))
        diff = sigmoid(
            (all_scores / temp).unsqueeze(dim=-2) - (user_subscores / temp).unsqueeze(
                dim=-1))

        # del x_0
        # del x_1
        # diff = scores.unsqueeze(dim=-2) - scores_max_relevant.unsqueeze(dim=-1)
        # diff = diff / temp
        # diff = sigmoid(diff)
        # del diff

        return torch.sum(diff, dim=-1) + 0.5
        # return rank

    def forward(self, scores_max_relevant, scores):
        # new_appr = partial(self.approximate_rank, self.temp)

        # return torch.stack(list(map(new_appr, scores_max_relevant, scores)))
        # b = torch.Tensor(list(map(new_appr, scores_max_relevant, scores)))
        # # torch.cat(list(map(new_appr, scores_max_relevant, scores)), out=b)
        # for user in range(scores_max_relevant.shape[0]):
        #     self.approximate_rank(scores_max_relevant[user], scores[user], self.temp)
        #
        # ___ x_0 = scores_max_relevant.unsqueeze(dim=-1).type(torch.half)
        # ___ x_1 = scores.unsqueeze(dim=-2).type(torch.half)
        # ___ diff = x_1 - x_0
        # # del x_0
        # # del x_1
        # # diff = scores.unsqueeze(dim=-2) - scores_max_relevant.unsqueeze(dim=-1)
        # ___ diff = diff / self.temp
        # ___ diff = self.sigmoid(diff)
        # # del diff
        #
        # ___ rank = torch.sum(diff, dim=-1) + 0.5
        # del diff
        # torch.cuda.empty_cache()
        # ___ return rank
        sigmoid = torch.nn.Sigmoid()
        # diff = sigmoid(
        #     (scores / self.temp).type(torch.half).unsqueeze(dim=-2) - (scores_max_relevant / self.temp).type(torch.half).unsqueeze(
        #         dim=-1))
        diff = sigmoid(
            (scores / self.temp).unsqueeze(dim=-2) - (scores_max_relevant / self.temp).unsqueeze(
                dim=-1))
        return torch.sum(diff, dim=-1) + 0.5

    def forward_c(self, scores, args):

        # a = scores[0]
        ranks = torch.empty((scores.size(0), scores.size(1)), dtype=torch.float32).to(args.device)
        sigmoid = torch.nn.Sigmoid()
        for j, a in enumerate(scores):
            print(f'User: {j}')
            b = torch.zeros(scores.size(1), scores.size(1)).to(args.device)
            for i, x in enumerate(a):
                b[i, i:] = a[i:] - a[i]
                b[i, i:] = sigmoid(b[i, i:] / self.temp)
        # b = sigmoid(torch.triu(b) / self.temp)
            b = b + b.T - torch.diag(torch.diag(b)) # completa matrice triangolare inferiore a cui va applicato 1-
            for i, x in enumerate(b):
                b[i, :i] = 1 - b[i, :i]
            ranks[j] = torch.sum(b, dim=-1) + 0.5
        # ones = torch.tril(torch.ones(scores.size(1), scores.size(1)), diagonal=-1)

        # b = ones - b
        return ranks

    def forward_cp(self, args, a):

        # a = scores[0]
        # ranks = torch.empty((a.size(0), a.size(1)), dtype=torch.float32).to(args.device)
        sigmoid = torch.nn.Sigmoid()
        b = torch.zeros(a.size(0), a.size(0)).to(args.device)
        for i, x in enumerate(a):
            b[i, i:] = a[i:] - a[i]
            b[i, i:] = sigmoid(b[i, i:] / self.temp)
    # b = sigmoid(torch.triu(b) / self.temp)
        b = b + b.T - torch.diag(torch.diag(b)) # completa matrice triangolare inferiore a cui va applicato 1-
        for i, x in enumerate(b):
            b[i, :i] = 1 - b[i, :i]
        b = torch.sum(b, dim=-1) + 0.5
        # ones = torch.tril(torch.ones(scores.size(1), scores.size(1)), diagonal=-1)

        # b = ones - b
        return b

    def forward_partial(self, scores, args):
        new_appr = partial(self.forward_cp, args)
        return torch.stack(list(map(new_appr, scores)))

    def forward_w(self, scores_max_relevant, scores):
        new_appr = partial(self.approximate_rank, self.temp)

        return torch.stack(list(map(new_appr, scores_max_relevant, scores)))
        # b = torch.Tensor(list(map(new_appr, scores_max_relevant, scores)))
        # # torch.cat(list(map(new_appr, scores_max_relevant, scores)), out=b)
        # for user in range(scores_max_relevant.shape[0]):
        #     self.approximate_rank(scores_max_relevant[user], scores[user], self.temp)
        #
        # x_0 = scores_max_relevant.unsqueeze(dim=-1)
        # x_1 = scores.unsqueeze(dim=-2)
        # diff = x_1 - x_0
        # # del x_0
        # # del x_1
        # # diff = scores.unsqueeze(dim=-2) - scores_max_relevant.unsqueeze(dim=-1)
        # diff = diff / self.temp
        # diff = self.sigmoid(diff)
        # # del diff
        #
        # rank = torch.sum(diff, dim=-1) + 0.5
        # del diff
        # torch.cuda.empty_cache()
        # return rank


class SmoothMRRLoss(nn.Module):

    def __init__(self, temp=1):
        super(SmoothMRRLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)

    def forward(self, scores, labels):
        ranks = self.smooth_ranker(scores)
        labels = torch.where(labels > 0, self.one, self.zero)
        rr = labels / ranks
        rr_max, _ = rr.max(dim=-1)
        mrr = rr_max.mean()
        loss = -mrr
        return loss


# class SmoothDCGLoss(nn.Module):
#
#     def __init__(self, temp=1):
#         super(SmoothDCGLoss, self).__init__()
#         self.smooth_ranker = SmoothRank(temp)
#         self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
#         self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
#         # self.topk = topk
#
#     def forward(self, scores_top, scores_all, labels):
#         ranks = self.smooth_ranker(scores_top, scores_all)
#         d = torch.log2(ranks + 1)
#         dg = labels / d
#         dcg = dg.sum(dim=-1)
#         # k = torch.sum(labels, dim=1).long()
#         # k = torch.clamp(k, max=self.topk, out=None)
#         # dcg = dcg / self.idcg_vector[k - 1]
#         dcg = dcg
#         # avg_dcg = dcg.mean()
#         # loss = -avg_dcg
#         return dcg


class SmoothDCGLoss(nn.Module):

    def __init__(self, args, topk, temp=1):
        super(SmoothDCGLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
        self.topk = topk
        self.device = args.device
        self.idcg_vector = self.idcg_k()


    def idcg_k(self):
        res = torch.zeros(self.topk).to(self.device)

        for k in range(1, self.topk+1):
            res[k-1] = sum([1.0 / math.log(i+2, 2) for i in range(k)])

        return res

    def forward(self, scores_top, scores, labels):
        ranks = self.smooth_ranker(scores_top, scores)
        # print("ranks:", ranks)
        d = torch.log2(ranks+1)
        dg = labels / d

        ndcg = None

        for p in range(1, self.topk+1):
            dg_k = dg[:,:p]
            dcg_k = dg_k.sum(dim=-1)
            k = torch.sum(labels, dim=-1).long()
            k = torch.clamp(k, max=p, out=None)
            ndcg_k = (dcg_k / self.idcg_vector[k-1]).reshape(-1, 1)

            ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)

        # print("ndcg:", ndcg.shape)

        # dcg = dg.sum(dim=-1)
        # k = torch.sum(labels, dim=-1).long()
        # k = torch.clamp(k, max = self.topk, out=None)
        # dcg = dcg / self.idcg_vector[k-1]
        # dcg = dcg

        return ndcg


def print_2d_tensor(name, value, prec=3):
    print('[{}]'.format(name))
    value = value.cpu().numpy()
    for i in range(len(value)):
        if prec == 0:
            value_i = [int(x) for x in value[i]]
        else:
            value_i = [round(x, prec) for x in value[i]]
        str_i = [str(x) for x in value_i]
        print('q{}: {}'.format(i, ' '.join(str_i)))
    print()

