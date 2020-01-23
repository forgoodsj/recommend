#!/user/bin/env python
#coding=utf-8
import random
import math
import operator
import csv

def SplitData(data,M,k,seed):
    '''
    首先，将用户行为数据集按照均匀分布随机分成M份（本章取M=8），挑选一份作为测试集，将剩下的M-1份作为训练集。
    每次实验选取不同的k（0≤k≤M1）和相同的随机数种子seed，进行M次实验就可
以得到M个不同的训练集和测试集，然后分别进行实验，用M次实验的平均值作为最后的评测指
标。
    :param data:
    :param M:  随机分成M份
    :param k:  每次选取不同的k
    :param seed:  随机数种子
    :return:
    '''
    test = {}
    train = {}
    random.seed(seed)
    for user, item in data.items():
        if user not in test.keys():
            test[user]={}
        if user not in train.keys():
            train[user]={}
        for i in item:
            if random.randint(0,M) == k:
                test[user][i] = 1
            else:
                train[user][i] = 1
    return train, test


def Recall(train,test,N):
    '''
    召回率描述有多少比例的用户—物品评分记录包含在最终的推荐列表中，
    :param train:
    :param test:
    :param N:每个用户推荐多少个
    :return:
    '''
    hit = 0
    all = 0
    W = ItemSimilarity(train)
    for user in train.keys():
        tu = test[user] #当前用户评分
        # rank = GetRecommendation(user, N)
        rank = Recommendation(user, train, W,10)
        rank = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu) #全部用户评分
    return hit/(all*1.0)

def Precision(train,test,N):
    '''
    准确率描述最终的推荐列表中有多少比例是发生过的用户—物品评分记录
    :param train:
    :param test:
    :param N: 每个用户推荐多少个
    :return:
    '''
    hit = 0
    all = 0
    W = ItemSimilarity(train)
    for user in train.keys():
        tu = test[user] #用户评分
        # rank = GetRecommendation(user, N)
        rank = Recommendation(user, train, W,10)
        rank = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N] #对算出来的rank进行排序，取前N个
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N #全部用户推荐列表
    return hit/(all*1.0)

def Coverage(train,test,N):
    '''
    该覆盖率表示最终的推荐列表中包含多大比例的物品。如果所有的物品都被推荐给至少一个
用户，那么覆盖率就是100%。
    :param train:
    :param test:
    :param N:
    :return:
    '''
    recommend_items = set()
    all_items = set()
    W = ItemSimilarity(train)
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)  #训练集中所有的物品
        # rank = GetRecommendation(user,N)
        rank = Recommendation(user, train, W,10)
        rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N]  # 对算出来的rank进行排序，取前N个
        for item, pui in rank:
            recommend_items.add(item) #推荐集中所有的物品
    return len(recommend_items)/(len(all_items)*1.0)

def Popularity(train,test,N):
    '''
    这里用推荐列表中物品的平均流行度度量推荐结果的新颖度.
    这里，在计算平均流行度时对每个物品的流行度取对数，这是因为物品的流行度分布满足长
尾分布，在取对数后，流行度的平均值更加稳定。
    :param train:
    :param test:
    :param N:
    :return:
    '''
    item_popularity = dict()
    W = ItemSimilarity(train)
    for user,items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item]= 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        # rank = GetRecommendation(user,N)
        rank = Recommendation(user, train, W,10)
        rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N]  # 对算出来的rank进行排序，取前N个
        for item ,pui in rank:
            ret += math.log(1+ item_popularity[item])
            n += 1
    ret /= n*1.0
    return ret

def ItemSimilarity(train):
    '''
    计算物品相似度
    :param train:
    :return:
    '''
    C = {}
    N = {}
    for u ,items in train.items():
        for i in items:
            if i not in N.keys():
                N[i] = 0
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                if i not in C.keys():
                    C[i] = {}
                if j not in C[i].keys():
                    C[i][j] = 0
                C[i][j] += 1 / math.log(1 + len(items) * 1.0)

    #计算相似度矩阵
    W = dict()
    for i,related_items in C.items():
        for j, cij in related_items.items():
            if i not in W.keys():
                W[i] = {}
            if j not in W[i].keys():
                W[i][j] = 0
            W[i][j] = cij/ math.sqrt(N[i]*N[j])
    return W

def Recommendation(user_id,train,W,K):
    '''

    :param train:
    :param user_id:
    :param W:
    :param K:
    :return:
    '''
    rank ={}
    ru = train[user_id]
    for i,pi in ru.items():
        for j,wj in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:K]:
            if j in ru:
                continue
            if j not in rank.keys():
                rank[j] = 0
            rank[j] += pi *wj
            # rank[j].reason[i] = pi*wj
    return rank

def loaddata():
    data = {}
    filename = "ratings.csv"
    with open(filename) as f:
        reader = csv.reader(f)
        for i in list(reader)[1:]:
            if i[0] not in data.keys():
                data[i[0]] = {}
            if i[1] not in data[i[0]].keys():
                data[i[0]][i[1]] = 1
    return data


# trainset1={'1':{'1':1,'b':1,'d':1},'2':{'b':1,'c':1,'e':1},'3':{'c':1,'d':1},'4':{'b':1,'c':1,'d':1},'5':{'1':1,'d':1}}
#
# w = ItemSimilarity(trainset1)
# r = Recommendation(trainset1,'1',w,5)
# print(r)
data = loaddata()
train,test = SplitData(data,8,3,2)

recall = Recall(train,test,50)
print("召回率是",recall)
prec = Precision(train,test,50)
print("准确率是",prec)
cov = Coverage(train,test,50)
print("覆盖率是",cov)
pop = Popularity(train,test,50)
print("流行度是",pop)