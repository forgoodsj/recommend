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
    W = UserSimilarity(train)
    for user in train.keys():
        tu = test[user] #当前用户评分
        # rank = GetRecommendation(user, N)
        rank = Recommend(user,train,W)
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
    W = UserSimilarity(train)
    for user in train.keys():
        tu = test[user] #用户评分
        # rank = GetRecommendation(user, N)
        rank = Recommend(user, train, W)
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
    W = UserSimilarity(train)
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)  #训练集中所有的物品
        # rank = GetRecommendation(user,N)
        rank = Recommend(user, train, W)
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
    W = UserSimilarity(train)
    for user,items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item]= 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        # rank = GetRecommendation(user,N)
        rank = Recommend(user, train, W)
        rank = sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N]  # 对算出来的rank进行排序，取前N个
        for item ,pui in rank:
            ret += math.log(1+ item_popularity[item])
            n += 1
    ret /= n*1.0
    return ret

def UserSimilarity_1(train):
    '''
    计算用户的余弦相似度：电影评分交集/根号（两者数量之积）
    :param train:格式{user1:{movie1:rank1,movie2:rank2},user2:{movie33:rank1,movie4:rank2}}
    :return:
    但这个方法的时间复杂度是O（U*U），用户数大的时候非常耗时。很多用户之间没有对同样物品产生过行为，没必要计算。
    W = dict()
    for u in train.keys():
        for v in train.keys():
            if u==v:
                continue
            W[u][v] = len(train[u] & train[v])
            W[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)
    return W

    下述方法先计算|N(u)∩N(v)|≠0的用户对（u，v)，然后再处理
    '''
    # 先把用户评论了什么物品的用户---物品关系，改为 A物品被哪些用户访问过的 物品---用户关系
    item_users = dict()
    for u ,items in train.items():
        for i in items.keys(): # i是物品id
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u) #u是用户id
    # item_users = {'a': {'B', 'A'}, 'b': {'C', 'A'}, 'd': {'D', 'A'}, 'c': {'D', 'B'}, 'e': {'D', 'C'}}

    # 计算两个用户间有多少相同的物品
    C1 = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            if u not in N.keys():
                N[u] = 0
            N[u] +=1
            for v in users:
                if u==v:
                    continue
                if u not in C1.keys():
                    C1[u]={}
                if v not in C1[u].keys():
                    C1[u][v] = 0
                C1[u][v] += 1
    # 计算最后的相似度
    W = dict()
    for u, related_users in C1.items():
        for v, cuv in related_users.items():
            if u not in W.keys():
                W[u] = {}
            if v not in W[u].keys():
                W[u][v] = 0
            W[u][v] = cuv / math.sqrt((N[u] * N[v]))
    return W

def UserSimilarity(train):
    '''
    惩罚了用户uv共同兴趣列表中热门物品对相似度的影响
    :param train:
    :return:
    '''
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set() #共同兴趣列表
            item_users[i].add(u)

    # 计算相似度，有多少相同物品
    C = {}
    N = {}
    for i , users in item_users.items(): #物品i ,喜欢它的用户users
        for u in users:
            if u not in N.keys():
                N[u] = 0
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if u not in C.keys():
                    C[u] = {}
                if v not in C[u].keys():
                    C[u][v] = 0

                C[u][v] += 1/ math.log(1 + len(users)) #分子部分，len(users）这个物品喜欢的用户数，越多则权重越小
    # 最终的相似度
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            if u not in W.keys():
                W[u] = {}
            if v not in W[u].keys():
                W[u][v] = 0
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W



def Recommend(user,train,W):
    '''
    返回用户对剩余物品的感兴趣值。
    原理：取与用户A最相似的前n个用户，然后取这些用户对电影的评分。 当前用户对这个电影的评分=sum(用户A与被选用户相似度*被选用户的该电影的评分)
    :param user: 用户编号
    :param train: 格式{user1:{movie1:rank1,movie2:rank2},user2:{movie33:rank1,movie4:rank2}}
    :param W: 用户相似度 格式{user1:{user2:score2,user3:score3},user2:{user1:score1,user3:score3},user3:{user1:score1,user2:score2}}
    :return: {movie3:rank3,movie5:rank5}
    '''
    rank =dict()
    interacted_items = train[user] #用户已经评分过的电影

    for v, wuv in sorted(W[user].items(),key=operator.itemgetter(1),reverse=True)[0:80]: #取与当前用户相似度最高的前3个用户，这里的3可以调整
        for i, rvi in train[v].items(): #获取这些用户对电影的评分
            if i in interacted_items.keys(): #如果其他用户的 电影 当前用户已经评分过，则跳过
                continue
            if i not in rank.keys():
                rank[i] = 0
            rank[i] += wuv *rvi
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





trainset1={'A':{'a':1,'b':1,'d':1},'B':{'a':1,'c':1},'C':{'b':1,'e':1},'D':{'c':1,'d':1,'e':1}}
# trainset1['A']['a']=4
# print(trainset1['A']['a'])
# w = UserSimilarity(trainset1)
# print(w)
# rank = Recommend('A',trainset1,w)
# print(rank)
data = loaddata()
train,test = SplitData(data,8,3,2)

recall = Recall(train,test,10)
print("召回率是",recall)
prec = Precision(train,test,10)
print("准确率是",prec)
cov = Coverage(train,test,10)
print("覆盖率是",cov)
pop = Popularity(train,test,10)
print("流行度是",pop)
