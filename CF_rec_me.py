import pandas as pd
# from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
# from surprise import Dataset
# from surprise.reader import Reader
# from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from scipy.stats import pearsonr
from numpy import corrcoef

import math 
from itertools import islice

from progress.bar import Bar

# from pearson_cc import pearson_CC


class cf_me:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    # Find the KNN
    def knn(self, user_id, k):
        user_n = self.user_artists['user_id'].unique()
        p_scores = {}
        
        for user in user_n:
            if(not user_id == user):
                p_value, _ , _ = self.weight_a_u(user_id, user)
                p_scores[user] = p_value
            
        p_scores = {k: v for k, v in sorted(p_scores.items(), key=lambda item: item[1], reverse = True)}
        # p_scores = {k: v for k, v in sorted(p_scores.items(), reverse = True)}
        # print(sorted(p_scores.items(), reverse = True))
        top_k_users = dict(islice(p_scores.items(), k))
        return top_k_users

    def sum_weights(self, user_a, item_i):
        sum_wght = 0
        wght_rating = 0
        user_n = self.user_artists['user_id'].unique()
        for user_u in user_n:
            if (user_u != user_a):
                w_a_u, u_rating, mean_u = self.weight_a_u(user_a, user_u)
                sum_wght += w_a_u
                wght_rating += (u_rating[item_i] - mean_u) * w_a_u
        print(wght_rating / sum_wght)



    def compare_list(self, user_id, rec_list):
        user_list = self.user_artists.loc[self.user_artists['user_id'] == user_id]['artist_id'].tolist()
        # print("User List: " + str(user_list))
        # print("Rec List: " + str(rec_list))

        print("Your recommendations are: ")
        print("***************************************")
        for rec in rec_list:
            # print(self.artists.loc[self.artists['id'] == rec]['name'].item())
            print(rec)
            # print(str(rec) + ": " + str(rec_list[rec]))
        
        print("*************************************** \n")

        hits = set(user_list) & set(rec_list)
        return hits

    def weight_a_u(self, a_value, u_value):
        user_a = self.user_artists.loc[self.user_artists['user_id'] == a_value]
        user_b = self.user_artists.loc[self.user_artists['user_id'] == u_value]

        len_artist = len(self.artists.index)
        a_rating = np.zeros(len_artist)
        b_rating = np.zeros(len_artist)

        for art_id in user_a['artist_id']:
            a_index = self.artists.index[self.artists['id'] == art_id].tolist()
            art_inp = user_a.loc[user_a['artist_id'] == art_id]
            a_rating[a_index] = art_inp['weight']

        for art_id in user_b['artist_id']:
            b_index = self.artists.index[self.artists['id'] == art_id].tolist()
            art_inp = user_b.loc[user_b['artist_id'] == art_id]
            b_rating[b_index] = art_inp['weight']

        p_value,tail_val = pearsonr(a_rating, b_rating)
        # print(p_value)
        return p_value, b_rating, user_b['weight'].mean()
    

    def sum_weights_knn(self, user_a, item_i, k_users):
        sum_wght = 0
        wght_rating = 0
        for user_u in k_users:
            # print(user_u)
            if (user_u != user_a):
                w_a_u, u_rating, mean_u = self.weight_a_u(user_a, user_u)
                sum_wght += w_a_u
                b_index = self.artists.index[self.artists['id'] == item_i].tolist()
                wght_rating += (u_rating[b_index] - mean_u) * w_a_u
                # wght_rating += (u_rating[item_i] - mean_u) * w_a_u
        return (wght_rating / sum_wght)
    
    def recommendations(self, user_id, k_users):
        
        # k_users = self.knn(user_id, k)
        rating_dict = {}
        artist_list = []
        for user in k_users:
            abc = self.user_artists.loc[self.user_artists['user_id'] == user]['artist_id'].tolist()
            artist_list.extend(abc)
        artist_list = list(dict.fromkeys(artist_list))
        # artist_list = artist_list[:40]

        bar = Bar('Processing', max=len(artist_list))
        for artist_id in artist_list:
            bar.next()
            rating_dict[artist_id] = self.sum_weights_knn(user_id, artist_id, k_users)
        bar.finish()

        rating_dict = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse = True)}
        # print(rating_dict)

        rec_list = dict(islice(rating_dict.items(), 20))

        hits = self.compare_list(user_id, rec_list)
        print(hits)
        return rec_list, hits
        



rating_weights = {428: 0.45536144763288433, 1210: 0.44519833251253693, 1866: 0.4389093553272231, 374: 0.43704009526254284, 
1643: 0.4135033623870659, 1209: 0.40142996393956076, 1585: 0.39824791948978616, 1202: 0.33009835355386025, 
1881: 0.32270479214152964, 761: 0.3055197638960583, 1102: 0.2912863296718431, 430: 0.2912366109702745, 
612: 0.28810537850495277, 243: 0.2863466640791418, 788: 0.28229809464195776, 1074: 0.27793806442472835, 
1600: 0.2711769687643835, 1327: 0.2696643816652784, 1514: 0.2692347775489213, 176: 0.2664651539669095}
# combined_weights = {1191: 0.30319281068858867, 374: 0.2974552643113256, 428: 0.27128141448580945, 1866: 0.2674023267735103, 
# 1210: 0.2672230770305728, 1585: 0.26036748320752856, 788: 0.25594983730101806, 1643: 0.2552996072686322, 
# 117: 0.2468510845825198, 1202: 0.24116710238958536, 1881: 0.24092803936746082, 1209: 0.23771297599278768, 
# 575: 0.23129140706378512, 290: 0.22528331171211907, 1900: 0.22398691352296723, 1327: 0.21440783412933517, 
# 96: 0.209510888517916, 1230: 0.205819473597518, 176: 0.20234974865183386, 264: 0.1974249285746011}
# cf = cf_me()
# cf.recommendations(2, 20, rating_weights)
# print(cf.knn(2, 20))
# cf.weight_a_u(2, 7)
# cf.sum_weights(2, 50)