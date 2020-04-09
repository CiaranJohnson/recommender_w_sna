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
    
    # def recommendations(self, user_id, k_users):
        
    #     # k_users = self.knn(user_id, k)
    #     rating_dict = {}
    #     artist_list = []
    #     for user in k_users:
    #         abc = self.user_artists.loc[self.user_artists['user_id'] == user]['artist_id'].tolist()
    #         artist_list.extend(abc)
    #     artist_list = list(dict.fromkeys(artist_list))
    #     # artist_list = artist_list[:40]

    #     bar = Bar('Processing', max=len(artist_list))
    #     for artist_id in artist_list:
    #         bar.next()
    #         rating_dict[artist_id] = self.sum_weights_knn(user_id, artist_id, k_users)
    #     bar.finish()

    #     rating_dict = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse = True)}
    #     # print(rating_dict)

    #     rec_list = dict(islice(rating_dict.items(), 20))

    #     hits = self.compare_list(user_id, rec_list)
    #     print(hits)
    #     return rec_list, hits

    

    def recommendations(self, user_id, k_users):
        rating_dict = {}

        bar = Bar('Processing', max=len(k_users))
        for user in k_users:
            bar.next()
            user_ratings = self.user_artists.loc[self.user_artists['user_id'] == user]
            artist_list = user_ratings['artist_id'].tolist()

            max_value = user_ratings['weight'].max()
            min_value = user_ratings['weight'].min()

            denom = max_value - min_value

            for artist in artist_list:
                artist_weight = user_ratings.loc[user_ratings['artist_id'] == artist]['weight'].item()

                # Normalise the users rating
                artist_rating = (artist_weight - min_value)/denom
                user_score = artist_rating * k_users[user]

                if artist in rating_dict:
                    rating_dict[artist] = rating_dict[artist] + user_score
                else:
                    rating_dict[artist] = user_score
        bar.finish()

        rating_dict = {k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1], reverse = True)}

        rec_list = dict(islice(rating_dict.items(), 20))
        hits = self.compare_list(user_id, rec_list)
        print(hits)

        return rec_list, hits


    def add_users(self, user_list):
        max_id = self.user_artists['user_id'].unique().max() + 1
        for user in user_list:
            print(user)

            user_pd = self.user_artists.loc[self.user_artists['user_id'] == user]

            tag_pd = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == user]
            
            max_weight = self.user_artists['weight'].max() * 2
            print(max_weight)
            a_id = 0

            for i in range(50):
                overwrite_user = user_pd.replace({'user_id':user}, max_id)
                overwrite_tag = tag_pd.replace({'user_id':user}, max_id)
                fake_df = pd.DataFrame({'user_id':[max_id], 'artist_id':[0], 'weight':[max_weight]})
                fake_friends = pd.DataFrame({'user_id':[max_id], 'friend_id':[max_id]})
                overwrite_user = overwrite_user.append(fake_df)
                max_id = max_id + 1
                # print(overwrite_user)
                # print(overwrite_tag)

                self.user_artists = self.user_artists.append(overwrite_user)
                self.user_taggedartists = self.user_taggedartists.append(overwrite_tag)
                self.user_friends = self.user_friends.append(fake_friends)
                # print(max_id)

        # print(self.user_artists.loc[self.user_artists['artist_id'] == 0])
        # print(self.user_taggedartists)
        



# rating_weights = {935: 6.378764636818178, 1929: 5.6488833785038155, 1103: 4.976958397288922, 1679: 4.548025744662862, 926: 3.410949757400132, 43: 2.215941792289245, 1723: 1.977124528807408, 1249: 1.9532033724272702, 264: 1.8890479337948862, 1191: 1.8528606087961175, 225: 1.8050647582932915, 1601: 1.4451149618915478, 1730: 1.3507889912179691, 544: 1.3314838531923932, 557: 1.3174211100507451, 117: 1.2755371527209487, 12: 1.2744469620255616, 747: 1.1612511962019518, 1604: 1.1428947197706218, 779: 1.1176786466133135}
# combined_weights = {1191: 0.30319281068858867, 374: 0.2974552643113256, 428: 0.27128141448580945, 1866: 0.2674023267735103, 
# 1210: 0.2672230770305728, 1585: 0.26036748320752856, 788: 0.25594983730101806, 1643: 0.2552996072686322, 
# 117: 0.2468510845825198, 1202: 0.24116710238958536, 1881: 0.24092803936746082, 1209: 0.23771297599278768, 
# 575: 0.23129140706378512, 290: 0.22528331171211907, 1900: 0.22398691352296723, 1327: 0.21440783412933517, 
# 96: 0.209510888517916, 1230: 0.205819473597518, 176: 0.20234974865183386, 264: 0.1974249285746011}
# cf = cf_me()
# cf.recommendations(40, rating_weights)
# print(cf.knn(2, 20))
# cf.weight_a_u(2, 7)
# cf.sum_weights(2, 50)