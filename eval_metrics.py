import pandas as pd
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise.reader import Reader
from surprise.model_selection import cross_validate
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from numpy import corrcoef

import math 
from itertools import islice

from scipy.sparse import lil_matrix

from progress.bar import Bar

import CF_rec_me as my_cf

# from pearson_cc import pearson_CC


class weight_calc:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)


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

    def weight_tags(self, a_value, u_value):
        tags_a = self.user_taggedartists.loc[self.user_taggedartists['user_id'] ==  a_value]
        tags_u = self.user_taggedartists.loc[self.user_taggedartists['user_id'] ==  u_value]
        
        joined_pd = pd.concat([tags_a, tags_u], axis=0)
        artist_list = joined_pd['artist_id'].unique()
        for artist in artist_list:
            a_tag_artist = tags_a.loc[tags_a['artist_id'] ==  artist]['tag_id'].tolist()
            u_tag_artist = tags_u.loc[tags_u['artist_id'] ==  artist]['tag_id'].tolist()
            
            common_tags = list(set(a_tag_artist).intersection(u_tag_artist))
            

            if len(common_tags) > 0:
                print(common_tags)
                print("User: " + str(u_value) + " num in Common: " + str(len(common_tags)))

        # p_value, tail_val = pearsonr(a_rating, b_rating)
        print(tags_a)
        print(tags_u)

    def create_tag_matrix(self, user_a):
        tag_size = len(self.tags)
        artist_size = len(self.artists)
        a_tags = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == user_a]
        adj_matrix = lil_matrix((artist_size, tag_size))
        for index, row in a_tags.iterrows():
            # print(row['tag_id'])
            adj_matrix[row['artist_id'], row['tag_id']] = 1
        # print(adj_matrix)
        return adj_matrix

    
    
    def try_all_users(self):
        user_a = 2
        tagged_users = self.user_taggedartists['user_id'].unique()
        for user_u in tagged_users:
            self.weight_tags(user_a, user_u)    



    def average_playcount(self):
        return self.user_artists['weight'].mean()
    
    def get_fraction_tag(self, user_a, user_u):

        a_tags = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == user_a]
        u_tags = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == user_u]

        combined_tags = pd.concat([a_tags, u_tags])
        tagged_artists = combined_tags['artist_id'].unique()

        avg_play = self.average_playcount()

        tag_weight = 0

        for artist in tagged_artists:
            art_tag_a = a_tags.loc[a_tags['artist_id'] == artist]
            art_tag_u = u_tags.loc[u_tags['artist_id'] == artist]
            denom_a = len(art_tag_a)
            denom_u = len(art_tag_u)

            common_tags = len(set(art_tag_a['tag_id']) & set(art_tag_u['tag_id']))
            if denom_a > 0 and denom_u > 0 and common_tags > 0:
                if tag_weight == 0:
                    tag_weight = 1
                # tag_weight = tag_weight + (common_tags/denom_a * common_tags/denom_u) * avg_play
                tag_weight = tag_weight * (common_tags/denom_a * common_tags/denom_u)
        return tag_weight

    
     
    def get_adjacency_matrix(self):
        avg_play = self.average_playcount()
        user_list = self.user_friends['user_id'].unique().tolist()
        mat_size = len(user_list)
        adj_matrix = lil_matrix((mat_size, mat_size))
        for user in user_list:
            user_ind = user_list.index(user)
            user_node = self.user_friends.loc[self.user_friends['user_id'] == user]
            
            friend_list = user_node['friend_id'].values
             
            for friend in friend_list:
                friend_ind = user_list.index(friend)
                adj_matrix[user_ind, friend_ind] = avg_play
                
        return adj_matrix, user_list
    
    def get_friend_weights(self, user_a, user_u, adj_matrix, user_list):
        user_ind = user_list.index(user_a)
        friend_ind = user_list.index(user_u)
        list_a = adj_matrix.getrow(user_ind)
        list_u = adj_matrix.getrow(friend_ind)

        array_a = list_a.toarray()
        array_u = list_u.toarray()
        p_value,tail_val = pearsonr(array_a[0], array_u[0])
        return p_value

    
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
        return p_value

    def combined_weights(self, alpha, beta, gamma, user_a, subset):
        user_weights = {}
        if subset:
            users = [922]
        else:
            users = self.user_artists.loc[self.user_artists['user_id']!= user_a]['user_id'].unique()
            print(users)

        bar = Bar('Processing', max=len(users))
        friend_martix, user_list = self.get_adjacency_matrix()

        for user_u in users:
            bar.next()
            friend_weight = self.get_friend_weights(user_a, user_u, friend_martix, user_list)
            rating_weight = self.weight_a_u(user_a, user_u)
            tag_weight = self.get_fraction_tag(user_a, user_u)
            # print()
            # print("Friend weight: " + str(friend_weight))
            # print("Rating weight: " + str(rating_weight))
            # print("Tag weight: " + str(tag_weight))

            user_weights[user_u] = alpha * rating_weight + beta * friend_weight + gamma * tag_weight
        bar.finish()

        sorted_weights = {k: v for k, v in sorted(user_weights.items(), key=lambda item: item[1], reverse = True)}
        knn_weights = dict(islice(sorted_weights.items(), 20))
        print(knn_weights)

        return knn_weights
        

        


# cf = weight_calc()
# subset = False
# user_a = 2
# user_u = 922
# alpha = 0.5
# beta = gamma = 0.25
# adj_matrix, user_list = cf.get_adjacency_matrix()
# cf.get_friend_weights(user_a, user_u, adj_matrix, user_list)
# cf.combined_weights(alpha, beta, gamma, user_a, subset)