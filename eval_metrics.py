import pandas as pd
# from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
# from surprise import Dataset
# from surprise.reader import Reader
# from surprise.model_selection import cross_validate
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

        # avg_play = self.average_playcount()

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

        
    def get_rating_weights(self, a_value, u_value):
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
        print(self.user_artists)
        user_weights = {}
        if subset:
            users = [922]
        else:
            users = self.user_artists.loc[self.user_artists['user_id']!= user_a]['user_id'].unique()

        bar = Bar('Processing', max=len(users))
        friend_martix, user_list = self.get_adjacency_matrix()

        for user_u in users:
            bar.next()
            friend_weight = self.get_friend_weights(user_a, user_u, friend_martix, user_list)
            rating_weight = self.get_rating_weights(user_a, user_u)
            tag_weight = self.get_fraction_tag(user_a, user_u)
            user_weights[user_u] = alpha * rating_weight + beta * friend_weight + gamma * tag_weight
        bar.finish()

        sorted_weights = {k: v for k, v in sorted(user_weights.items(), key=lambda item: item[1], reverse = True)}
        knn_weights = dict(islice(sorted_weights.items(), 20))
        print(knn_weights)

        return knn_weights

    
    def remove_test_val(self, user_list):
        self.removed_users = pd.DataFrame()
        # TODO: Remove all test and validation users from the dataset
        for user in user_list:
            self.removed_users = pd.concat([self.removed_users, self.user_artists.loc[self.user_artists['user_id']==user]])
            self.user_artists = self.user_artists.loc[self.user_artists['user_id'] != user]
        # TODO: Store all test and user information for when they are to be used
        

    
    def half_ratings(self, user_ratings):
        self.half_ratings_df = pd.DataFrame()
        for user in user_ratings:
            user_df = self.removed_users.loc[self.removed_users['user_id'] == user]
            for rating in user_ratings[user]:
                rating_df = user_df.loc[user_df['artist_id'] == rating]
                self.half_ratings_df = pd.concat([self.half_ratings_df, rating_df])
        self.user_artists = pd.concat([self.user_artists, self.half_ratings_df])
                

        


# cf = weight_calc()
# user_list = [6, 40, 133, 332, 491, 925, 1084, 1136, 1301, 1581, 912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
# user_ratings = {3: [129, 110, 121, 130, 123, 103, 137, 126, 133, 128, 139, 122, 150, 112, 132, 114, 102, 144, 149, 147, 107, 124, 141, 117, 119]
# ,6: [285, 277, 284, 279, 249, 240, 276, 282, 259, 77, 247, 255, 251, 241, 269, 271, 257, 286, 267, 270, 246, 258, 260, 263, 265]
# ,12: [494, 487, 508, 333, 512, 230, 488, 378, 496, 405, 511, 520, 377, 466, 509, 489, 513, 486, 501, 506, 505, 498, 514, 492, 191]
# ,40:[1322, 1327, 1323, 1328, 1315, 1299, 1320, 1326, 1298, 1329, 1313, 1310, 1292, 1316, 32, 1308, 1066, 1296, 1335, 1318, 1332, 1304, 1303, 1336, 1311]
# ,128:[3108, 217, 3101, 3099, 3103, 1470, 3113, 1075, 3121, 193, 622, 1519, 962, 3102, 3106, 51, 3114, 206, 72, 3120, 453, 3098, 3118, 211, 3116]
# ,133: [3174, 332, 1043, 1429, 466, 528, 89, 498, 302, 1042, 486, 3172, 461, 2548, 802, 349, 291, 311, 157, 67, 464, 681, 329, 300, 3175]
# ,332: [230, 311, 5765, 1249, 5766, 305, 689, 2392, 316, 3763, 97, 4489, 697, 704, 2082, 378, 542, 300, 7, 466, 55, 720, 180, 4575, 256]
# ,478: [4823, 7337, 1974, 3444, 706, 7331, 1545, 5450, 4834, 366, 1669, 4820, 5630, 3398, 7329, 4375, 7330, 498, 1645, 424, 4821, 4793, 250, 412, 4822]
# ,491: [920, 1206, 1364, 7419, 1260, 1369, 1358, 1975, 917, 949, 3624, 2765, 821, 2744, 1528, 706, 533, 12, 7420, 1257, 843, 2834, 1360, 1523, 3509]
# ,582: [832, 1555, 4004, 823, 1557, 2143, 2576, 1568, 8294, 845, 4498, 1044, 8302, 2586, 8301, 1196, 813, 3548, 805, 5135, 8292, 1551, 831, 1549, 3997]
# ,912: [6282, 7097, 734, 2834, 942, 1803, 1249, 10994, 1810, 2346, 732, 735, 932, 3427, 1513, 2797, 1339, 1654, 163, 1372, 2038, 1412, 917, 10996, 4165]
# ,925: [3063, 3060, 227, 4037, 610, 683, 533, 72, 1118, 697, 233, 1092, 1985, 5249, 154, 53, 204, 316, 986, 701, 599, 1131, 6225, 1244, 238]
# ,1084: [1109, 1983, 4728, 4247, 1398, 969, 441, 1390, 620, 3259, 207, 3767, 173, 718, 959, 56, 12195, 65, 1073, 225, 425, 7743, 1062, 757, 4922]
# ,1136: [691, 55, 292, 5149, 333, 2392, 461, 464, 7706, 1481, 524, 526, 2083, 289, 3322, 689, 2543, 10684, 679, 3404, 2521, 288, 430, 1059, 89]
# ,1278: [1009, 191, 7268, 701, 288, 1444, 300, 680, 4564, 2498, 466, 333, 5530, 1057, 1456, 7868, 312, 10987, 475, 1042, 9684, 1449, 5988, 545, 464]
# ,1301: [289, 5036, 67, 89, 3488, 257, 55, 972, 288, 8363, 3200, 302, 292, 1601, 291, 3575, 1098, 375, 304, 301, 534, 680, 389, 3206, 298]
# ,1375: [2864, 429, 2579, 707, 6906, 3197, 805, 2608, 3482, 607, 2711, 9716, 1244, 14201, 14196, 2575, 3405, 11032, 1369, 14198, 7804, 1104, 163, 10613, 1195]
# ,1458: [4292, 679, 1444, 5594, 536, 1047, 299, 7449, 8364, 1243, 184, 89, 239, 67, 293, 1021, 2080, 257, 455, 4468, 329, 481, 342, 486, 289]
# ,1509: [291, 461, 207, 1241, 2080, 306, 1246, 498, 4463, 321, 3378, 538, 65, 982, 540, 4484, 190, 4474, 198, 328, 89, 4828, 1673, 333, 4471]
# ,1581: [3186, 439, 5439, 15424, 11484, 5217, 15427, 840, 12104, 11215, 222, 615, 5549, 199, 7832, 173, 1043, 3333, 1383, 898, 15423, 2661, 1470, 5027, 1239]
# }
# cf.remove_test_val(user_list)
# cf.half_ratings(user_ratings)
# subset = False
# user_a = 40
# user_u = 922
# alpha = 0.5
# beta = gamma = 0.25
# # cf.weight_a_u(user_a, user_u)
# # adj_matrix, user_list = cf.get_adjacency_matrix()
# # cf.get_friend_weights(user_a, user_u, adj_matrix, user_list)
# cf.combined_weights(alpha, beta, gamma, user_a, subset)