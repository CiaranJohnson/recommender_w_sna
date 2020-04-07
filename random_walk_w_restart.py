import random
import numpy as np 
import pandas as pd
from scipy.sparse import lil_matrix
import operator

class random_walk_restart():
    def __init__(self):
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)


    def get_path(self, restart_prob, curr_node):
        user_node = self.user_friends.loc[self.user_friends['user_id'] == curr_node]
        num_nodes = user_node['friend_id'].size
        print(user_node)
        print(num_nodes)
        # subtract the probability of a restart and divide the number of nodes 
        node_prob = 1 - restart_prob
        node_probs = node_prob/ num_nodes


    # cant be a matrix try a dictionary of dictionaries. 
    def get_adjacency_matrix(self):
        # Both lists contain the same values if you use user_id or friend_id
        user_list = self.user_friends['user_id'].unique().tolist()
        mat_size = len(user_list)
        adj_matrix = lil_matrix((mat_size, mat_size))
        for user in user_list:
            user_ind = user_list.index(user)
            user_node = self.user_friends.loc[self.user_friends['user_id'] == user]
            num_nodes = user_node['friend_id'].size
            node_probs = 1/ num_nodes
            # find friends of user 
            friend_list = user_node['friend_id'].values
            # find their indexes in friend list 
            for friend in friend_list:
                friend_ind = user_list.index(friend)
                adj_matrix[user_ind, friend_ind] = node_probs
                # print(adj_matrix[user_ind, friend_ind])
        # print(adj_matrix)
        return adj_matrix

    def random_w_r(self, user, restart_prob):
        user_list = self.user_friends['user_id'].unique().tolist()
        mat_size = len(user_list)
        user_ind = user_list.index(user)
        s = np.zeros(mat_size)
        s[user_ind] = 1 
        x = s
        x_prev = s = np.zeros(mat_size)
        adj_matrix = self.get_adjacency_matrix()
        csr_adj_mat = adj_matrix.tocsr().transpose()
        while(not ((x == x_prev).all())):
            storage = restart_prob * s + (1 - restart_prob) * csr_adj_mat * x
            # print(storage)
            x_prev = x 
            x = storage
        # print(x)
        return x, user_list
    
    def n_top_influencers(self, start_node, restart_prob):
        print("Random Walk with Restart")
        x, user_list = self.random_w_r(start_node, restart_prob)
        d = {k:v for k, v in enumerate(x)}
        sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
        result_list = [k[0] for k in sorted_d][:3]
        # print(result_list)
        id_dict = {}
        for result in result_list:
            id_dict[user_list[result]] = x[result]
        # print(id_list)
        return id_dict


# rwr = random_walk_restart()
# start_node = 6
# restart_prob = 0.2
# print(rwr.n_top_influencers(start_node, restart_prob))
# user_friends = get_data()
# get_path(user_friends, restart_prob, start_node) 
# get_adjacency_matrix(user_friends)

# print(len(x))
# print(max(x))


# p_t is a column vector of all the users 