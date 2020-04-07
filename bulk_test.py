import CF_rec_me as cf 
import eval_metrics as weight_calc
from random_walk_w_restart import random_walk_restart
import friend_and_fof as FoF
import sys
import os
from itertools import islice
import time
import pandas as pd

class bulk_test:
    def __init__(self):

        self.cf_rec = cf.cf_me()
        self.fof = FoF.FriendsAndFof()
        self.rwr = random_walk_restart()
        self.weight_mat = weight_calc.weight_calc()

        remove_user_list = [6, 40, 133, 332, 491, 925, 1084, 1136, 1301, 1581, 912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
        # keep_user_ratings = {3: [129, 110, 121, 130, 123, 103, 137, 126, 133, 128, 139, 122, 150, 112, 132, 114, 102, 144, 149, 147, 107, 124, 141, 117, 119]
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

        self.keep_user_ratings = {3: [],6: [],12: []
        ,40:[1066, 1297, 1300],128:[]
        ,133: [89, 288]
        ,332: [292, 300, 333],478: []
        ,491: [917, 841]
        ,582: [],912: []
        ,925: [227, 614]
        ,1084: [64, 65, 56],1136: [288, 89, 292],1278: []
        ,1301: [289, 288, 292, 295, 89, 67],1375: [],1458: [],1509: [],1581: [439, 217, 898]
        }

        self.weight_mat.remove_test_val(remove_user_list)
        self.weight_mat.half_ratings(self.keep_user_ratings)


    def weigthed_cf(self, alpha, beta, gamma, user_a, subset):
        print("Combined")
        
        start = time.time()
        combined_weights = self.weight_mat.combined_weights(alpha, beta, gamma, user_a, subset)
        rec_list, hits = self.cf_rec.recommendations(user_a, combined_weights)
        end = time.time()

        print("Time to generate recommendations: " + str(end - start) + "\n\n")
        f.write("Rec List: " + str(rec_list) + "\n\n" + " Hits: " + str(hits) + "\n\n")
        f.write("Time to generate recommendations: " + str(end - start) + "\n\n")
        f.close()

    def other_friend_stuff(self):
        artist_dict, unique_tag_count = fof.essential_tags()
        friend_list.insert(0, user_a)
        print("Friend list after adding target user: " + str(friend_list))
        spare_mat, friend_index = fof.reduce_dimensions(artist_dict, unique_tag_count, friend_list)
        clusters = fof.svd_on_tags(spare_mat)
        user_index = friend_index[user_a]
        user_cluster = clusters[user_index]
        print("Target user cluster: " + str(clusters[user_index]))
        cluster_friends = []
        for friend in friend_index:
            mates_index = friend_index[friend]
            if clusters[mates_index] == user_cluster:
                cluster_friends.append(friend)
        print("Cluster friends: " + str(cluster_friends))

    
    def friend_of_friend(self, user_a, alpha, beta):
        print("FoF")
        start = time.time()
        # Get all the friends of the target user
        friend_list = self.fof.get_friends(user_a)

        # Find all the similarity score for the friends tagging information
        # Add all friends who have tagged at least one artist the same to
        # a list of similar friends
        friend_tag_weight = {}
        similar_friend = []
        for friend in friend_list:
            friend_tag_weight[friend] = self.weight_mat.get_fraction_tag(user_a, friend)
            if friend_tag_weight[friend] > 0:
                similar_friend.append(friend)

        print("Friend tag weights: " + str(friend_tag_weight))
        print("Similar Friend: " + str(similar_friend))
        f.write("Similar Friend: " + str(similar_friend) + "\n")

        # 
        fof_weight = self.fof.handle_friend_rec("fof", user_a, count_only=False)
        print(fof_weight)


        sim_weights = pd.Series()
        if len(similar_friend) > 0:
            print("Sim Friends baby")
            sim_weights = self.fof.get_sim_friend_weight(similar_friend)

        alpha_friend = fof_weight.multiply(alpha)
        beta_friend = sim_weights.multiply(beta)

        total_weights = alpha_friend.add(beta_friend, fill_value=0)

        rec_list = self.fof.get_top_artists(total_weights)
        print("Rec List: " + str(rec_list))
        hits = self.cf_rec.compare_list(user_a, rec_list)

        end = time.time()
        f.write("Rec List: " + str(rec_list) + "\n\n" + " Hits: " + str(hits))
        f.write("Time to generate recommendations: " + str(end - start) + "\n\n")
        f.close()


    def sna_combine_influencers(self, first_list, first_score, second_list, second_score, third_list, third_score, f, start):
        print("<============================================>")
        print("First list:")
        for first_item in first_list:
            first_list[first_item] = first_list[first_item] * first_score
        print(first_list)
        print("<============================================>")
        print("Second list:")
        print(second_list)
        for second_item in second_list:
            second_list[second_item] = second_list[second_item] * second_score
        print("<============================================>")
        print("Third list:")
        print(third_list)
        for third_item in third_list:
            third_list[third_item] = third_list[third_item] * third_score
        print("<============================================>")

        all_keys_list = []
        all_keys_list.extend(list(first_list.keys()))
        all_keys_list.extend(list(second_list.keys()))
        all_keys_list.extend(list(third_list.keys()))
        unique_keys = set(all_keys_list)
        output = {}
        for single_key in unique_keys:
            print(single_key)
            score = 0
            if single_key in first_list:
                score += first_list[single_key]
            if single_key in second_list: 
                score += second_list[single_key]
            if single_key in third_list: 
                score += third_list[single_key]
            output[single_key] = score
        # print(output)

        rating_dict = {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse = True)}
                # print(rating_dict)

        rec_list = dict(islice(rating_dict.items(), 20))
        print(rec_list)

        hits = self.cf_rec.compare_list(user_a, rec_list)
        print(hits)
        end = time.time()

        print("Time to generate recommendations: " + str(end - start) + "\n\n")
        f.write("Rec List: " + str(rec_list) + "\n\n" + " Hits: " + str(hits))
        f.write("Time to generate recommendations: " + str(end - start) + "\n\n")
        f.close()



if __name__ == "__main__":
    # users = [6,40,133,332,491,925,1084,1136,1301,1581]
    users = [133, 491, 925]
    
    tests = bulk_test()


    parameters = {'friend_remove_1':[0, 1, 0, 'friends', False, 0], 'cf_add_1': [0.8, 0.15, 0.05, 'combined', False, 0]}
    for p in parameters:
        print(p)
        alpha = parameters[p][0]
        beta = parameters[p][1]
        gamma = parameters[p][2]
        weight_type = parameters[p][3]
        subset = parameters[p][4]
        restart_prob = parameters[p][5]

        for user_a in users:
            print(f"User A: ",user_a)
            print(alpha)
            print(beta)
            print(gamma)

            file_name = "Final_results/user_" + str(user_a) + "/test_big_"+str(p) + "_" + str(user_a) +".txt"
            f = open(file_name, "w")
            f.write("Alpha: " + str(alpha) + " Beta: "+ str(beta) + " Gamma: " + str(gamma) + "\n")
            f.write("Weight type: " + str(weight_type) + " restart probability: " + str(restart_prob) + "\n\n")

            if restart_prob == 0:
                print("No Random Walk with Restart")
                if weight_type == "combined":
                    tests.weigthed_cf(alpha,beta,gamma,user_a,subset)
                elif weight_type == "ratings":
                    print("Ratings")
                    k = 20
                    combined_weights = tests.cf_rec.knn(user_a, k)
                elif weight_type == "fof":
                    tests.friend_of_friend(user_a, alpha, beta)
                elif weight_type == "friends":
                    print("Friends")
                    remove_artist = tests.keep_user_ratings[user_a]
                    start = time.time()
                    rec_list = tests.fof.handle_friend_rec("friends", user_a, count_only=True, remove_artist = remove_artist)
                    hits = tests.cf_rec.compare_list(user_a, rec_list)

                    end = time.time()
                    f.write("Rec List: " + str(rec_list) + "\n\n" + " Hits: " + str(hits))
                    f.write("Time to generate recommendations: " + str(end - start) + "\n\n")
                    f.close()
                else:
                    print(weight_type + " is an invalid weight type.\nEnter either: combined or rating")
                    sys.exit(1)
            else:
                print("SNA")
                start = time.time()
                top_influencers = tests.rwr.n_top_influencers(user_a, restart_prob)
                print(top_influencers)
                subset = False
                count = 1
                for influencer in top_influencers:
                    print(influencer)
                    combined_weights = tests.weight_mat.combined_weights(alpha, beta, gamma, influencer, subset)
                    rec_list, hits = tests.cf_rec.recommendations(influencer, combined_weights)
                    influencer_score = top_influencers[influencer]

                    if count == 1:
                        print("first")
                        # find out what is in the rec list - if it is more of a dictionary then use these values
                        first_list = rec_list
                        first_hit = hits
                        first_score = influencer_score
                    elif count == 2:
                        print("second")
                        second_list = rec_list 
                        second_hit = hits
                        second_score = influencer_score
                    elif count == 3:
                        print("third")
                        third_list = rec_list 
                        third_hit = hits
                        third_score = influencer_score
                    else:
                        print("Error wrong count number")
                        sys.exit(1)
                    count = count + 1
                tests.sna_combine_influencers(first_list, first_score, second_list, second_score, third_list, third_score, f, start)
            

