import CF_rec_me as cf 
import eval_metrics as weight_calc
from random_walk_w_restart import random_walk_restart
import sys
import os
from itertools import islice

if __name__ == "__main__":
    try:
        user_a = int(sys.argv[1])
        alpha = float(sys.argv[2])
        beta = float(sys.argv[3])
        gamma = float(sys.argv[4])
        weight_type = sys.argv[5]
        subset = sys.argv[6]
        restart_prob = float(sys.argv[7])
    except IndexError:
        print("Usage: " + os.path.basename(__file__) + " <user_a> <alpha> <beta> <gamma> <weight_type> <subset> <restart_prob>")
        sys.exit(1)
    
    weight_mat = weight_calc.weight_calc()
    cf_rec = cf.cf_me()
    rwr = random_walk_restart()


if restart_prob == 0:
    print("No Random Walk with Restart")
    if weight_type == "combined":
        if subset == "True":
            subset = True
        else:
            subset = False
        print("Combined")
        combined_weights = weight_mat.combined_weights(alpha, beta, gamma, user_a, subset)

    elif weight_type == "ratings":
        print("Ratings")
        k = 20
        combined_weights = cf_rec.knn(user_a, k)
    else:
        print(weight_type + " is an invalid weight type.\nEnter either: combined or rating")
        sys.exit(1)
    cf_rec.recommendations(user_a, combined_weights)
else:
    print("SNA")
    top_influencers = rwr.n_top_influencers(user_a, restart_prob)
    print(top_influencers)
    subset = False
    count = 1
    for influencer in top_influencers:
        print(influencer)
        combined_weights = weight_mat.combined_weights(alpha, beta, gamma, influencer, subset)
        rec_list, hits = cf_rec.recommendations(influencer, combined_weights)
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

    # b = 2
    # for a in x:
    #     print(x[a] * b)
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

    hits = cf_rec.compare_list(user_a, rec_list)
    print(hits)

