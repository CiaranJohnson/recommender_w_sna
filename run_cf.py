import CF_rec_me as cf 
import eval_metrics as weight_calc
from random_walk_w_restart import random_walk_restart
import sys
import os

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
        combined_weights = weight_mat.combined_weights(alpha, beta, gamma, influencer, subset)
        rec_list, hits = cf_rec.recommendations(influencer, combined_weights)
        if count == 1:
            print("first")
            first_list = rec_list
            first_hit = hits 
        elif count == 2:
            print("second")
            second_list = rec_list 
            second_hit = hits
        elif count == 3:
            print("third")
            third_list = rec_list 
            third_hit = hits 
        else:
            print("Done")
            sys.exit(1)
        count = count + 1

