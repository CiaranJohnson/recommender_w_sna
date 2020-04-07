import random
import pandas as pd

class test_generator:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    def get_random_user(self):
        test_list = []
        validation_set = [6,40,133,332,491,925,1084,1136,1301,1581]
        test_set = [912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
        full_set = [6, 40, 133, 332, 491, 925, 1084, 1136, 1301, 1581, 912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
        user_list = list(self.user_artists['user_id'].unique())
        user_list = [i for i in user_list if i not in validation_set]
        print(len(user_list))
        while len(test_list) < 10:
            user = random.randint(0, len(user_list))
            print(user)
            user_connections = self.how_many_friends(user)
            user_ratings = self.how_many_rated(user)
            if user_connections > 3 and user_ratings == 50:
                print("more than 3 friends and 50 artists rated")
                test_list.append(user)
                user_list.remove(user)
        print(test_list)

    def pick_half_ratings(self):
        full_set = [6, 40, 133, 332, 491, 925, 1084, 1136, 1301, 1581, 912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
        for user in full_set:
            file_name = "test_data/rating_set_"+str(user)+".txt"
            f = open(file_name, "w")
            f.write("User: " + str(user) + "\n\n")
            print(f"User: ", user)
            art_list = list(self.user_artists.loc[self.user_artists['user_id']==user]['artist_id'].unique())
            f.write("Art List: " + str(art_list) + "\n\n")
            print(f"Art List: ", art_list)
            user_list = []
            while len(user_list) < 25:
                art_num = random.randint(0, len(art_list)-1)
                # print(art_num)
                art = art_list[art_num]
                user_list.append(art)
                art_list.remove(art)
            f.write("User List: " + str(user_list) + "\n\n")
            print(f"User List: ", user_list)




    def how_many_friends(self, user):
        user_connections = self.user_friends.loc[self.user_friends['user_id'] == user]
        print(user_connections)
        print("Connections: " + str(len(user_connections)))
        return len(user_connections)
    
    def how_many_rated(self, user):
        user_ratings = self.user_artists.loc[self.user_artists['user_id'] == user]
        print(user_ratings)
        print("Friends: " + str(len(user_ratings)))
        return len(user_ratings)



if __name__ == "__main__":
    test_gen = test_generator()
    # test_gen.get_random_user()
    test_gen.pick_half_ratings()
