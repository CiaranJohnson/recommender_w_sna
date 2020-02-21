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
        user_list = list(self.user_artists['user_id'].unique())
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
    test_gen.get_random_user()
