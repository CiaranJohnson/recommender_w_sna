import pandas as pd
from itertools import islice
import CF_rec_me

class baseline:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    def remove_test_users(self, test_users):
        removed_users = self.user_artists
        for user in test_users:
            indices = list(removed_users.index[removed_users['user_id'] == user])
            removed_users = removed_users.drop(indices)
        return removed_users


    def get_top_20_artists(self, removed_users):
        count_artists = {}
        artists_list = removed_users['artist_id'].unique()
        for artist in artists_list:
            count = len(removed_users.loc[removed_users['artist_id'] == artist])
            count_artists[artist] = count

        sorted_weights = {k: v for k, v in sorted(count_artists.items(), key=lambda item: item[1], reverse = True)}
        top_artists = dict(islice(sorted_weights.items(), 20))
        return top_artists
    
    def find_hits(self, users, rec_list):
        for user in users:
            file_name = "Final_results/user_" + str(user) + "/test_baseline_popular_"+ str(user) +".txt"
            f = open(file_name, "w")
            user_list = self.user_artists.loc[self.user_artists['user_id'] == user]['artist_id'].tolist()
            print(str(user) + ": " + str(set(user_list) & set(rec_list)))
            f.write("Rec List: " + str(rec_list) + "\n\n" + " Hits: " + str(set(user_list) & set(rec_list)))
            f.close()
        

    def display_list(self, rec_list):
        print("Your recommendations are: ")
        print("***************************************")
        for rec in rec_list:
            print(str(rec) + ": " + self.artists.loc[self.artists['id'] == rec]['name'].item())
            # print(str(rec) + ": " + str(rec_list[rec]))
        
        print("*************************************** \n")



if __name__ == "__main__":
    baseline = baseline()
    cf = CF_rec_me.cf_me()
    users = [6, 40, 133, 332, 491, 925, 1084, 1136, 1301, 1581, 912, 12, 128, 1458, 582, 3, 1375, 478, 1278, 1509]
    
    removed_users = baseline.remove_test_users(users)
    top_artists = baseline.get_top_20_artists(removed_users)
    print(top_artists)
    rec_list = list(top_artists.keys())

    print(rec_list)

    baseline.find_hits(users, rec_list)

    baseline.display_list(rec_list)
    
 