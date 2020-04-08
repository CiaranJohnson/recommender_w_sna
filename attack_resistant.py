import pandas as pd

class attack_resistant:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    
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
                overwrite_user = overwrite_user.append(fake_df)
                max_id = max_id + 1
                # print(overwrite_user)
                # print(overwrite_tag)

                self.user_artists = self.user_artists.append(overwrite_user)
                self.user_taggedartists = self.user_taggedartists.append(overwrite_tag)
                # print(max_id)

        print(self.user_artists.loc[self.user_artists['artist_id'] == 0])
        # print(self.user_taggedartists)

            



ar = attack_resistant()
ar.add_users([3,12])