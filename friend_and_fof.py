import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

class FriendsAndFof:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    def find_best_values(self, list_a, list_b, k_left):
        print(len(list_a))
        result = list_b.filter(items = list_a)
        return list(result.nlargest(n=k_left).keys())

    def get_top_artists(self, series):
        return list(series.nlargest(n=20).keys())


    # find the users friends and all the artists that their friends listen to
    def find_friends_artists(self, user_u):
        u_friends = self.user_friends.loc[self.user_friends["user_id"] == user_u]
        u_friend_list = list(u_friends['friend_id'])
        artist_pd = pd.DataFrame()
        for friend in u_friend_list:
            friend_pd = self.user_artists.loc[self.user_artists["user_id"] == friend]
            artist_pd = pd.concat([artist_pd, friend_pd])
        return artist_pd
        
    def get_sum_count(self, artist_pd):
        sum_artist = artist_pd.groupby('artist_id')['weight'].sum()
        count_artist = artist_pd.groupby('artist_id')['weight'].count()
        return count_artist, sum_artist
    
    def get_rec_list(self, count_artist, sum_artist, remove_artist):
        print(type(count_artist))
        rec = []
        i = 1
        while(len(rec) < 20):
            max_count = count_artist.max()
            max_list = list(count_artist[count_artist == max_count].keys())
            if(len(rec) + len(max_list) > 20):
                k_left = 20 - len(rec)
                max_list = self.find_best_values(max_list, sum_artist, k_left)
            
            max_list = [i for i in max_list if i not in remove_artist] 
            rec.extend(max_list)
            count_artist = count_artist[count_artist != max_count]
            i = i+1
        print(rec)
        return rec

    


    def handle_friend_rec(self, type_rec, user_u, count_only=True, remove_artist = []):
        if (type_rec == "fof"):
            print("Fof")

            # Get a list of all the users friends and the artists they listen to 
            u_friends = self.user_friends.loc[self.user_friends["user_id"] == user_u]
            artist_pd = self.find_friends_artists(user_u)

            # get the total users that listen to each artist and the sum of their
            # time listened to each
            count_artist, sum_artist = self.get_sum_count(artist_pd)
            max_rating = artist_pd['weight'].max()

            # Repeat the steps for all the friends of the target users friends
            for user in u_friends['friend_id']:
                artist_pd = self.find_friends_artists(user)
                friend_count_artist, friend_sum_artist = self.get_sum_count(artist_pd)
                count_artist = count_artist.add(friend_count_artist, fill_value = 0)
                sum_artist = sum_artist.add(friend_sum_artist, fill_value = 0)
                
                # this does not find 
                friend_max_rating = artist_pd['weight'].max()
                if friend_max_rating > max_rating:
                    max_rating = friend_max_rating
        elif(type_rec == "friends"):
            # Get just the target users friends recommendations
            artist_pd = self.find_friends_artists(user_u)
            count_artist, sum_artist = self.get_sum_count(artist_pd)
        else:
            print("Handle Friend Rec error: fof or friends")
            
        
        if count_only:
            return self.get_rec_list(count_artist, sum_artist, remove_artist)
        else:
            # divide the sum of all artist by the highest listened to artist 
            # add this to the count
            # This weights listen count highly
            ratio = sum_artist.divide(max_rating)
            friend_weights = count_artist.add(ratio)
        return friend_weights 
    
    def get_sim_friend_weight(self, u_friend_list):
        artist_pd = pd.DataFrame()
        for friend in u_friend_list:
            friend_pd = self.user_artists.loc[self.user_artists["user_id"] == friend]
            artist_pd = pd.concat([artist_pd, friend_pd])
        max_rating = artist_pd['weight'].max()
        count_artist, sum_count = self.get_sum_count(artist_pd)
        ratio = sum_count.divide(max_rating)
        combined_ratio = count_artist.add(ratio)
        print(combined_ratio)
        return combined_ratio

    
    def k_means(self, list_of_users):
        tag_list = self.user_taggedartists
        print(tag_list)
        for user in list_of_users:
            tag_list = tag_list.loc[tag_list['user_id'] != user]
        print(tag_list)
        user_tag = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == 6]
        kmeans = KMeans(2, random_state = 0).fit(tag_list)

        

    def essential_tags(self):
        artist_dict = {}
        count = 0
        for artist in self.user_taggedartists['artist_id'].unique():
            artist_pd = self.user_taggedartists.loc[self.user_taggedartists['artist_id'] ==  artist]
            tag_list = list(artist_pd['tag_id'].unique())
            if len(tag_list) > 0:
                tag_list.insert(0, count)
                artist_dict[artist] = tag_list
                count += len(tag_list)
        return artist_dict, count

    def not_in_ratings(self):
        tag_art = set(self.user_taggedartists['artist_id'].unique())
        rate_art = set(self.user_artists['artist_id'].unique())
        art = set(self.artists['id'].unique())

        remove_artists = tag_art - art
        print(rate_art - art)
        remove_a2 = tag_art - rate_art

        
    
    def reduce_dimensions(self, artist_dict, unique_tag_count, users):
        user_count_mapping = {}
        count = 0
        sparse_tag = lil_matrix((len(users), unique_tag_count))
        for user in users:
            user_count_mapping[user] = count
            user_tags = self.user_taggedartists[self.user_taggedartists['user_id'] == user]
            artist_user_tagged = user_tags['artist_id'].unique()
            for artist_t in artist_user_tagged:
                tag_list = list(user_tags.loc[user_tags['artist_id'] == artist_t]['tag_id'])
                artist_tags = artist_dict[artist_t]
                base_index = artist_tags[0]
                for tag in tag_list:
                    index = artist_tags.index(tag)
                    sparse_tag[count, (base_index + (index-1))] = 1
            count += 1
        return sparse_tag, user_count_mapping

    def get_users(self, all_users):
        if all_users:
            return list(self.user_taggedartists['user_id'].unique())
        else:
            return [6,40,133,332,491,925,1084,1136,1301,1581]

# TODO: remove the test users from the tagging clusters 
    def svd_on_tags(self, sparse_tag):
        svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        x_new = svd.fit_transform(sparse_tag.tocsr())

        # k_means = KMeans(2, random_state = 0).fit_predict(x_new)
        k_means = KMeans(2, random_state = 0).fit_predict(sparse_tag.tocsr())
        print(k_means)
        zero_count = 0
        one_count = 0
        for k in k_means:
            if k == 0:
                zero_count += 1
            else:
                one_count += 1
        print("zero count: " + str(zero_count))
        print("one count: " + str(one_count))
        return k_means
        
    def get_friends(self, user):
        friends = list(self.user_friends.loc[self.user_friends['user_id'] == user]['friend_id'])
        print(friends)
        return friends

        # print(k_means.predict(y_new))

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
                overwrite_tag = tag_pd.replace({'user_id':user}, max_id)sss
                fake_df = pd.DataFrame({'user_id':[max_id], 'artist_id':[0], 'weight':[max_weight]})
                fake_friends = pd.DataFrame({'user_id':[max_id], 'friend_id':[max_id]})
                overwrite_user = overwrite_user.append(fake_df)
                max_id = max_id + 1
                # print(overwrite_user)
                # print(overwrite_tag)

                self.user_artists = self.user_artists.append(overwrite_user)
                self.user_taggedartists = self.user_taggedartists.append(overwrite_tag)
                self.user_friends = self.user_friends.append(fake_friends)
                # print(max_id)

        # print(self.user_artists.loc[self.user_artists['artist_id'] == 0])
        # print(self.user_taggedartists)
        









# fof = FriendsAndFof()
# type_rec = "fof"
# user_u = 6
# # fof.create_scores(user_u)
# fof.handle_friend_rec(type_rec, user_u)
# artist_dict, unique_tag_count = fof.essential_tags()
# all_users = fof.get_users(True)
# sparse_tag, user_count_mapping = fof.reduce_dimensions(artist_dict, unique_tag_count, all_users)
# user = user_count_mapping[6]
# print(user)
# # test_users = fof.get_users(False)
# # test_sparse, test_count_mapping = fof.reduce_dimensions(artist_dict, unique_tag_count, test_users)
# fof.svd_on_tags(sparse_tag)
# fof.not_in_ratings()
# fof.reduce_dimensions()