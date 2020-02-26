#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:27:19 2020

@author: s1654111
"""
import pandas as pd

class test_generator:
    def __init__(self):
        self.artists = pd.read_csv('lastFmData/artists.dat', sep='\t', names=['id','name','url','pictureURL'], skiprows = 1)
        self.tags = pd.read_csv('lastFmData/tags.dat', sep='\t', names=['tagID','tagValue'], skiprows = 1)
        self.user_artists = pd.read_csv('lastFmData/user_artists.dat', sep='\t', names=['user_id','artist_id','weight'], skiprows = 1)
        self.user_friends = pd.read_csv('lastFmData/user_friends.dat', sep='\t', names=['user_id','friend_id'], skiprows=1)
        self.user_taggedartists = pd.read_csv('lastFmData/user_taggedartists.dat', sep='\t', names=['user_id','artist_id', 'tag_id', 'day', 'month', 'year'], skiprows=1)
        self.user_taggedartists_timestamps = pd.read_csv('lastFmData/user_taggedartists-timestamps.dat', sep='\t', names=['user_id','artist_id','tag_id','timestamp'], skiprows=1)

    def get_user_info(self, userlist):
        for user in userlist:
            print(user)
            num_friends = self.user_friends.loc[self.user_friends['user_id'] == user]
            print(num_friends)
            print(len(num_friends))
            num_tags = self.user_taggedartists.loc[self.user_taggedartists['user_id'] == user]
            print(num_tags)
            print(len(num_tags))




if __name__ == "__main__":
    userlist = [6,40,133,332,491,925,1084,1136,1301,1581]
    test_gen = test_generator()
    test_gen.get_user_info(userlist)    

