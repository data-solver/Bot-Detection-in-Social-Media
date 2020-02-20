# -*- coding: utf-8 -*-
from Twitter_API import config
import time
import tweepy

# import authentication details for twitter developer account
try:
    consumer_key = config.consumer_key
    consumer_secret = config.consumer_secret
    access_token = config.access_token
    access_token_secret = config.access_token_secret
except:
    print("Config file for authentication information missing")
    
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)    

tweets_list = []
text_query = '2020 US Election'
count = 100


try:
# Pulling individual tweets from query
    for tweet in api.search(q=text_query, count=count):
# Adding to list that contains all tweets
      tweets.append((tweet.created_at,tweet.id,tweet.text))
except BaseException as e:
    print('failed on_status,',str(e))
    time.sleep(3)
