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


# get tweets from US election
for tweet in tweepy.Cursor(api.search, q='#uselection', count=100, lang='en',
                           since='2020-02-19').items():
    additional_info = [tweet.user, tweet.tweet, tweet.created_at]
    metadata = [tweet.retweet_count, tweet.reply_count,
                tweet.favorite_count, tweet.num_hashtags,
                tweet.num_urls, tweet.num_mentions]
    interaction_info = [tweet.in_reply_to_status_id, tweet.in_reply_to_user_id]
    tweets_list.append(additional_info + metadata)
