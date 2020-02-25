# -*- coding: utf-8 -*-
import time
import tweepy
import csv
import os
from Twitter_API import config
from data_preprocessing import lstm_data_processing as ldp

data_dir = "./Datasets/US Primaries"

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
#api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
api = tweepy.API(auth)

text_query = '2020 US Election'
count = 100
n = None
nt = 0


# get tweets from US election
c = tweepy.Cursor(api.search, q='#uselection OR #primaries',
                  lang='en', since='2020-02-20',
                  tweet_mode='extended').items()
stop = False
# ================= missing reply count
header = ['text', 'user_id', 'created_at', 'retweet_count', 'favorite_count',
          'num_hashtag', 'num_urls', 'num_mentions']
with open(os.path.join(data_dir, 'usdata.csv'), 'w', newline='',
          encoding="utf-8") as w:
    writer = csv.writer(w)
    writer.writerow(header)
    while not stop:
        try:
            tweet = next(c)
            # handling truncated tweets
            try:
                text = tweet.retweeted_status.full_text
            except:
                text = tweet.full_text
            additional_info = [text, tweet.user.id, tweet.created_at]
            # get tokenized version of tweet
            token = ldp.tokenizer1(text)
            token = ldp.refine_token(token)
            # get counts for metadata
            num_hashtags = token.count('<hashtag>')
            num_urls = token.count('<url>')
            num_mentions = token.count('<user>')
            # ----------------- missing reply_count for now
            metadata = [tweet.retweet_count, tweet.favorite_count,
                        num_hashtags, num_urls, num_mentions]
            interaction_info = [tweet.in_reply_to_status_id,
                                tweet.in_reply_to_user_id]
            row = additional_info + metadata
            writer.writerow(row)
            nt += 1
            if nt == n:
                break
        except tweepy.TweepError:
            try:
                time.sleep(60*15)
                print("max rate, sleeping")
            except KeyboardInterrupt:
                stop = True
                break
            continue
        except StopIteration:
            print("end of search")
            break
