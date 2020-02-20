# -*- coding: utf-8 -*-
from Twitter_API import config

# import authentication details for twitter developer account
try:
    consumer_key = config.consumer_key
    consumer_secret = config.consumer_secret
    access_token = config.access_token
    access_token_secret = config.access_token_secret
except:
    print("Config file for authentication information missing")