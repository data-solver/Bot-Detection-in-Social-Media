# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:54:03 2019

@author: Kumar
"""

with open(os.path.join(entry[0], entry[1], 'tweets.csv'), 'r',
          encoding="Latin-1") as r:
        reader = csv.reader(r)
        header = next(reader)
        counter = 100 * [0]
        except_count = 0
        blah = 0
        length = 0
        while True:
            row = next(reader)
            length += 1
            if row[9:13] == test:
                blah +=1
                size = len(row)
                counter[size] += 1
#            try:
#                if row[3] == '887281':
#                    print(row)
#                    break
#            except:
#                continue
#
#            if size != 27:
#                except_count+=1
                
                
                
                
#    header = ['id', 'text', 'source', 'user_id', 'truncated', 
#              'in_reply_to_status_id', 'in_reply_to_user_id',
#              'in_reply_to_screen_name', 'retweeted_status_id', 'geo', 'place',
#              'contributors', 'unknown', 'retweet_count', 'reply_count', 'favorite_count',
#              'favorited', 'retweeted', 'possibly_sensitive', 'num_hashtags',
#              'num_urls', 'num_mentions', 'created_at', 'timestamp',
#              'crawled_at', 'updated']