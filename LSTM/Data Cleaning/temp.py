# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 20:54:03 2019

@author: Kumar
"""

with open(os.path.join(entry[0], entry[1], 'tweets.csv'), 'r',
          encoding="Latin-1") as r:
        reader = csv.reader(r)
        counter = 100 * [0]
        except_count = 0
        while True:
            row = next(reader)
            size = len(row)
            counter[size] += 1
            if size != 27:
                except_count+=1
                if except_count == 100:
                    print(row)
                    break
                
                