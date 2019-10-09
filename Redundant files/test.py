# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:58:30 2019

@author: Kumar
"""
a = myModel(embed_mat)
b = a.genData()
start = time.time()
for i in range(100):
    next(b)
end = time.time()

test = myModel(embed_mat)
begin = time.time()
test.fit()
end = time.time()
