# -*- coding: utf-8 -*-
import lstm_data_processing as ldp

test1 = "Hi there 😀 what are you up to tonight? 😣 SOOOO tired..."

test2 = "My number is 0773828495, DONT FORGET!!!!! also my email address is \
         bob@gmail.com. cya tmorrow @mark."
            
test3 = "I'm so HAPPPPYYYYYYYYYY today 😀 BECause maybe ill get a raise #awesome"

list_of_tests = [test1,test2,test3]

result = []

for test in list_of_tests:
    token_list = ldp.tokenizer1(test)
    refined_token_list = ldp.refine_token(token_list)
    result.append(refined_token_list)