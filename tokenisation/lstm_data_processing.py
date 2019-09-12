# -*- coding: utf-8 -*-

"""
make a function which takes as input a tweet (a string), and does the following:
    replace occurences of certain characters as follows:
        hashtags -> "<hashtags>"
        URLs -> "<url>"
        numbers -> "<number>"
        user mentions -> "<user>"
        
    replace emojis as follows:
        "<smile>", "<heart>", "<lolface>", "<neutralface>", "<angryface>"
    
    words written in all upper case e.g. "HAPPY", converted into two tokens:
        "<happy>" and "<allcaps>"
        
    **** not sure about this one ***** 
    words with repeated letters converted into two tokens, e.g.:
        "hhhhheeey" is converted into "<hey>" and "<repeatedletters>"
    
    all tokens converted to lower case
"""
def dat_proc(tweet):
    # replace occurences of certain characters
    
    for letter in tweet:
        if letter == '#':
            