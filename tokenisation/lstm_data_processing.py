# -*- coding: utf-8 -*-
"""
make a function which takes as input a tweet (a string), and does the following:
    replace occurences of certain characters as follows:
        hashtags -> "<hashtag>"
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
import nltk
import re


class token_class:
    def __init__(self, token):
        self.token = token

    def is_url(self):
        """
        checks if token is a url
        """
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|\
                          (?:%[0-9a-fA-F][0-9a-fA-F]))+', self.token)
        
        if len(urls) == 0:
            return False
        else:
            return True

    def is_number(self):
        """
        checks if token is a number
        """
        try:
            float(self.token)
            return True
        except ValueError:
            return False

    def is_user_mention(self):
        """
        checks if token is a user mention
        """
        
        return None

    def is_emoji(self):
        """
        checks if token is an emoji
        if it is, it returns True and the type of emoji that the token is
        """
        
        return [None, None]
    
    def is_repeated(self):
        """
        checks if token has repeated letters
        if it does, it returns True and the word without repeated letters
        if it does not, it returns False and the original word
        """
        output = [None, None]

        
        return [None, None]
    
#    def is_allcaps(self):
#        """
#        checks if token has all capital letters
#        if it does, returns True, else False
#        """
#        if self.token.isupper() == True
#        return None
    
    def get_token(self):
        return self.token

def tokenizer(tweet):
    """
    splits tweet into initial list of tokens
    """
    tknzr = nltk.TweetTokenizer(preserve_case = False, strip_handles = False)
    token_list = tknzr.tokenize(tweet)
    return token_list

def refine_token(token_list):
    """
    refines list of tokens to be in line with rules at top of this file
    """
    
    refined_token_list = []
    for token in token_list:
        token = token_class(token)
        if token == '#':
            refined_token_list.append('<hashtag>')
            continue
        if token.is_url() == True:
            refined_token_list.append('<url>')
            continue
        if token.is_number() == True:
            refined_token_list.append('<number>')
            continue
        if token.is_user_mention() == True:
            refined_token_list.append('<user>')
            continue
        
        temp = token.is_emoji()
        if temp[0] == True:
            refined_token_list.append(temp[1])
            continue
        
        temp = token.is_repeated()
        if temp[0] == True:
            refined_token_list.append(temp[1].lower())
            refined_token_list.append('<repeatedletters>')
            if token.isupper() == True:
                refined_token_list.append('<allcaps>')
            continue
        
        if token.isupper() == True:
            refined_token_list.append(token.lower())
            refined_token_list.append('<allcaps>')
            continue
        refined_token_list.append(token.get_token().lower())
        
        
        
        
            