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
        "hhhhheeey" is converted into "<hey>" and "<repeat>"
    
    all tokens converted to lower case
"""
import nltk
import re
from itertools import groupby
import emoji


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
        temp = nltk.TweetTokenizer(strip_handles=True)
        result = temp.tokenize(self.token)
        if result == []:
            return True
        else:
            return False

    def is_emoji(self):
        """
        checks if token is an emoji
        ***
        for now the function will just return [True, '<emoji>'] if the token is
        an emoji, [False, None] if it is not
        ***
        TO BE IMPLEMENTED:
        if it is, it returns True and the type of emoji that the token is
        else, returns False and None
        """
        if self.token in emoji.UNICODE_EMOJI:
            return [True, '<neutralface>']
        else:
            return [False, None]

    def is_repeated(self):
        """
        checks if token has repeated letters
        if it does, it returns True and the word without repeated letters
        if it does not, it returns False and the original word
        """
        reduced_token = ''.join(''.join(s)[:2] for _, s in groupby(self.token))
        if reduced_token == self.token:
            return [False, self.token]
        else:
            return [True, reduced_token]

    def get_token(self):
        return self.token


def tokenizer1(tweet):
    """
    splits tweet into initial list of tokens
    """
    tknzr = nltk.TweetTokenizer(preserve_case=True, strip_handles=False)
    token_list = tknzr.tokenize(tweet)
    return token_list


def refine_token(token_list):
    """
    refines list of tokens to be in line with rules at top of this file
    """

    refined_token_list = []
    for token in token_list:
        token = token_class(token)
        if token.is_url():
            refined_token_list.append('<url>')
            continue
        if token.is_number():
            refined_token_list.append('<number>')
            continue
        if token.is_user_mention():
            refined_token_list.append('<user>')
            continue
        temp = token.is_emoji()
        if temp[0]:
            refined_token_list.append(temp[1])
            continue
        temp = token.is_repeated()
        if temp[0]:
            refined_token_list.append(temp[1].lower())
            refined_token_list.append('<repeat>')
            if token.get_token().isupper():
                refined_token_list.append('<allcaps>')
            continue
        if token.get_token().isupper():
            refined_token_list.append(token.get_token().lower())
            refined_token_list.append('<allcaps>')
            continue
        # check for 1 hashtag at start of token
        if token.get_token()[0] == '#':
            refined_token_list.append('<hashtag>')
            refined_token_list.append(token.get_token()[1:])
            continue
        # check for 2 hashtags at start of token (there can't be more than 2
        # since we already dealt with such cases with is_repeated)
        if token.get_token()[0] == '#' and token.get_token()[1] == '#':
            refined_token_list.append('<hashtag>')
            refined_token_list.append(token.get_token()[2:])
            continue
        refined_token_list.append(token.get_token().lower())
    return refined_token_list
