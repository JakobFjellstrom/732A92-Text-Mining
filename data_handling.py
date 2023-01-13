#!/usr/bin/env python3

import snscrape.modules.twitter as sntwitter
import itertools
import pandas as pd
import regex as re
import emoji


# A function that will return a list of a specified number of raw tweets from a specified user. 
# This will be returned as a lsit of strings. 

def web_scrape_tweets(number_tweets = 200, user = 'strandhall'):
    
    df = pd.DataFrame(itertools.islice(sntwitter.TwitterTweetScraper(
       'from:' + user).get_items(),number_tweets))
        
    return list(df.content)
    

    
    
def pre_process(data):
    
    # Some replacements
    data = [w.replace('”', '\"') for w in data]
    data = [w.replace('“', '\"') for w in data]
    data = [w.replace('&', 'och') for w in data]
    
   # Remove all emojis 
    emoji_free = [emoji.get_emoji_regexp().sub('', s) for s in data]
    
    # Remove all URL strings, "\n" that's due to line breaks in the tweets, and also removing all user-mentions, i.e. @user
    re_data = [re.sub(r'(@|http?)\S+|•|\n|\xad|–|—|→|ー|’|´|\||_|/|\'|ø|ツ|…|Я|§ |¯|\\', ' ', string) for string in emoji_free]
    
    # Also removing unicode type of words, such as \u2019 
    r_data = [re.sub(r"[\u2019|\u200b|\u2066|\u2069|à|é|á|ó ]+", " ", string) for string in re_data]
    
    # Finally removing all strings that now are empty
    clean_data = [s for s in r_data if s.strip()]
    
    # Also removing all left over white spaces in the strings
    cleaned_data = [" ".join(s.split()) for s in clean_data]
    
    # Only keep the tweets that are longer than 50 characters (arbitrary)
    final_list = [lista for lista in cleaned_data if len(lista) > 50]
    print("The number of tweets left are:", len(final_list))
    
    return final_list
    

