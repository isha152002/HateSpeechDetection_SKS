import re
import html


def clean_tweet(text):
    text=html.unescape(text) # convert HTML entities first
    text=text.lower() # lowercase
    text=re.sub(r'http\S+|www\S+', '', text) # remove URLs
    text=re.sub(r'@\w+', '', text) # remove mentions
    text=re.sub(r'#(\w+)', r'\1', text) # remove # keep word
    text=re.sub(r'(.)\1{2,}', r'\1', text) # normalise repeated chars
    text=re.sub(r'\s+', ' ', text).strip() # clean whitespace
    return text
