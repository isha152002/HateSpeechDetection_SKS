import pandas as pd

def load_dictionary(filepath):
    df=pd.read_csv(filepath) #dataframe
    dictionary=dict(zip(df['ngram'],df['prophate']))
    return dictionary

def get_category(token,derogatory_dict):
    if token in derogatory_dict and derogatory_dict[token]>=0.5 :
        return "C1"
    else:
        return "C0"
        
        
