from email.mime import base
import pandas as pd
import json

def extractTextOnly(path, limit=100):
    """extracts text from loose json string in preparation for topic modelling

    Args:
        path (string): path to json file

    Returns:
        pd.DataFrame: dataframe containing each review as a row
    """
    ds = [json.loads(line) for line in open(path, 'r', encoding='utf-8')]

    df = pd.DataFrame.from_dict(ds)
    [col for col in df.columns]

    reviewText = df['reviewText'].head(limit).astype('U')
    return reviewText

if __name__ == '__main__':
    extractTextOnly()