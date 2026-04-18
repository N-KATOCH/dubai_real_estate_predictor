import pandas as pd
import re

def clean_text_data(text):
    """Refined NLP cleaning for Dubai Real Estate descriptions."""
    if pd.isna(text): return ""
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def create_nlp_flags(df, text_col='clean_text'):
    """Creating binary flags for Power BI Slicers."""
    keywords = ['pool', 'view', 'upgraded', 'luxury', 'beach', 'furnished']
    for word in keywords:
        # Check if word exists in the text
        df[f'has_{word}'] = df[text_col].str.contains(word, na=False).astype(int)
    return df

def prepare_silver_layer(df):
    """The master function for data cleaning."""
    # We assume 'description' is the column name from the raw data
    df['clean_text'] = df['description'].apply(clean_text_data)
    df = create_nlp_flags(df)
    return df
