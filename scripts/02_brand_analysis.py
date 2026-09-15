"""
Phase 2 - Brand candidate analysis
Measures usable conversations per brand before selection.
"""

import pandas as pd
import re

DATA_PATH = r"C:\Users\abdul\Downloads\archive extract\twcs\twcs.csv"

CANDIDATE_BRANDS = [
    'SpotifyCares',
    'AppleSupport', 
    'Delta',
    'AmericanAir',
    'Uber_Support',
    'XboxSupport',
    'TMobileHelp',
]

def is_english(text):
    """
    Rough English detector.
    Counts ASCII characters vs total.
    Not perfect but fast and good enough for filtering.
    """
    if not isinstance(text, str):
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return (ascii_chars / max(len(text), 1)) > 0.85

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

print("Loading full dataset (this will take 2-3 minutes)...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Total rows loaded: {len(df):,}")

# Strip whitespace from author_id
df['author_id'] = df['author_id'].astype(str).str.strip()

for brand in CANDIDATE_BRANDS:
    section(f"BRAND: {brand}")
    
    # All brand tweets
    brand_tweets = df[df['author_id'] == brand]
    print(f"Total brand tweets: {len(brand_tweets):,}")
    
    # All customer tweets directed at this brand
    # Customer tweets mention the brand in text
    pattern = f'@{brand}'
    customer_tweets = df[
        (df['inbound'] == True) & 
        (df['text'].str.contains(pattern, case=False, na=False))
    ]
    print(f"Customer tweets mentioning brand: {len(customer_tweets):,}")
    
    # English rate in brand tweets
    brand_english = brand_tweets['text'].apply(is_english).mean()
    print(f"English rate (brand tweets): {brand_english:.1%}")
    
    # English rate in customer tweets
    if len(customer_tweets) > 0:
        cust_english = customer_tweets['text'].apply(is_english).mean()
        print(f"English rate (customer tweets): {cust_english:.1%}")
    
    # Average brand tweet length
    brand_tweets_len = brand_tweets['text'].str.len()
    print(f"Brand tweet length — mean: {brand_tweets_len.mean():.0f}, "
          f"median: {brand_tweets_len.median():.0f}")
    
    # How many brand tweets are responses (have a parent)
    has_parent = brand_tweets['in_response_to_tweet_id'].notna().sum()
    print(f"Brand tweets with parent (actual responses): {has_parent:,}")
    
    # Template/duplicate detection
    # Check what % of brand tweets are near-identical
    total = len(brand_tweets)
    unique_texts = brand_tweets['text'].nunique()
    dup_rate = 1 - (unique_texts / max(total, 1))
    print(f"Duplicate brand tweet rate: {dup_rate:.1%} "
          f"({total - unique_texts:,} duplicates)")
    
    # Sample 3 brand replies
    sample_replies = brand_tweets[
        brand_tweets['in_response_to_tweet_id'].notna()
    ]['text'].dropna().sample(min(3, has_parent), random_state=42)
    print(f"\nSample brand replies:")
    for i, reply in enumerate(sample_replies):
        print(f"  [{i+1}] {reply[:150]}")

section("COMPARISON SUMMARY")
print(f"{'Brand':<20} {'BrandTweets':>12} {'DupRate':>10} {'EngRate':>10}")
print("-" * 55)
for brand in CANDIDATE_BRANDS:
    bt = df[df['author_id'] == brand]
    total = len(bt)
    if total == 0:
        continue
    unique = bt['text'].nunique()
    dup = 1 - (unique / total)
    eng = bt['text'].apply(is_english).mean()
    print(f"{brand:<20} {total:>12,} {dup:>10.1%} {eng:>10.1%}")

print("\nDONE — paste full output")