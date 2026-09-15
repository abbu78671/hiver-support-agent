"""
Phase 1b - Fix missing sections: brand distribution + text samples
"""

import pandas as pd

DATA_PATH = r"C:\Users\abdul\Downloads\archive extract\twcs\twcs.csv"
NROWS = 500_000  # larger sample for better brand counts

print("Loading data...")
df = pd.read_csv(DATA_PATH, nrows=NROWS, low_memory=False)
print(f"Rows loaded: {len(df):,}")

# ── SECTION A: TEXT SAMPLES ───────────────────────────────────────
print("\n" + "="*60)
print("  A. SAMPLE CUSTOMER TWEETS (inbound=True)")
print("="*60)
customer_tweets = df[df['inbound'] == True]['text'].dropna()
for i, text in enumerate(customer_tweets.sample(5, random_state=42)):
    print(f"\n  [{i+1}] {text[:200]}")

print("\n" + "="*60)
print("  B. SAMPLE BRAND TWEETS (inbound=False)")
print("="*60)
brand_tweets = df[df['inbound'] == False]['text'].dropna()
for i, text in enumerate(brand_tweets.sample(5, random_state=42)):
    print(f"\n  [{i+1}] {text[:200]}")

# ── SECTION B: BRAND DISTRIBUTION ────────────────────────────────
print("\n" + "="*60)
print("  C. TOP 50 BRANDS (author_id where inbound=False)")
print("="*60)
brand_counts = (
    df[df['inbound'] == False]['author_id']
    .value_counts()
    .head(50)
)
print(brand_counts.to_string())

# ── SECTION C: CUSTOMER ID SAMPLES ───────────────────────────────
print("\n" + "="*60)
print("  D. SAMPLE CUSTOMER author_ids (inbound=True)")
print("="*60)
customer_ids = df[df['inbound'] == True]['author_id'].value_counts().head(20)
print(customer_ids.to_string())

# ── SECTION D: CONVERSATION STRUCTURE ────────────────────────────
print("\n" + "="*60)
print("  E. CONVERSATION STRUCTURE CHECK")
print("="*60)
print(f"Tweets WITH in_response_to_tweet_id (have a parent): "
      f"{df['in_response_to_tweet_id'].notna().sum():,}")
print(f"Tweets WITHOUT in_response_to_tweet_id (conversation starters): "
      f"{df['in_response_to_tweet_id'].isna().sum():,}")
print(f"Tweets WITH response_tweet_id (got a reply): "
      f"{df['response_tweet_id'].notna().sum():,}")
print(f"Tweets WITHOUT response_tweet_id (no reply received): "
      f"{df['response_tweet_id'].isna().sum():,}")

# ── SECTION E: SAMPLE CONVERSATION THREAD ────────────────────────
print("\n" + "="*60)
print("  F. ONE SAMPLE CONVERSATION THREAD")
print("="*60)
# Find a brand tweet and trace back its conversation
brand_sample = df[
    (df['inbound'] == False) & 
    (df['in_response_to_tweet_id'].notna())
].head(1)

if len(brand_sample) > 0:
    row = brand_sample.iloc[0]
    print(f"Brand tweet:")
    print(f"  tweet_id: {row['tweet_id']}")
    print(f"  author_id: {row['author_id']}")
    print(f"  text: {row['text'][:200]}")
    print(f"  in_response_to: {row['in_response_to_tweet_id']}")
    
    parent_id = int(row['in_response_to_tweet_id'])
    parent = df[df['tweet_id'] == parent_id]
    if len(parent) > 0:
        p = parent.iloc[0]
        print(f"\nParent tweet (customer):")
        print(f"  tweet_id: {p['tweet_id']}")
        print(f"  author_id: {p['author_id']}")
        print(f"  inbound: {p['inbound']}")
        print(f"  text: {p['text'][:200]}")

print("\n" + "="*60)
print("  DONE — paste all output above")
print("="*60)