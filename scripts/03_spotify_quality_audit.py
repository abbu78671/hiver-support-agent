"""
Phase 2 - Data Quality Audit for SpotifyCares
Only runs on SpotifyCares data.
All numbers are OBSERVED from actual data.
"""

import pandas as pd
import re

DATA_PATH = r"C:\Users\abdul\Downloads\archive extract\twcs\twcs.csv"
BRAND = 'SpotifyCares'

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ── LOAD ─────────────────────────────────────────────────────────
print("Loading full dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df['author_id'] = df['author_id'].astype(str).str.strip()
print(f"Total rows: {len(df):,}")

# ── SPOTIFY SUBSET ───────────────────────────────────────────────
spotify_brand = df[df['author_id'] == BRAND].copy()
print(f"SpotifyCares brand tweets: {len(spotify_brand):,}")

# ── SECTION 1: COMPLETE CONVERSATION PAIRS ───────────────────────
section("1. COMPLETE CONVERSATION PAIRS")
# A complete pair = customer tweet that Spotify replied to
# We find this by: Spotify reply has in_response_to_tweet_id
# That parent tweet should be a customer tweet (inbound=True)

spotify_replies = spotify_brand[
    spotify_brand['in_response_to_tweet_id'].notna()
].copy()
spotify_replies['parent_id'] = (
    spotify_replies['in_response_to_tweet_id'].astype(int)
)

# Get parent tweets
parent_ids = set(spotify_replies['parent_id'].tolist())
parents = df[df['tweet_id'].isin(parent_ids)].copy()
print(f"Spotify replies with a parent tweet ID: {len(spotify_replies):,}")
print(f"Parent tweets found in dataset: {len(parents):,}")
print(f"Parent tweets that are inbound (customer): "
      f"{parents[parents['inbound']==True].shape[0]:,}")
print(f"Parent tweets that are NOT inbound (brand→brand?): "
      f"{parents[parents['inbound']==False].shape[0]:,}")

# ── SECTION 2: DM DEFLECTION RATE ────────────────────────────────
section("2. DM DEFLECTION RATE")
dm_patterns = ['dm', 'direct message', 'private message', 
               'send us a', 'message us', 'reach out']
dm_pattern = '|'.join(dm_patterns)
dm_replies = spotify_brand[
    spotify_brand['text'].str.lower().str.contains(dm_pattern, na=False)
]
total_replies = len(spotify_brand)
dm_rate = len(dm_replies) / total_replies
print(f"Total Spotify replies: {total_replies:,}")
print(f"Replies containing DM/message deflection: {len(dm_replies):,}")
print(f"DM deflection rate: {dm_rate:.1%}")

# ── SECTION 3: REPLY LENGTH DISTRIBUTION ─────────────────────────
section("3. REPLY LENGTH DISTRIBUTION")
spotify_brand_copy = spotify_brand.copy()
spotify_brand_copy['text_len'] = spotify_brand_copy['text'].str.len()
desc = spotify_brand_copy['text_len'].describe(
    percentiles=[.10, .25, .50, .75, .90]
)
print(desc.to_string())

# ── SECTION 4: CONVERSATION LENGTH ───────────────────────────────
section("4. CONVERSATION THREAD LENGTH")
# Build simple thread chains
# Count how many turns involve Spotify in a conversation

# Get all tweet IDs involving Spotify
spotify_tweet_ids = set(spotify_brand['tweet_id'].tolist())

# For each Spotify reply, trace how deep the conversation is
thread_lengths = []
sample_threads = spotify_replies.head(500)  # sample for speed

for _, row in sample_threads.iterrows():
    depth = 1
    current_id = row['parent_id']
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        parent_row = df[df['tweet_id'] == current_id]
        if len(parent_row) == 0:
            break
        parent_row = parent_row.iloc[0]
        depth += 1
        if pd.isna(parent_row['in_response_to_tweet_id']):
            break
        current_id = int(parent_row['in_response_to_tweet_id'])
        if depth > 20:  # safety limit
            break
    thread_lengths.append(depth)

thread_series = pd.Series(thread_lengths)
print(f"Sample size: {len(thread_lengths)} threads")
print(f"Thread length distribution:")
print(thread_series.describe(percentiles=[.25, .50, .75, .90]).to_string())
print(f"\nThreads with 1 turn: {(thread_series==1).sum():,}")
print(f"Threads with 2 turns: {(thread_series==2).sum():,}")
print(f"Threads with 3+ turns: {(thread_series>=3).sum():,}")

# ── SECTION 5: NOISE IN CUSTOMER TWEETS ──────────────────────────
section("5. NOISE IN CUSTOMER TWEETS DIRECTED AT SPOTIFY")
cust_to_spotify = df[
    (df['inbound'] == True) &
    (df['text'].str.contains('@SpotifyCares', case=False, na=False))
].copy()
print(f"Customer tweets @SpotifyCares: {len(cust_to_spotify):,}")

cust_to_spotify['text_len'] = cust_to_spotify['text'].str.len()
print(f"\nCustomer tweet length:")
print(cust_to_spotify['text_len'].describe(
    percentiles=[.10, .25, .50, .75, .90]
).to_string())

# Very short messages (noise)
very_short = cust_to_spotify[cust_to_spotify['text_len'] < 30]
print(f"\nVery short (<30 chars): {len(very_short):,} "
      f"({len(very_short)/len(cust_to_spotify):.1%})")

# Sample very short messages
print("\nSample very short customer messages:")
for text in very_short['text'].head(5):
    print(f"  → {text}")

# ── SECTION 6: SAMPLE COMPLETE CONVERSATIONS ─────────────────────
section("6. SAMPLE COMPLETE CONVERSATIONS (customer→Spotify)")
# Find pairs where we have both customer tweet and Spotify reply
paired = spotify_replies.merge(
    parents[parents['inbound']==True][['tweet_id', 'text', 'author_id']],
    left_on='parent_id',
    right_on='tweet_id',
    suffixes=('_spotify', '_customer')
)
print(f"Complete customer→Spotify pairs found: {len(paired):,}")
print("\n5 sample complete pairs:")

sample_pairs = paired.sample(min(5, len(paired)), random_state=42)
for i, (_, row) in enumerate(sample_pairs.iterrows()):
    print(f"\n  Pair {i+1}:")
    print(f"  CUSTOMER: {str(row.get('text_customer', ''))[:150]}")
    print(f"  SPOTIFY:  {str(row.get('text_spotify', ''))[:150]}")

# ── SECTION 7: DUPLICATE CUSTOMER MESSAGES ───────────────────────
section("7. NEAR-DUPLICATE CUSTOMER MESSAGES")
# Strip @SpotifyCares mention and check for duplication
cust_to_spotify['clean_text'] = (
    cust_to_spotify['text']
    .str.lower()
    .str.replace(r'@\w+', '', regex=True)
    .str.strip()
)
dup_count = cust_to_spotify['clean_text'].duplicated().sum()
print(f"Exact duplicate customer messages (after cleaning): {dup_count:,}")
print(f"Duplicate rate: {dup_count/len(cust_to_spotify):.1%}")

section("AUDIT COMPLETE — paste full output")