"""
Phase 3+4 - Reconstruct SpotifyCares conversations
and explore intent patterns for taxonomy design.
All findings are OBSERVED from actual data.
"""

import pandas as pd
import json
import re
from collections import defaultdict

DATA_PATH = r"C:\Users\abdul\Downloads\archive extract\twcs\twcs.csv"
BRAND = 'SpotifyCares'
OUTPUT_PATH = r"data\processed\spotify_conversations.jsonl"
SAMPLE_OUTPUT = r"data\samples\spotify_sample_100.jsonl"

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def clean_text(text, brand=BRAND):
    """Remove @mentions at start, strip whitespace."""
    if not isinstance(text, str):
        return ""
    # Remove @BrandName and @CustomerID mentions
    text = re.sub(r'^(@\w+\s*)+', '', text).strip()
    return text

def is_dm_deflection(text):
    """Detect if Spotify reply is deflecting to DM."""
    if not isinstance(text, str):
        return False
    patterns = ['dm', 'direct message', 'private message',
                'send us a', 'message us', 'reach out',
                'chat there', 'carry on', 'backstage']
    text_lower = text.lower()
    return any(p in text_lower for p in patterns)

# ── LOAD ─────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df['author_id'] = df['author_id'].astype(str).str.strip()
df['tweet_id'] = df['tweet_id'].astype(int)
df['in_response_to_tweet_id'] = pd.to_numeric(
    df['in_response_to_tweet_id'], errors='coerce'
)
print(f"Rows: {len(df):,}")

# Build lookup for speed
tweet_lookup = df.set_index('tweet_id').to_dict('index')

# ── FIND SPOTIFY REPLIES ─────────────────────────────────────────
section("1. BUILDING CONVERSATION PAIRS")
spotify_replies = df[
    (df['author_id'] == BRAND) &
    (df['in_response_to_tweet_id'].notna())
].copy()
print(f"Spotify replies with parent: {len(spotify_replies):,}")

# ── BUILD CONVERSATION OBJECTS ───────────────────────────────────
conversations = []
skipped = 0

for _, reply_row in spotify_replies.iterrows():
    parent_id = int(reply_row['in_response_to_tweet_id'])
    
    # Get parent (customer) tweet
    if parent_id not in tweet_lookup:
        skipped += 1
        continue
    
    parent = tweet_lookup[parent_id]
    
    # Must be a customer tweet
    if parent.get('inbound') != True:
        skipped += 1
        continue
    
    customer_text = clean_text(parent.get('text', ''))
    spotify_text = reply_row['text']
    
    # Skip very short customer messages (follow-up noise)
    if len(customer_text) < 20:
        skipped += 1
        continue
    
    conv = {
        'conversation_id': int(reply_row['tweet_id']),
        'customer_tweet_id': parent_id,
        'spotify_tweet_id': int(reply_row['tweet_id']),
        'customer_message': customer_text,
        'spotify_reply': spotify_text,
        'spotify_reply_clean': clean_text(spotify_text),
        'is_dm_deflection': is_dm_deflection(spotify_text),
        'customer_message_len': len(customer_text),
        'spotify_reply_len': len(spotify_text),
        'created_at': reply_row.get('created_at', ''),
    }
    conversations.append(conv)

print(f"Conversations built: {len(conversations):,}")
print(f"Skipped (missing parent / too short): {skipped:,}")

conv_df = pd.DataFrame(conversations)

# ── SECTION 2: DEFLECTION BREAKDOWN ─────────────────────────────
section("2. DM DEFLECTION IN BUILT CONVERSATIONS")
dm_count = conv_df['is_dm_deflection'].sum()
total = len(conv_df)
print(f"Total conversations: {total:,}")
print(f"DM deflection (will be ESCALATE label): {dm_count:,} ({dm_count/total:.1%})")
print(f"Substantive replies (AUTO-HANDLE candidates): "
      f"{total-dm_count:,} ({(total-dm_count)/total:.1%})")

# ── SECTION 3: SAMPLE SUBSTANTIVE CONVERSATIONS ─────────────────
section("3. SAMPLE SUBSTANTIVE CONVERSATIONS (non-deflection)")
substantive = conv_df[conv_df['is_dm_deflection'] == False]
print(f"\nShowing 15 substantive customer messages for taxonomy exploration:")
samples = substantive.sample(min(15, len(substantive)), random_state=42)
for i, (_, row) in enumerate(samples.iterrows()):
    print(f"\n[{i+1}] CUSTOMER: {row['customer_message'][:180]}")
    print(f"     SPOTIFY:  {row['spotify_reply_clean'][:180]}")

# ── SECTION 4: SAMPLE DM DEFLECTION CONVERSATIONS ───────────────
section("4. SAMPLE DM DEFLECTION CONVERSATIONS (will be ESCALATE)")
deflections = conv_df[conv_df['is_dm_deflection'] == True]
print(f"\nShowing 10 deflected conversations:")
def_samples = deflections.sample(min(10, len(deflections)), random_state=42)
for i, (_, row) in enumerate(def_samples.iterrows()):
    print(f"\n[{i+1}] CUSTOMER: {row['customer_message'][:180]}")
    print(f"     SPOTIFY:  {row['spotify_reply_clean'][:180]}")

# ── SECTION 5: SAVE OUTPUTS ──────────────────────────────────────
section("5. SAVING OUTPUTS")

import os
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/samples', exist_ok=True)

# Save full conversation set
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')
print(f"Full conversation set saved: {OUTPUT_PATH}")
print(f"Total conversations: {len(conversations):,}")

# Save small sample (100 random) for quick testing
import random
random.seed(42)
sample_100 = random.sample(conversations, min(100, len(conversations)))
with open(SAMPLE_OUTPUT, 'w', encoding='utf-8') as f:
    for conv in sample_100:
        f.write(json.dumps(conv) + '\n')
print(f"100-conversation sample saved: {SAMPLE_OUTPUT}")

# ── SECTION 6: KEYWORD FREQUENCY IN CUSTOMER MESSAGES ───────────
section("6. KEYWORD FREQUENCY (helps identify intents)")
# Look at common words in customer messages to spot patterns
all_customer_text = ' '.join(conv_df['customer_message'].tolist()).lower()
# Simple word frequency
words = re.findall(r'\b[a-z]{4,}\b', all_customer_text)
word_freq = pd.Series(words).value_counts()

# Filter out generic stop words
stop = {'that', 'this', 'with', 'have', 'your', 'just', 'been',
        'when', 'from', 'they', 'what', 'will', 'more', 'also',
        'some', 'there', 'know', 'dont', 'cant', 'about', 'like',
        'still', 'even', 'then', 'them', 'than', 'into', 'were',
        'would', 'could', 'their', 'does', 'using', 'please',
        'help', 'spotify', 'spotifycares'}
filtered = word_freq[~word_freq.index.isin(stop)]
print("\nTop 40 words in customer messages (after removing stop words):")
print(filtered.head(40).to_string())

section("DONE — paste full output")