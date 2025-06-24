# import pandas as pd
# from tone_analyze import predict_tone
# from store_to_pg import store_record
# from datetime import datetime

# df = pd.read_csv('data/processed_articles.csv')

# for _, row in df.iterrows():
#     ts = datetime.fromisoformat(row['publishedAt'].replace('Z', '+00:00')) if 'Z' in row['publishedAt'] else datetime.utcnow()
#     store_record(row['cleaned_text'], predict_tone(row['cleaned_text']), row['source'], ts)


import pandas as pd
from tone_analyze import predict_tone
from store_to_pg import store_record
from datetime import datetime

print("📥 Loading processed articles...")
df = pd.read_csv("data/processed_articles.csv")

for _, row in df.iterrows():
    try:
        text = str(row["cleaned_text"]).strip()
        if not text:
            continue

        ts = datetime.fromisoformat(row["publishedAt"].replace("Z", "+00:00")) if "Z" in row["publishedAt"] else datetime.utcnow()

        tone_result = predict_tone(text)
        label = tone_result.get("label", "unknown")

        store_record(text, label, row["source"], ts)
    except Exception as e:
        print(f"❌ Error while processing row: {e}")
