import pandas as pd
import numpy as np
import sqlite3

from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

################### WikiHow Text Preprocessing ###################

### Set paths
# Anchored to this script's own location (not the terminal's cwd), so it
# works the same whether you run it from Project/, the repo root, or an IDE
# with a different working directory configured.
base_dir = Path(__file__).resolve().parent

project_dir = base_dir

data_dir = project_dir / "data" / "wikihow-cleaned"

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = data_dir / "wikihow-cleaned.csv"

db_path = output_dir / "wikihow.db"

title_index_path = output_dir / "wikihow_title.index"
title_summary_index_path = output_dir / "wikihow_title_summary.index"

title_mapping_path = output_dir / "title_index_to_article_id.npy"
title_summary_mapping_path = output_dir / "title_summary_index_to_article_id.npy"

print(f"CSV path: {csv_path}")
print(f"CSV exists: {csv_path.exists()}")


### Load WikiHow dataset
wikihow = pd.read_csv(csv_path)

print("Loaded WikiHow dataframe")
print("Original shape:", wikihow.shape)

# Use only first 100 rows for testing.
#wikihow = wikihow.iloc[:100].copy() ########COMMENT OUT LATER#####

# Add article_id for SQLite and FAISS mapping
wikihow.insert(0, "article_id", range(len(wikihow)))

print("Working shape:", wikihow.shape)
print(wikihow.head())

# Create SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS wikihow (
    article_id INTEGER PRIMARY KEY,
    title TEXT,
    summary TEXT,
    text TEXT
)
""")

# Delete old data so the script can be rerun safely
cursor.execute("DELETE FROM wikihow")

wikihow.to_sql("wikihow", conn, if_exists="append", index=False)

conn.commit()

count = cursor.execute("SELECT COUNT(*) FROM wikihow").fetchone()[0]
print(f"\nInserted {count} rows into SQLite database.")
print(f"Database saved at: {db_path}")


### Load Sentence-BERT model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Prepare text for two embedding sets
titles = wikihow["title"].fillna("").tolist()

title_summary = (wikihow["title"].fillna("") + " " + wikihow["summary"].fillna("")).tolist()

article_ids = wikihow["article_id"].to_numpy()


# Compute separate embeddings
title_embeddings = model.encode(titles, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

title_summary_embeddings = model.encode( title_summary, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

print("\nTitle embeddings shape:", title_embeddings.shape)
print("Title + summary embeddings shape:", title_summary_embeddings.shape)


### Build FAISS indices
embedding_dim = title_embeddings.shape[1]

# IndexFlatIP uses inner product.
# Since embeddings are normalized, this works like cosine similarity.
title_index = faiss.IndexFlatIP(embedding_dim)
title_summary_index = faiss.IndexFlatIP(embedding_dim)

title_index.add(title_embeddings)
title_summary_index.add(title_summary_embeddings)

print("\nVectors in title index:", title_index.ntotal)
print("Vectors in title + summary index:", title_summary_index.ntotal)


### Save FAISS indices and mappings
faiss.write_index(title_index, str(title_index_path))
faiss.write_index(title_summary_index, str(title_summary_index_path))

np.save(title_mapping_path, article_ids)
np.save(title_summary_mapping_path, article_ids)

print("\nSaved FAISS indices and article ID mappings.")


### Query-based semantic retrieval test
query = "how to make pasta"

query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

top_k = 5


### Test title-only index
scores, positions = title_index.search(query_embedding, top_k)

print("\nResults using TITLE index")

for score, position in zip(scores[0], positions[0]):
    article_id = int(article_ids[position])

    row = cursor.execute(
        """
        SELECT article_id, title, summary
        FROM wikihow
        WHERE article_id = ?
        """,
        (article_id,)
    ).fetchone()

    print("\nScore:", float(score))
    print("Article ID:", row[0])
    print("Title:", row[1])
    print("Summary:", row[2])


### Test title + summary index
scores, positions = title_summary_index.search(query_embedding, top_k)

print("\nResults using TITLE + SUMMARY index")

for score, position in zip(scores[0], positions[0]):
    article_id = int(article_ids[position])

    row = cursor.execute(
        """
        SELECT article_id, title, summary
        FROM wikihow
        WHERE article_id = ?
        """,
        (article_id,)
    ).fetchone()

    print("\nScore:", float(score))
    print("Article ID:", row[0])
    print("Title:", row[1])
    print("Summary:", row[2])


# Close database connection
conn.close()
print("\nDone.")










