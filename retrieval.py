from pathlib import Path
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel

class ArticleRetriever:
    """
    Semantic retriever for WikiHow articles.

    Loads the preprocessing artifacts produced by `WikiHow_Preprocessing.py`:
    - SQLite database with article metadata
    - FAISS index for titles
    - FAISS index for title + summary
    - mappings from FAISS positions to article IDs
    """

    def __init__(
        self,
        project_dir=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        title_weight=0.4,
        title_summary_weight=0.6,
    ):
        # Use this file's own directory by default, not the terminal's cwd
        # (so it works the same regardless of where python was launched from)
        if project_dir is None:
            base_dir = Path(__file__).resolve().parent
            self.project_dir = base_dir
        else:
            self.project_dir = Path(project_dir)

        self.output_dir = self.project_dir / "outputs"

        # Paths created by WikiHow_Preprocessing.py
        self.db_path = self.output_dir / "wikihow.db"
        self.title_index_path = self.output_dir / "wikihow_title.index"
        self.title_summary_index_path = self.output_dir / "wikihow_title_summary.index"
        self.title_mapping_path = self.output_dir / "title_index_to_article_id.npy"
        self.title_summary_mapping_path = self.output_dir / "title_summary_index_to_article_id.npy"

        self.title_weight = title_weight
        self.title_summary_weight = title_summary_weight

        self._check_files()

        # Load sentence embedding model
        self.model = SentenceTransformer(model_name)

        # Load FAISS indices
        self.title_index = faiss.read_index(str(self.title_index_path))
        self.title_summary_index = faiss.read_index(str(self.title_summary_index_path))

        # Load mappings: FAISS position -> article_id
        self.title_mapping = np.load(self.title_mapping_path)
        self.title_summary_mapping = np.load(self.title_summary_mapping_path)

        # Connect to SQLite database
        # check_same_thread=False: this connection is created once at
        # startup (main thread), but Gradio runs each chat turn in its own
        # worker thread. We only ever read from it here, so sharing it
        # across threads is safe.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def _check_files(self):
        """
        Check whether all required preprocessing output files exist.
        """
        required_files = [
            self.db_path,
            self.title_index_path,
            self.title_summary_index_path,
            self.title_mapping_path,
            self.title_summary_mapping_path,
        ]

        missing_files = [path for path in required_files if not path.exists()]

        if missing_files:
            missing_text = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "Some required preprocessing files are missing:\n"
                f"{missing_text}\n\n"
                "Run WikiHow_Preprocessing.py first."
            )

    def _fetch_article(self, article_id):
        """
        Retrieve article metadata from SQLite using article_id.
        """
        cursor = self.conn.cursor()

        row = cursor.execute(
            """
            SELECT article_id, title, summary
            FROM wikihow
            WHERE article_id = ?
            """,
            (int(article_id),),
        ).fetchone()

        if row is None:
            return None

        return {
            "article_id": row[0],
            "title": row[1],
            "summary": row[2],
        }

    def search(self, query, top_k=5, semantic_threshold=0.15):
        """
        Search for relevant WikiHow articles.

        The query is encoded once, then searched against:
        1. title-only FAISS index
        2. title + summary FAISS index

        Results are combined using configurable weights.
        """

        # Encode query with the same MiniLM model used in preprocessing
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        # Search both indices
        title_scores, title_positions = self.title_index.search(query_embedding, top_k)
        summary_scores, summary_positions = self.title_summary_index.search(query_embedding, top_k)

        combined_results = {}

        # Add title-only results
        for score, position in zip(title_scores[0], title_positions[0]):
            if position == -1:
                continue

            article_id = int(self.title_mapping[position])

            if article_id not in combined_results:
                combined_results[article_id] = {
                    "article_id": article_id,
                    "score": 0.0,
                    "title_score": None,
                    "title_summary_score": None,
                }

            combined_results[article_id]["score"] += float(score) * self.title_weight
            combined_results[article_id]["title_score"] = float(score)

        # Add title + summary results
        for score, position in zip(summary_scores[0], summary_positions[0]):
            if position == -1:
                continue

            article_id = int(self.title_summary_mapping[position])

            if article_id not in combined_results:
                combined_results[article_id] = {
                    "article_id": article_id,
                    "score": 0.0,
                    "title_score": None,
                    "title_summary_score": None,
                }

            combined_results[article_id]["score"] += float(score) * self.title_summary_weight
            combined_results[article_id]["title_summary_score"] = float(score)

        # Sort by combined weighted score
        ranked_results = sorted(
            combined_results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # Keep only meaningful semantic results
        ranked_results = [
            result for result in ranked_results
            if result["score"] >= semantic_threshold
        ]

        # Optional keyword fallback
        if len(ranked_results) == 0:
            return self.keyword_search(query, top_k=top_k)

        # Add metadata from SQLite
        final_results = []

        for result in ranked_results[:top_k]:
            article = self._fetch_article(result["article_id"])

            if article is None:
                continue

            article.update({
                "score": result["score"],
                "title_score": result["title_score"],
                "title_summary_score": result["title_summary_score"],
                "retrieval_type": "semantic",
            })

            final_results.append(article)

        return final_results

    def keyword_search(self, query, top_k=5):
        """
        Simple SQL fallback if semantic retrieval finds no meaningful results.
        """
        cursor = self.conn.cursor()

        keyword = f"%{query}%"

        rows = cursor.execute(
            """
            SELECT article_id, title, summary
            FROM wikihow
            WHERE title LIKE ? OR summary LIKE ?
            LIMIT ?
            """,
            (keyword, keyword, top_k),
        ).fetchall()

        results = []

        for row in rows:
            results.append({
                "article_id": row[0],
                "title": row[1],
                "summary": row[2],
                "score": None,
                "title_score": None,
                "title_summary_score": None,
                "retrieval_type": "keyword",
            })

        return results

    def format_context(self, articles):
        """
        Format retrieved articles as context snippets for the RAG chatbot.
        """

        if not articles:
            return "No relevant WikiHow articles were retrieved."

        context_parts = []

        for i, article in enumerate(articles, start=1):
            score = article["score"]

            if score is not None:
                score_text = f"{score:.4f}"
            else:
                score_text = "N/A"

            snippet = (
                f"[Article {i}]\n"
                f"Title: {article['title']}\n"
                f"Summary: {article['summary']}\n"
                f"Retrieval type: {article['retrieval_type']}\n"
                f"Similarity score: {score_text}"
            )

            context_parts.append(snippet)
        return "\n\n".join(context_parts)
    
    def retrieve_context(self, query, top_k=5):
        articles = self.search(query, top_k=top_k)
        context = self.format_context(articles)
        return context, articles

    def close(self):
        """
        Close SQLite connection.
        """
        self.conn.close()

class VideoRetriever:
    """
    CLIP-based retriever for HowTo100M video clips.

    Loads the preprocessing artifacts produced by `HowTo100M_Preprocessing.py`:
    - FAISS index with video clip embeddings
    - CSV file with clip metadata
    """

    def __init__(
        self,
        project_dir=None,
        model_name="openai/clip-vit-base-patch32",
    ):
        if project_dir is None:
            self.project_dir = Path(__file__).resolve().parent
        else:
            self.project_dir = Path(project_dir)

        self.output_dir = self.project_dir / "outputs"

        self.clip_index_path = self.output_dir / "howto100m_clip.index"
        self.clip_metadata_path = self.output_dir / "howto100m_clip_metadata.csv"

        self._check_files()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading CLIP model for video retrieval...")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.clip_index = faiss.read_index(str(self.clip_index_path))
        self.clip_metadata = pd.read_csv(self.clip_metadata_path)


    def _check_files(self):
        required_files = [self.clip_index_path, self.clip_metadata_path]

        missing_files = [path for path in required_files if not path.exists()]

        if missing_files:
            missing_text = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                "Some required video preprocessing files are missing:\n"
                f"{missing_text}\n\n"
                "Run HowTo100M_Preprocessing.py first."
            )
        
    def _encode_query(self, query):
        """
        Encode a text query into a CLIP text embedding.
        """

        text_inputs = self.clip_processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model.text_model(**text_inputs)
            text_embedding = outputs.pooler_output
            text_embedding = self.clip_model.text_projection(text_embedding)

        # Normalize embedding
        text_embedding = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )

        return text_embedding.cpu().numpy().astype("float32")
    
    def search(self, query, top_k=5):
        """
        Retrieve the most relevant video clips for a text query.
        """

        #Encode query into CLIP embedding
        query_embedding = self._encode_query(query)

        # Search FAISS index
        scores, positions = self.clip_index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, position in zip(scores[0], positions[0]):
            if position == -1:
                continue

            clip = self.clip_metadata.iloc[position]

            results.append({
                "clip_id": int(clip["clip_id"]),
                "video_path": clip["video_path"],
                "start_time": float(clip["start_time"]),
                "end_time": float(clip["end_time"]),
                "score": float(score)
        })

        return results
    
    def format_results(self, videos):
        """
        Format retrieved video clips for readable terminal output.
        """

        if not videos:
            return "No relevant video clips were retrieved."

        formatted_parts = []

        for i, video in enumerate(videos, start=1):
            snippet = (
                f"[Video Clip {i}]\n"
                f"Clip ID: {video['clip_id']}\n"
                f"Video file: {Path(video['video_path']).name}\n"
                f"Time: {video['start_time']:.1f}s - {video['end_time']:.1f}s\n"
                f"Similarity score: {video['score']:.4f}"
            )
            formatted_parts.append(snippet)
        return "\n\n".join(formatted_parts)

if __name__ == "__main__":
    retriever = ArticleRetriever()

    query = "how to make pasta"
    results = retriever.search(query, top_k=5)

    print("\nRetrieved articles:")
    for article in results:
        print("\nTitle:", article["title"])
        print("Summary:", article["summary"])
        print("Score:", article["score"])
        print("Retrieval type:", article["retrieval_type"])

    print("\nFormatted context:")
    print(retriever.format_context(results))

    video_retriever = VideoRetriever()
    print("\nVideo retrieval results:\n")
    
    videos = video_retriever.search("how to make pasta", top_k=5)

    print(video_retriever.format_results(videos))

    retriever.close()
