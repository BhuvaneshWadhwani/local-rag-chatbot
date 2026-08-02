from pathlib import Path
import cv2
import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


################### HowTo100M Video Preprocessing ###################

### Set paths
# Anchored to this script's own location (not the terminal's cwd), so it
# works the same whether you run it from Project/, the repo root, or an IDE
# with a different working directory configured.
base_dir = Path(__file__).resolve().parent

project_dir = base_dir

data_dir = project_dir / "data" / "howto100m_bundle"
video_dir = data_dir / "videos"

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

available_videos_path = data_dir / "available_videos.csv"
task_ids_path = data_dir / "task_ids.csv"

clip_index_path = output_dir / "howto100m_clip.index"
clip_metadata_path = output_dir / "howto100m_clip_metadata.csv"

print(f"CSV paths: {available_videos_path}, {task_ids_path}")
print(f"CSV exists: {available_videos_path.exists()}")
print(f"CSV exists: {task_ids_path.exists()}")


### Load HowTo100M dataset
task_ids = pd.read_csv(task_ids_path, sep='\t', header=None, names=["task_id", "task_name"])
available_videos = pd.read_csv(available_videos_path)

# Remove rows with invalid video_ids (e.g. "#NAME?")
available_videos = available_videos[
    available_videos["video_id"].astype(str).str.contains("#NAME?") == False
]

print(task_ids.head())
print(available_videos.head())
print(task_ids.columns)
print(available_videos.columns)
print("task_ids shape:", task_ids.shape)
print("available_videos shape:", available_videos.shape)

# Use only first 5 videos for testing.
#available_videos = available_videos.iloc[:5].copy() ########COMMENT OUT LATER#####

selected_videos = available_videos.merge(
    task_ids,
    on="task_id",
    how="left"
)

print("\nSelected videos:")
print(selected_videos)

### Find video file helper
def find_video_path(row, video_dir):
    video_id = str(row["video_id"])

    video_path = video_dir / f"{video_id}.mp4"

    if video_path.exists():
        return video_path

    return None

### CLIP generation: sliding windows
def generate_sliding_window_clips(video_path, clip_length_seconds=10):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or frame_count <= 0:
        print(f"Invalid video metadata: {video_path}")
        cap.release()
        return []

    duration = frame_count / fps

    clips = []
    start = 0.0

    while start < duration:
        end = min(start + clip_length_seconds, duration)

        clips.append({
            "video_path": str(video_path),
            "start_time": start,
            "end_time": end
        })

        start += clip_length_seconds

    cap.release()
    return clips

### Sample frames from one clip
def sample_frames_from_clip(video_path, start_time, end_time, num_frames=3):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps = np.linspace(start_time, end_time, num_frames + 2)[1:-1]
    frames = []

    for timestamp in timestamps:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if success:
            # OpenCV uses BGR, PIL/CLIP needs RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            frames.append(image)

    cap.release()
    return frames

### Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.to(device)
clip_model.eval()

### Encode frames into one clip embedding
def encode_clip_frames(frames):
    if len(frames) == 0:
        return None

    inputs = clip_processor(
        images=frames,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs)
        image_features = outputs.pooler_output
        image_features = clip_model.visual_projection(image_features)

    # Normalize frame embeddings
    image_features = image_features / image_features.norm(
        dim=-1,
        keepdim=True
    )

    # Average frame embeddings into one clip embedding
    clip_embedding = image_features.mean(dim=0)

    # Normalize final clip embedding
    clip_embedding = clip_embedding / clip_embedding.norm()

    return clip_embedding.cpu().numpy().astype("float32")


### Preprocess videos and build clip embeddings
clip_embeddings = []
clip_metadata = []

clip_id = 0

for _, row in selected_videos.iterrows():
    video_path = find_video_path(row, video_dir)

    if video_path is None:
        print("\nCould not find video file for row:")
        print(row)
        continue

    print(f"\nProcessing video: {video_path}")

    clips = generate_sliding_window_clips(
        video_path,
        clip_length_seconds=10
    )

    print(f"Generated {len(clips)} clips.")

    for clip in clips:
        frames = sample_frames_from_clip(
            clip["video_path"],
            clip["start_time"],
            clip["end_time"],
            num_frames=3
        )

        embedding = encode_clip_frames(frames)

        if embedding is None:
            continue

        clip_embeddings.append(embedding)

        clip_metadata.append({
            "clip_id": clip_id,
            "video_id": row["video_id"],
            "task_id": row["task_id"],
            "task_name": row["task_name"],
            "video_path": clip["video_path"],
            "start_time": clip["start_time"],
            "end_time": clip["end_time"]
        })

        clip_id += 1


clip_embeddings = np.array(clip_embeddings).astype("float32")
clip_metadata = pd.DataFrame(clip_metadata)

print("\nTotal clip embeddings:", clip_embeddings.shape)
print("\nClip metadata:")
print(clip_metadata.head())

# Check that we have clip embeddings
if len(clip_embeddings) == 0:
    raise RuntimeError(
        "No clip embeddings were created. Check selected_videos and video paths."
    )


### Build FAISS index
embedding_dim = clip_embeddings.shape[1]

clip_index = faiss.IndexFlatIP(embedding_dim)
clip_index.add(clip_embeddings)

print("\nVectors in clip index:", clip_index.ntotal)

### Save FAISS index and metadata
faiss.write_index(clip_index, str(clip_index_path))
clip_metadata.to_csv(clip_metadata_path, index=False)

print("\nSaved clip FAISS index:")
print(clip_index_path)

print("\nSaved clip metadata:")
print(clip_metadata_path)


### Text to clip retrieval text
query = "how to cook food"

text_inputs = clip_processor(
    text=[query],
    return_tensors="pt",
    padding=True
).to(device)

with torch.no_grad():
    outputs = clip_model.text_model(**text_inputs)
    text_embedding = outputs.pooler_output
    text_embedding = clip_model.text_projection(text_embedding)

text_embedding = text_embedding / text_embedding.norm(
    dim=-1,
    keepdim=True
)

text_embedding = text_embedding.cpu().numpy().astype("float32")

top_k = 5
scores, positions = clip_index.search(text_embedding, top_k)

print("Text-to-clip retrieval results")
print("Query:", query)

for score, position in zip(scores[0], positions[0]):
    result = clip_metadata.iloc[position]

    print("\nScore:", float(score))
    print("Clip ID:", result["clip_id"])
    print("Video:", result["video_path"])
    print("Start:", result["start_time"])
    print("End:", result["end_time"])

print("\nDone.")




















