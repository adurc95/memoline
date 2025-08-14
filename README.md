# Memoline — Album Graph Creator

Process large photo albums → describe photos, detect faces, compute similarities, and build a **centrality graph** of the most “important” images.

- **Tech**: Ollama (LLava) for image captions + embeddings, DeepFace (ArcFace) for faces, BERTScore/SBERT for text similarity, NetworkX for centrality, CUDA for speed.
- **Input**: a folder of images  
- **Output**: Parquet results, similarity pairs, and a centrality graph PNG (plus JSON exports)

## 🔗 Repository (Public)
**https://github.com/adurc95/memoline**  


---

## 🧭 Key Code Sections (with short purpose & importance)

> File: [`final_projact_album_graph_creater.py`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **Root logging (multi-process safe)** – sets up a shared queue and listener so every process writes to a single log file (critical for debugging parallel runs).  
  Link: [`setup_root_logger`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`OllamaService` (captions + embeddings)** – generates rich image descriptions and text embeddings with LLava. This is the backbone for text‑based similarity.  
  Link: [`class OllamaService`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`DeepFaceService` (faces + verify)** – extracts ArcFace embeddings and verifies same-person matches; enables **face bonus** to boost similarity for images with the same people.  
  Link: [`class DeepFaceService`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`ResultStorageParquet` (I/O + merge)** – writes per-image results and per-pair similarities to **Parquet**, and merges them efficiently for downstream steps.  
  Link: [`class ResultStorageParquet`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`SimilarityService` (BERTScore / SBERT)** – computes full similarity matrices via **BERTScore** (F1 / P / R) or **SBERT cosine**; pluggable for experimentation.  
  Link: [`class SimilarityService`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`FaceComparer` (CUDA bonus + clustering IDs)** – builds a face–face CUDA sim matrix, counts cross‑image matches, and assigns cluster IDs to faces; adds a capped **bonus** to text sim.  
  Link: [`class FaceComparer`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`CentralityGraphGenerator` (NetworkX graph)** – constructs the similarity graph (edges = strong similarity), computes **degree / closeness / harmonic / eigenvector / PageRank / betweenness**, and exports PNG + JSON.  
  Link: [`class CentralityGraphGenerator`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`ImageAnalyzer` (orchestration)** – parallel image processing, description/embedding extraction, similarity comparison (batch/CUDA), checkpointing, and storage.  
  Link: [`class ImageAnalyzer`](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

- **`main()` (entry point)** – defines the album path, kicks off processing, merges outputs, and generates the final centrality graph.  
  Link: [`main()` usage](https://github.com/adurc95/memoline/blob/main/final_projact_album_graph_creater.py)

> Why this structure matters: it separates **I/O**, **model calls**, **similarity logic**, and **graph analytics**, making the pipeline easier to test, profile, and extend (e.g., swap similarity method or face model).

---

## ▶️ How to Run

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2) Install dependencies (examples)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy tqdm pillow networkx matplotlib scikit-learn deepface sentence-transformers bert-score pynvml deprecated ollama

# 3) Make sure Ollama is running locally with the 'llava' model
#    (and that your GPU drivers/CUDA are available if you want acceleration)

# 4) Run the script
python final_projact_album_graph_creater.py
```

**Inputs/paths**: By default, `main()` points at:
```
C:/Users/user/PycharmProjects/memoline/test_album/mass_album
```
Change this to your album folder before running.

**Outputs** (timestamped):
- `results_parquet_<TS>/result_*.parquet` — per-image results (captions, embeddings, EXIF, faces)
- `sim_parquet_<TS>/sim_*.parquet` — per-pair similarity rows
- `image_results_<TS>.parquet` — merged per-image data
- `similarities_<TS>.parquet` — merged similarities
- `centrality_graph_<TS>.png` — graph visualization
- `centrality_metrics_<TS>.json` — centrality scores
- `graph_<TS>.json` — graph (node-link) export

---

## ⚙️ Configuration Tips
- **Similarity method**: switch in `compare_descriptions_textsim(...)` (`bertscore_f1`, `bertscore_precision_recall`, `sbert_cosine`).
- **Thresholds**: tune `text_thresh`, `face_thresh`, `base_bonus`, `max_bonus` depending on album diversity.
- **Performance**: batch size and parallelism are set in `ImageAnalyzer`; CUDA is used for heavy similarity and face computations.

