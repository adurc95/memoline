import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Suppress all TF logs below ERROR:
#    0 = all logs shown (default)
#    1 = filter out INFO
#    2 = filter out INFO & WARNING
#    3 = filter out INFO, WARNING & ERROR (only FATAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 3. (Optional) Further enforce ERROR-only logging via Python’s logging API:
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import numpy as np
import logging
import logging.handlers
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import json
import uuid  # For generating unique file names
from PIL import Image, ExifTags
from tqdm import tqdm
from pathlib import Path
from ollama import Client
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
import traceback
import argparse
import multiprocessing
from multiprocessing import Pool
from collections import Counter, defaultdict
from deprecated import deprecated
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import threading
import time
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import uuid

# Global variables for the log queue and manager.
log_queue = None
manager_instance = None


# Create a global queue using a single Manager instance.
def get_log_queue():
    global manager_instance
    if manager_instance is None:
        from multiprocessing import Manager
        manager_instance = Manager()  # Manager automatically starts its server.
    return manager_instance.Queue()


# Configure the root logger and start a QueueListener in the main process.
def setup_root_logger():
    global log_queue
    log_queue = get_log_queue()
    formatter = logging.Formatter(
        '%(asctime)s - %(processName)s [%(process)d] - %(levelname)s - %(message)s'
    )
    log_filename = f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # console_handler.setLevel(logging.INFO)
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()
    print(f"Logging system initialized. Logs will be written to {log_filename}")
    return listener, log_filename


# Configure a logger in worker processes so that log messages go to the shared log_queue.
def configure_process_logger():
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    proc_name = multiprocessing.current_process().name
    logger.debug(f"Logger configured in process: {proc_name} (PID: {os.getpid()})")
    return logger


# Worker initializer that accepts the shared log_queue.
def worker_init(q):
    global log_queue
    log_queue = q
    logger = configure_process_logger()
    logger.info(
        f"Worker process initialized: {multiprocessing.current_process().name} (PID: {os.getpid()})"
    )







# ------------------ Service Classes ------------------

class GPUUtilMonitor:
    def __init__(self, gpu_index: int = 0, interval: float = 0.5, desc: str = "GPU Utilization"):
        self.enabled = torch.cuda.is_available()
        if not self.enabled:
            return
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

        self.interval = interval



        self.bar = tqdm(
            total   = 100,
            desc    = desc,
            unit    = "%",
            bar_format="{l_bar}{bar} {n:3d}%",
            leave=False
        )
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        while not self._stop_event.is_set():
            util = nvmlDeviceGetUtilizationRates(self.handle).gpu
            self.bar.n = util
            self.bar.refresh()
            time.sleep(self.interval)

    def start(self):
        if not hasattr(self, "enabled") or not self.enabled:
            return
        self._thread.start()

    def stop(self):
        if not hasattr(self, "enabled") or not self.enabled:
            return
        self._stop_event.set()
        self._thread.join()
        self.bar.clear()
        self.bar.close()



class OllamaService:
    def __init__(self, host='http://localhost:11434'):
        self.client = Client(host=host)
        self.logger = configure_process_logger()
        self.prompt = (
            'your job is to describe photos from a family album, go into details about how many people are in the pictures, '
            'what they are wearing, hair eyes and skin color, location, scenery, time of day, interactions '
            'between people, and what activity they are participating in'
            'give me your insight on the following'
            'place,	amount of people,  day/night time, activity, event, emotion/mood, objects, semantic summery, central image'
        )

    def describe_image(self, image_path):
        try:
            self.logger.info(f"Getting image description for {image_path}")
            with open(image_path, 'rb') as file:
                response = self.client.generate(
                    model='llava',
                    prompt=self.prompt,
                    images=[file.read()],
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'seed': 42,
                        'repeat_penalty': 1.1,
                    }
                )
            self.logger.info(f"Successfully got description for {image_path}")
            return response['response']
        except Exception as e:
            self.logger.error(f"Error getting description for {image_path}: {e}")
            return ""

    def get_text_embedding(self, text):
        try:
            self.logger.info("Getting text embedding")
            response = self.client.embeddings(
                model='llava',
                prompt=text
            )
            self.logger.info("Successfully got text embedding")
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return np.zeros(4096)



class DeepFaceService:
    def __init__(self):
        self.logger = configure_process_logger()

        # set ArcFace as the default model
        self.default_model_name = "ArcFace"
        self.detector_backend = "retinaface"
        self.enforce_detection = True

        # preload ArcFace once
        try:
            self.model = DeepFace.build_model(self.default_model_name)
            self.logger.debug(f"[DeepFaceService] Loaded model {self.default_model_name}")
        except Exception as e:
            self.logger.error(f"[DeepFaceService] Failed to load {self.default_model_name}: {e}")
            self.model = None

    def get_face_embeddings(self, image_path):
        """
        Same signature & return format as before (list of dicts):
          [
            {"facial_area": {...}, "embedding": [...], ...},
            ...
          ]
        """
        try:
            embeddings = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.default_model_name,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )
            self.logger.debug(f"Successfully calculated face embeddings for {image_path}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error calculating face embeddings for {image_path}: {e}")
            return []

    def verify_faces(
        self,
        img1,               # file path or HxWx3 np.ndarray (one face)
        img2,               # same
        distance_metric="cosine"
    ) -> dict:
        """
        Verify whether two single-face inputs are the same person using ArcFace.

        Returns DeepFace.verify’s dict:
          {
            "verified": bool,
            "distance": float,
            "threshold": float,
            "model": str,
            "distance_metric": str
          }
        """
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name=self.default_model_name,
                distance_metric=distance_metric,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )
            self.logger.debug(
                f"verify_faces: {img1} vs {img2} -> "
                f"{result['distance']:.4f} (thr={result['threshold']:.4f})"
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in verify_faces for {img1} vs {img2}: {e}")
            return {
                "verified": False,
                "distance": None,
                "threshold": None,
                "model": self.default_model_name,
                "distance_metric": distance_metric
            }

class EXIFService:
    def __init__(self):
        self.logger = configure_process_logger()

    def extract_exif_data(self, file_path):
        try:
            with Image.open(file_path) as img:
                dimensions = img.size
                exif = img.getexif()
                if exif:
                    exif_dict = {}
                    for tag_id in exif:
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        value = exif.get(tag_id)
                        if isinstance(value, bytes):
                            try:
                                value = value.decode()
                            except:
                                value = str(value)
                        exif_dict[str(tag)] = str(value)
                    self.logger.debug(f"Successfully extracting EXIF data from {file_path}")
                    return dimensions, json.dumps(exif_dict)
                else:
                    return dimensions, json.dumps({})
        except Exception as e:
            self.logger.error(f"Error extracting EXIF data from {file_path}: {e}")
            return None, json.dumps({})


class ResultStorageParquet:
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.results_dir = Path(f"results_parquet_{timestamp}")
        self.similarities_dir = Path(f"sim_parquet_{timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        self.similarities_dir.mkdir(exist_ok=True)
        self.checkpoint_file = f"checkpoint_{timestamp}.json"
        if not os.path.exists(self.checkpoint_file):
            self.save_checkpoint({'processed_files': [], 'compared_pairs': []})

    def save_checkpoint(self, data):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'processed_files': [], 'compared_pairs': []}

    def write_result(self, unique_filename, result):
        df = pd.DataFrame([result])
        file_path = self.results_dir / unique_filename
        df.to_parquet(file_path, index=False)
        return file_path

    def merge_results(self):
        all_files = list(self.results_dir.glob("*.parquet"))
        if not all_files:
            return None
        df_list = [pd.read_parquet(f) for f in all_files]
        #merged_df = pd.concat(df_list, ignore_index=True)
        final_file = f"image_results_{self.timestamp}.parquet"

        # 🧼 Defensive patch: sanitize dimensions column
        for df in df_list:
            if 'dimensions' in df.columns:
                df['dimensions'] = df['dimensions'].apply(
                    lambda x: f"{x[0]}x{x[1]}" if isinstance(x, (list, tuple)) else str(x) if x is not None else ""
                )

        merged_df = pd.concat(df_list, ignore_index=True)



        merged_df.to_parquet(final_file, index=False)
        return final_file

    def write_similarities(self, unique_filename, similarities):
        df = pd.DataFrame(similarities, columns=[
            'image1_path', 'image2_path', 'similarity_score', 'face_bonus', 'face_count'
        ])
        file_path = self.similarities_dir / unique_filename
        df.to_parquet(file_path, index=False)
        return file_path

    def merge_similarities(self):
        all_files = list(self.similarities_dir.glob("*.parquet"))
        if not all_files:
            return None
        df_list = [pd.read_parquet(f) for f in all_files]
        merged_df = pd.concat(df_list, ignore_index=True)
        final_file = f"similarities_{self.timestamp}.parquet"
        merged_df.to_parquet(final_file, index=False)
        return final_file




class SimilarityService:
    _sbert_model = None  # cache for SBERT model

    @staticmethod
    def calculate_similarity(embedding1, embedding2):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

    @staticmethod
    def compute_bertscore_f1_matrix(texts: list[str], lang='en'):
        N = len(texts)
        scores = torch.zeros((N, N))
        pairs_i, pairs_j, cand, ref = [], [], [], []

        for i in range(N):
            for j in range(i + 1, N):
                pairs_i.append(i)
                pairs_j.append(j)
                cand.append(texts[i])
                ref.append(texts[j])

        _, _, F1 = bert_score(cand, ref, lang=lang, verbose=True)

        for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
            f1 = F1[idx].item()
            scores[i, j] = scores[j, i] = f1

        return scores

    @staticmethod
    def compute_bertscore_precision_recall(texts: list[str], lang='en'):
        N = len(texts)
        sim_mat = torch.zeros((N, N))
        P_mat = torch.zeros((N, N))
        R_mat = torch.zeros((N, N))
        pairs_i, pairs_j, cand, ref = [], [], [], []

        for i in range(N):
            for j in range(i + 1, N):
                pairs_i.append(i)
                pairs_j.append(j)
                cand.append(texts[i])
                ref.append(texts[j])

        P, R, _ = bert_score(cand, ref, lang=lang, verbose=True)

        for idx, (i, j) in enumerate(zip(pairs_i, pairs_j)):
            p, r = P[idx].item(), R[idx].item()
            sim = (p + r) / 2
            sim_mat[i, j] = sim_mat[j, i] = sim
            P_mat[i, j] = P_mat[j, i] = p
            R_mat[i, j] = R_mat[j, i] = r

        return sim_mat, P_mat, R_mat

    @staticmethod
    def compute_sbert_cosine_matrix(texts: list[str], model_name='all-mpnet-base-v2'):
        if SimilarityService._sbert_model is None:
            SimilarityService._sbert_model = SentenceTransformer(model_name)

        emb = SimilarityService._sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        sim_matrix = util.pytorch_cos_sim(emb, emb)  # shape (N, N)
        return sim_matrix

    @staticmethod
    def compute_text_similarity_matrix(texts: list[str], method: str = 'bertscore_f1'):
        """
        Computes a full similarity matrix using the selected method.
        method: 'bertscore_f1' | 'bertscore_precision_recall' | 'sbert_cosine'
        Returns:
            - similarity matrix
            - optionally precision/recall matrices if applicable
        """
        if method == 'bertscore_f1':
            return SimilarityService.compute_bertscore_f1_matrix(texts)
        elif method == 'bertscore_precision_recall':
            return SimilarityService.compute_bertscore_precision_recall(texts)
        elif method == 'sbert_cosine':
            return SimilarityService.compute_sbert_cosine_matrix(texts)
        else:
            raise ValueError(f"Unknown similarity method: {method}")





class FaceComparer:
    def __init__(self, storage, deepface_service: DeepFaceService = None):
        self.storage = storage
        self.logger = configure_process_logger()
        self.deepface = deepface_service or DeepFaceService()

    def compute_face_bonus_verify(
            self,
            image1_path: str,
            image2_path: str,
            base_bonus: float = 0.1,
            max_bonus: float = 0.3,
            distance_metric: str = "cosine"
    ) -> tuple[float, int]:
        """
        Uses DeepFaceService.verify_faces under the hood.
        If ArcFace thinks the images depict the same person, you get one match.
        Returns (bonus, matches) in the same form as calculate_face_similarity_bonus.
        """
        try:
            result = self.deepface.verify_faces(
                image1_path,
                image2_path,
                distance_metric=distance_metric
            )
            verified = result.get("verified", False)
            matches = 1 if verified else 0
            bonus = min(base_bonus * matches, max_bonus)
            self.logger.info(
                f"[compute_face_bonus_verify] {image1_path} vs {image2_path}-> "
                f"verified={verified}, distance={result.get('distance'):.4f}, "
                f"bonus={bonus:.4f}"
            )
            return bonus, matches
        except Exception as e:
            self.logger.error(
                f"Error in compute_face_bonus_verify for {image1_path} vs {image2_path}: {e}"
            )
            return 0.0, 0


    def get_face_embeddings_from_result(self, image_path):
        try:
            merged_file = f"image_results_{self.storage.timestamp}.parquet"
            if not os.path.exists(merged_file):
                return []
            df = pd.read_parquet(merged_file)
            row = df[df['image_path'] == str(image_path)]
            if row.empty:
                return []
            face_embeddings_str = row.iloc[0]['face_embeddings']
            try:
                face_embeddings = json.loads(face_embeddings_str)
                for emb in face_embeddings:
                    if 'embedding' in emb and isinstance(emb['embedding'], list):
                        emb['embedding'] = np.array(emb['embedding'])
                return face_embeddings
            except Exception as e:
                self.logger.error(f"Error parsing face embeddings for {image_path}: {e}")
                return []
        except Exception as e:
            self.logger.error(f"Error loading face embeddings for {image_path}: {e}")
            return []

    def calculate_face_similarity_bonus(self, image1_path, image2_path, base_bonus=0.1):
        try:
            self.logger.info(f"Comparing faces between {image1_path} and {image2_path}")
            faces1 = self.get_face_embeddings_from_result(image1_path)
            faces2 = self.get_face_embeddings_from_result(image2_path)
            self.logger.debug(f"Loaded face embeddings - Image1: {len(faces1)} faces, Image2: {len(faces2)} faces")
            if not faces1 or not faces2:
                self.logger.info(f"No faces found in one or both images: {image1_path}, {image2_path}")
                return 0.0, 0
            matches = 0
            for i, face1 in enumerate(faces1):
                for j, face2 in enumerate(faces2):
                    similarity = cosine_similarity(
                        np.array(face1['embedding']).reshape(1, -1),
                        np.array(face2['embedding']).reshape(1, -1)
                    )[0][0]
                    self.logger.debug(f"Face similarity between face {i + 1} and face {j + 1}: {similarity:.4f}")
                    if similarity > 0.85:
                        matches += 1
                        self.logger.info(
                            f"Found matching faces! Face {i + 1} from image1 matches face {j + 1} from image2 (similarity: {similarity:.4f})"
                        )
            bonus = min(base_bonus * matches, 0.3)
            self.logger.info(f"Final face bonus: {bonus:.4f} based on {matches} matching faces")
            return bonus, matches
        except Exception as e:
            self.logger.error(f"Error calculating face similarity bonus: {e}")
            return 0.0, 0

    def collect_face_embeddings_cuda(self, paths: list[str]):
        """
        1) Walk each image_path, pull out all face embeddings,
        2) Build a flat list of embedding arrays, and 3) record
        which image‐index each face came from.
        Returns: (face_embeddings_list, image_to_face_index)
        """
        face_embeddings_list = []
        image_to_face_index = []
        for img_idx, img_path in enumerate(paths):
            emb_list = self.get_face_embeddings_from_result(img_path)
            for face in emb_list:
                face_embeddings_list.append(
                    np.array(face['embedding'], dtype=np.float32)
                )
                image_to_face_index.append(img_idx)
        return face_embeddings_list, image_to_face_index


    def compute_face_bonus_cuda(
            self,
            S_text: torch.Tensor,
            face_embeddings_list: list[np.ndarray],
            image_to_face_index: list[int],
            face_thresh: float,
            base_bonus: float,
            max_bonus: float,
            image_paths: list[str]  # 👈 add this

    ):
        """
        Given your text‐sim matrix and the raw face embeddings/index map,
        1) build and normalize the face‐face sim matrix,
        2) threshold & count cross‐image hits,
        3) build a bonus matrix clipped by max_bonus.
        Returns: (bonus_mat, Counter_of_matches)
        """



        # quick path if no faces
        if not face_embeddings_list:
            return torch.zeros_like(S_text), Counter()
        else:
            for i, face in enumerate(face_embeddings_list):
                norm = np.linalg.norm(face)
                #print(f"[DEBUG] Norm of face {i}: {norm:.4f}")

        #from collections import defaultdict

        # Step 1: Assign unique temp ID to each face
        face_ids = [str(uuid.uuid4()) for _ in face_embeddings_list]

        # Step 2: Initialize Union-Find parent map
        parent = {fid: fid for fid in face_ids}

        def find(fid):
            while parent[fid] != fid:
                parent[fid] = parent[parent[fid]]
                fid = parent[fid]
            return fid

        def union(f1, f2):
            parent[find(f1)] = find(f2)


        # a) build face‐face similarity
        F = torch.from_numpy(
            np.stack(face_embeddings_list, axis=0)
        ).to(DEVICE)
        Fn = torch.nn.functional.normalize(F, p=2, dim=1) #DeepFace is already returning L2-normalized embeddings for ArcFace
        S_face = torch.matmul(Fn, Fn.t())

        # b) find all face–face pairs above threshold
        M = S_face.size(0)
        tri = torch.triu_indices(M, M, offset=1, device=S_face.device)
        sims = S_face[tri[0], tri[1]]


        # --- LOG: All face–face similarity scores ---
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log_all_face_similarities(sims, tri, image_to_face_index, image_paths, S_face)



        hits = (sims > face_thresh).nonzero(as_tuple=False).squeeze(1)

        # c) count matches per image‐pair
        cnt = Counter()
        for h in hits.tolist():
            a, b = tri[0, h].item(), tri[1, h].item()
            ia, ib = image_to_face_index[a], image_to_face_index[b]
            union(face_ids[a], face_ids[b])
            if ia != ib:
                pair = tuple(sorted((ia, ib)))
                cnt[pair] += 1

        # Step 3: Generate a canonical match_id per group
        group_map = {}
        for fid in face_ids:
            root = find(fid)
            if root not in group_map:
                group_map[root] = str(uuid.uuid4())
        match_ids = [group_map[find(fid)] for fid in face_ids]



        # d) build bonus matrix
        bonus_mat = torch.zeros_like(S_text)
        for (i, j), match_count in cnt.items():
            bonus = min(base_bonus * match_count, max_bonus)
            bonus_mat[i, j] = bonus
            bonus_mat[j, i] = bonus

        return bonus_mat, cnt, match_ids, image_to_face_index

    def _log_all_face_similarities(
            self,
            sims: torch.Tensor,
            tri: torch.Tensor,
            image_to_face_index: list[int],
            image_paths: list[str],
            S_face: torch.Tensor
    ):

        img_idxs = torch.tensor(image_to_face_index, device=S_face.device)
        img_i = img_idxs[tri[0]]
        img_j = img_idxs[tri[1]]
        filename_lookup = {idx: Path(p).name for idx, p in enumerate(image_paths)}

        for idx in range(sims.size(0)):
            face_a = tri[0, idx].item()
            face_b = tri[1, idx].item()
            img_a = img_i[idx].item()
            img_b = img_j[idx].item()
            name_a = filename_lookup.get(img_a, f"[{img_a}]")
            name_b = filename_lookup.get(img_b, f"[{img_b}]")
            self.logger.debug(
                f"[face-sim] {name_a} (face {face_a}) vs {name_b} (face {face_b}) -> sim={sims[idx].item():.4f}"
            )


class CentralityGraphGenerator:
    def __init__(self, storage, timestamp):
        self.storage = storage
        self.timestamp = timestamp
        self.logger = configure_process_logger()

    def _compute_centralities(self, G):
        """
        Compute exact centrality metrics on G (up to ~1 000 nodes) using
        NumPy-accelerated backends where available.
        Returns a dict: {metric_name: {node: value}}.
        """
        metrics = {}

        # 1) Degree centrality (O(V + E))
        metrics['degree'] = nx.degree_centrality(G)

        # 2) Closeness centrality (O(V·(V + E)))
        metrics['closeness'] = nx.closeness_centrality(G)

        # 3) Harmonic centrality (handles disconnected graphs)
        metrics['harmonic'] = nx.harmonic_centrality(G)

        # 4) Eigenvector centrality via NumPy (dense eigendecomposition)
        try:
            metrics['eigenvector'] = nx.eigenvector_centrality_numpy(G)
        except Exception as e:
            self.logger.warning(f"NumPy eigenvector failed, falling back to power iteration: {e}")
            metrics['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)

        # 5) PageRank via NumPy
        try:
            metrics['pagerank'] = nx.pagerank_numpy(G)
        except Exception as e:
            self.logger.warning(f"NumPy PageRank failed, falling back to iterative PageRank: {e}")
            metrics['pagerank'] = nx.pagerank(G)

        # 6) Betweenness centrality (Brandes' exact algorithm, O(V·E))
        metrics['betweenness'] = nx.betweenness_centrality(G, normalized=True, weight='weight')

        return metrics

    def create_graph(self, threshold=0.8, save_path=None, compute_metrics=None):
        self.logger.info(f"Creating centrality graph with threshold {threshold}")
        # start GPU monitor
        #print("starting gpu monitor for create_graph")
        gpu_monitor = GPUUtilMonitor()
        gpu_monitor.start()

        G = self._load_similarity_graph(threshold)
        if G is None:
            return None

        metrics = self._compute_requested_metrics(G, compute_metrics)

        meta_lookup = self._load_metadata()
        self._enrich_nodes(G, metrics, meta_lookup)

        self._save_metrics(metrics)
        self._visualize_graph(G, metrics, save_path)
        self._export_graph(G)

        #print("stopping gpu monitor for create_graph")
        gpu_monitor.stop()


        return metrics

    def _load_similarity_graph(self, threshold):
        """1) Load parquet, filter by threshold, build and return graph or None."""
        file = f"similarities_{self.storage.timestamp}.parquet"
        if not os.path.exists(file):
            self.logger.warning("No similarities file found")
            return None

        df = pd.read_parquet(file)
        df = df[df['similarity_score'] >= threshold]
        G = nx.Graph()
        for _, row in df.iterrows():
            n1, n2 = Path(row['image1_path']).name, Path(row['image2_path']).name
            G.add_edge(n1, n2, weight=row['similarity_score'])

        if G.number_of_nodes() == 0:
            self.logger.warning("No nodes in graph – no similarities above threshold")
            return None
        return G

    def _compute_requested_metrics(self, G, compute_metrics):
        """2) Compute centralities and pick only requested ones."""
        all_metrics = self._compute_centralities(G)
        if compute_metrics:
            return {m: all_metrics[m] for m in compute_metrics if m in all_metrics}
        return all_metrics

    def _load_metadata(self):
        """3) Read per-image parquet and build basename→metadata dict."""
        file = f"image_results_{self.storage.timestamp}.parquet"
        if not os.path.exists(file):
            self.logger.warning(f"No results file {file}; skipping metadata enrichment")
            return {}
        df = pd.read_parquet(file)
        df['basename'] = df['image_path'].map(lambda p: Path(p).name)
        return df.set_index('basename').to_dict(orient='index')

    def _enrich_nodes(self, G, metrics, meta_lookup):
        """4) Attach metadata and centrality scores as node attributes."""
        for node in G.nodes():
            attrs = {}

            # a) metadata
            meta = meta_lookup.get(node, {})
            for col, val in meta.items():
                if col == 'image_path':
                    continue
                v = val
                if col in ('embedding','face_embeddings') and isinstance(val, str):
                    try: v = json.loads(val)
                    except: pass
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                if not isinstance(v, (str,bool,int,float,list,dict)):
                    v = str(v)
                attrs[col] = v

            # b) centrality scores
            for m_name, score_map in metrics.items():
                m = score_map.get(node)
                if isinstance(m, np.generic):
                    m = m.item()
                attrs[m_name] = m

            nx.set_node_attributes(G, {node: attrs})

    def _save_metrics(self, metrics):
        """5) Dump metrics to JSON."""
        fname = f"centrality_metrics_{self.timestamp}.json"
        with open(fname, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Saved centrality metrics to {fname}")

    def _visualize_graph(self, G, metrics, save_path):
        """6) Draw and either show or save the matplotlib figure."""
        deg = metrics.get('degree', {})
        eig = metrics.get('eigenvector', {})
        fig, ax = plt.subplots(figsize=(12,8))
        pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=[eig.get(n,0)*3000 for n in G.nodes()],
            node_color=[deg.get(n,0) for n in G.nodes()],
            cmap=plt.cm.viridis
        )
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.2,
            width=[G[u][v]['weight']*2 for u,v in G.edges()]
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.set_title(
            "Image Similarity Network\n"
            "Node size: Eigenvector Centrality | Node color: Degree Centrality"
        )

        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis,
            norm=plt.Normalize(vmin=min(deg.values()), vmax=max(deg.values()))
        )
        sm.set_array(list(deg.values()))
        fig.colorbar(sm, ax=ax, label='Degree Centrality')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            self.logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def _export_graph(self, G):
        """7) Export enriched graph as JSON (or GraphML/GEXF)."""
        base = f"graph_{self.timestamp}"
        with open(f"{base}.json", 'w') as f:
            json.dump(nx.node_link_data(G), f, indent=2)
        self.logger.info("Exported enriched graph to JSON format")
        # optionally: nx.write_graphml(G, f"{base}.graphml")
        # optionally: nx.write_gexf(G, f"{base}.gexf")


class ImageAnalyzer:
    def __init__(self, folder_path, batch_size=1000):
        self.folder_path = Path(folder_path)
        self.batch_size = batch_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = configure_process_logger()
        self.logger.info(f"ImageAnalyzer initialized with folder: {folder_path}")
        self.storage = ResultStorageParquet(self.timestamp)
        self.ollama_service = None
        self.deepface_service = None
        self.exif_service = None
        self.face_comparer = None
        self.processed_count = 0
        self.clear_cache_interval = 5
        self.face_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ["logger", "ollama_service", "deepface_service", "exif_service", "face_comparer"]:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = configure_process_logger()
        self.ollama_service = OllamaService()
        self.deepface_service = DeepFaceService()
        self.exif_service = EXIFService()
        #self.face_comparer = FaceComparer(self.storage)

    def process_single_image(self, file_path):
        self.logger.info(f"Processing {file_path}")
        try:
            checkpoint = self.storage.load_checkpoint()
            if str(file_path) in checkpoint.get('processed_files', []):
                self.logger.info(f"Skipping already processed file: {file_path}")
                return None
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return None
            result = {
                'image_path': str(file_path),
                'description': '',
                'embedding': '',
                'processed': False,
                'exif': '',
                'face_embeddings': '',
                'face_count': 0,
                'file_size': 0,
                'creation_time': 0,
                'modification_time': 0,
                'dimensions': ''
            }
            description = self.ollama_service.describe_image(file_path)
            embedding = self.ollama_service.get_text_embedding(description)
            result['description'] = description
            try:
                result['embedding'] = json.dumps(embedding)
            except Exception as e:
                self.logger.error(f"Error serializing embedding for {file_path}: {e}")
                result['embedding'] = json.dumps([])
            file_stats = os.stat(file_path)
            result['file_size'] = file_stats.st_size
            result['creation_time'] = file_stats.st_ctime
            result['modification_time'] = file_stats.st_mtime

            dimensions, exif_data = self.exif_service.extract_exif_data(file_path)

            # Always convert dimensions to a clean string, or "" if missing
            if dimensions:
                result['dimensions'] = f"{dimensions[0]}x{dimensions[1]}" if isinstance(dimensions, tuple) else str(
                    dimensions)
            else:
                result['dimensions'] = ""



            result['dimensions'] = dimensions if dimensions else ""
            result['exif'] = exif_data
            result['processed'] = True
            embeddings = self.deepface_service.get_face_embeddings(file_path)
            if isinstance(embeddings, dict):
                embeddings = [embeddings]
            processed_embeddings = []
            for emb in embeddings:
                if isinstance(emb.get('embedding'), np.ndarray):
                    emb['embedding'] = emb['embedding'].tolist()
                processed_embeddings.append(emb)
            result['face_embeddings'] = json.dumps(processed_embeddings)
            result['face_count'] = len(processed_embeddings)
            self.processed_count += 1
            if self.processed_count % self.clear_cache_interval == 0:
                self.logger.info("Clearing face cache and running garbage collection")
                self.face_cache.clear()
                import gc
                gc.collect()
            unique_filename = f"result_{os.getpid()}_{uuid.uuid4()}.parquet"
            file_full_path = self.storage.write_result(unique_filename, result)
            checkpoint.setdefault('processed_files', []).append(str(file_path))
            self.storage.save_checkpoint(checkpoint)
            return str(file_full_path)
        except Exception as e:
            self.logger.error(f"Critical error processing {file_path}: {e}")
            traceback.print_exc()
            return None


    def process_images_parallel(self):
        # start GPU monitor
       # print("starting gpu monitor for process images parallel")
        gpu_monitor = GPUUtilMonitor()
        gpu_monitor.start()


        # 1. Gather all image files
        image_files = list(self.folder_path.glob('*'))

        # 2. Decide on worker count
        num_processes = max(1, os.cpu_count() // 3)

        chunksize=np.size(image_files)//num_processes
        self.logger.info(f"Starting parallel processing with {num_processes} processes")

        # 3. Process everything through one Pool, using your existing log_queue & worker_init
        file_paths = []
        with Pool(
            processes=num_processes,
            initializer=worker_init,
            initargs=(log_queue,)
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(self.process_single_image, image_files, chunksize=chunksize),
                total=len(image_files),
                desc="Processing images",
            ):
                if result is not None:
                    file_paths.append(result)

        #stop the monitor
        gpu_monitor.stop()
        #print("stopping gpu monitor for process images parallel")


        return file_paths


    def process_images_parallel_old(self):
        image_files = list(self.folder_path.glob('*'))
        num_processes = max(1, os.cpu_count() //3)
        self.logger.info(f"Starting parallel processing with {num_processes} processes")
        chunk_size = min(num_processes, len(image_files))
        file_paths = []
        for i in range(0, len(image_files), chunk_size):
            chunk = image_files[i:i + chunk_size]
            self.logger.info(
                f"Processing chunk {i // chunk_size + 1} of {(len(image_files) + chunk_size - 1) // chunk_size}"
            )
            with Pool(num_processes, initializer=worker_init, initargs=(log_queue,)) as pool:
                chunk_file_paths = list(tqdm(
                    pool.imap_unordered(self.process_single_image, chunk),
                    total=len(chunk),
                    desc=f"Processing images {i + 1}-{min(i + chunk_size, len(image_files))}"
                ))
                file_paths.extend([fp for fp in chunk_file_paths if fp is not None])
                self.logger.info(f"Completed chunk {i // chunk_size + 1}, running garbage collection")
                import gc
                gc.collect()
        return file_paths


    @deprecated(version='22.0', reason="This method is deprecated compare_embeddings_batch deprecate")
    def compare_image_pair(self, pair_data):
        row1, row2 = pair_data
        pair_key = f"{row1['image_path']},{row2['image_path']}"
        self.logger.info(f"Processing {row1['image_path']},{row2['image_path']}")
        try:
            try:
                emb1_list = json.loads(row1['embedding'])
            except Exception as e:
                self.logger.error(f"Error parsing JSON for row1 embedding: {e}")
                return None
            try:
                emb2_list = json.loads(row2['embedding'])
            except Exception as e:
                self.logger.error(f"Error parsing JSON for row2 embedding: {e}")
                return None
            emb1 = np.array(emb1_list)
            emb2 = np.array(emb2_list)
            base_similarity = SimilarityService.calculate_similarity(emb1, emb2)
            self.logger.debug(f"Base similarity for {pair_key}: {base_similarity:.4f}")
            face_bonus, face_count = self.face_comparer.calculate_face_similarity_bonus(
                row1['image_path'], row2['image_path']
            )
            final_similarity = min(base_similarity + face_bonus, 1.0)
            self.logger.info(
                f"Final similarity for {pair_key}: {final_similarity:.4f} (base: {base_similarity:.4f}, bonus: {face_bonus:.4f})"
            )
            return {
                'image1_path': row1['image_path'],
                'image2_path': row2['image_path'],
                'similarity_score': final_similarity,
                'face_bonus': face_bonus,
                'face_count': face_count,
                'pair_key': pair_key
            }
        except Exception as e:
            self.logger.error(f"Error comparing {row1['image_path']} and {row2['image_path']}: {e}")
            return None

    def compare_descriptions_textsim(
            self,
            batch_df,
            method: str = 'bertscore_f1',  # 'bertscore_f1', 'bertscore_precision_recall', or 'sbert_cosine'
            text_thresh: float = 0.9,
            base_bonus: float = 0.1,
            max_bonus: float = 0.3,
            face_thresh: float = 0.460
    ):
        self.face_comparer = FaceComparer(self.storage)

        # Start GPU monitor
        gpu_monitor = GPUUtilMonitor(desc=f"Comparing ({method})")
        gpu_monitor.start()

        # 1) --- LOAD TEXTS (descriptions) ---
        paths = batch_df['image_path'].tolist()
        texts = batch_df['description'].tolist()
        N = len(paths)

        # 2) --- COMPUTE TEXT SIMILARITY MATRIX ---
        sim_data = SimilarityService.compute_text_similarity_matrix(texts, method=method)
        if method == 'bertscore_precision_recall':
            S_text, _, _ = sim_data
        else:
            S_text = sim_data

        S_text = S_text.to(DEVICE)# if not S_text.is_cuda else S_text

        # 3) --- FACE BONUS ---
        face_embeddings_list, image_to_face_index = self.face_comparer.collect_face_embeddings_cuda(paths)
        if face_embeddings_list:
            bonus_mat, cnt, match_ids, image_to_face_index = self.face_comparer.compute_face_bonus_cuda(
                S_text,
                face_embeddings_list,
                image_to_face_index,
                face_thresh,
                base_bonus,
                max_bonus,
                paths
            )
            S_final = torch.clamp(S_text + bonus_mat, max=1.0)
        else:
            bonus_mat = torch.zeros_like(S_text)
            cnt = {}
            S_final = S_text

        # 4) --- THRESHOLD AND COLLECT PAIRS ---
        mask = torch.triu(torch.ones(N, N, device=S_final.device), diagonal=1).bool()
        vals = S_final[mask]
        idxs = (vals >= text_thresh).nonzero(as_tuple=False).squeeze(1)

        tri = torch.triu_indices(N, N, offset=1, device=S_final.device)
        sel_i = tri[0, idxs].tolist()
        sel_j = tri[1, idxs].tolist()



        # 5) update the face_embeddings with the mutches id
        img_to_faces = defaultdict(list)
        for global_face_idx, (img_idx, match_id) in enumerate(zip(image_to_face_index, match_ids)):
            img_to_faces[img_idx].append((global_face_idx, match_id))

        df = pd.read_parquet(f"image_results_{self.storage.timestamp}.parquet")
        for img_idx, face_infos in img_to_faces.items():
            row = df.iloc[img_idx]
            try:
                face_data = json.loads(row['face_embeddings'])
                for i, (_, match_id) in enumerate(face_infos):
                    face_data[i]['match_id'] = match_id
                df.at[img_idx, 'face_embeddings'] = json.dumps(face_data)
            except Exception as e:
                self.logger.warning(f"Couldn't patch match_ids for image index {img_idx}: {e}")

        df.to_parquet(f"image_results_{self.storage.timestamp}.parquet", index=False)



        results = []
        for i, j in tqdm(zip(sel_i, sel_j), desc=f"{method} pair comparisons", total=len(sel_i), unit="pair"):
            results.append({
                'image1_path': paths[i],
                'image2_path': paths[j],
                'similarity_score': float(S_final[i, j].cpu()),
                'face_bonus': float(bonus_mat[i, j].cpu() if face_embeddings_list else 0.0),
                'face_count': int(cnt.get(tuple(sorted((i, j))), 0) if face_embeddings_list else 0),
                'pair_key': f"{paths[i]},{paths[j]}"
            })

        # 5) SAVE RESULTS + CHECKPOINT
        if results:
            unique_sim_filename = f"sim_{os.getpid()}_{uuid.uuid4()}.parquet"
            self.storage.write_similarities(
                unique_sim_filename,
                [[r['image1_path'], r['image2_path'],
                  r['similarity_score'], r['face_bonus'], r['face_count']]
                 for r in results]
            )
            cp = self.storage.load_checkpoint()
            cp['compared_pairs'].extend(r['pair_key'] for r in results)
            self.storage.save_checkpoint(cp)

        gpu_monitor.stop()
        return results

    def compare_embeddings_batch_new(
            self,
            batch_df,
            text_thresh: float = 0.8,
            base_bonus: float = 0.1,
            max_bonus: float = 0.3,
            distance_metric: str = "cosine"
    ):
        # re-init your FaceComparer (which holds the preloaded ArcFace model)
        self.face_comparer = FaceComparer(self.storage,self.deepface_service)

        # 1) start GPU monitor
        gpu_monitor = GPUUtilMonitor()
        gpu_monitor.start()

        # 2) build the text‐similarity matrix (N×N) exactly as before
        paths = batch_df['image_path'].tolist()
        text_emb = np.stack([np.array(json.loads(e), dtype=np.float32)
                             for e in batch_df['embedding']], axis=0)
        T = torch.from_numpy(text_emb).to(DEVICE)
        Tn = torch.nn.functional.normalize(T, p=2, dim=1)
        S_text = torch.matmul(Tn, Tn.t())

        # 3) prepare face‐bonus containers
        N = len(paths)
        bonus_mat = torch.zeros_like(S_text)
        cnt = Counter()

        # 4) for each i<j, call ArcFace verify and build bonus_mat exactly once
        for i in range(N):
            for j in range(i + 1, N):
                bonus, matches = self.face_comparer.compute_face_bonus_verify(
                    paths[i],
                    paths[j],
                    base_bonus=base_bonus,
                    max_bonus=max_bonus,
                    distance_metric=distance_metric
                )
                if matches:
                    bonus_mat[i, j] = bonus
                    bonus_mat[j, i] = bonus
                    cnt[(i, j)] = matches

        # 5) combine into your final similarity
        S_final = torch.clamp(S_text + bonus_mat, max=1.0)

        # 6) threshold & recover (i,j) just as you already do
        mask = torch.triu(torch.ones(N, N, device=S_final.device), diagonal=1).bool()
        vals = S_final[mask]
        idxs = (vals >= text_thresh).nonzero(as_tuple=False).squeeze(1)

        triN = torch.triu_indices(N, N, offset=1, device=S_final.device)
        sel_i = triN[0, idxs].tolist()
        sel_j = triN[1, idxs].tolist()

        # 7) assemble your results list
        results = []
        for i, j in tqdm(zip(sel_i, sel_j),
                         desc="CUDA pair comparisons",
                         total=len(sel_i),
                         unit="pair"):
            results.append({
                'image1_path': paths[i],
                'image2_path': paths[j],
                'similarity_score': float(S_final[i, j].cpu()),
                'face_bonus': float(bonus_mat[i, j]),
                'face_count': int(cnt.get((i, j), 0)),
                'pair_key': f"{paths[i]},{paths[j]}"
            })

        # 8) stop monitor and return
        gpu_monitor.stop()
        return results

    def compare_embeddings_batch_cuda(self, batch_df, text_thresh=0.8,face_thresh=0.85,base_bonus=0.1,max_bonus=0.3):

        self.face_comparer = FaceComparer(self.storage)

        # start GPU monitor
        #print("starting gpu monitor for compare_embeddings_batch_cuda")
        gpu_monitor = GPUUtilMonitor()
        gpu_monitor.start()


        # 1) --- TEXT EMBEDDINGS MATRIX (N×D_text)
        paths = batch_df['image_path'].tolist()
        text_embs = np.stack([np.array(json.loads(e), dtype=np.float32)
                              for e in batch_df['embedding']], axis=0)
        T = torch.from_numpy(text_embs).to(DEVICE)  # (N, D)
        Tn = torch.nn.functional.normalize(T, p=2, dim=1)  # unit‐length
        S_text = torch.matmul(Tn, Tn.t())  # (N, N)

        '''
        N=S_text.size(0)
        mask = ~torch.eye(N, dtype=bool, device=S_text.device)
        vals = S_text[mask]

        mn, mx = vals.min(), vals.max()
        S_text = (S_text - mn) / (mx - mn + 1e-8)

'''

        # 1) mask out the diagonal if you don't want the 1.0’s in your stats
        N = S_text.size(0)
        mask = ~torch.eye(N, dtype=torch.bool, device=S_text.device)

        # 2) compute mean & std over only those entries
        vals = S_text[mask]  # shape (N*(N-1),)
        mean, std = vals.mean(), vals.std(unbiased=False)

        # 3) z-normalize the entire matrix
        eps = 1e-6  # to avoid div0
        S_text = (S_text - mean) / (std + eps)






        # 2) --- COLLECT FACE EMBEDDINGS (M×D_face) + INDEX MAP
        paths = batch_df['image_path'].tolist()
        face_embeddings_list, image_to_face_index = self.face_comparer.collect_face_embeddings_cuda(paths)


        # 3) BUILD FINAL SIM MATRIX, WITH OR WITHOUT FACE-BONUS
        bonus_mat, cnt = self.face_comparer.compute_face_bonus_cuda(
            S_text,
            face_embeddings_list,
            image_to_face_index,
            face_thresh,
            base_bonus,
            max_bonus
        )





        if face_embeddings_list:
            S_final = torch.clamp(S_text + bonus_mat, max=1.0)
        else:
            S_final = S_text


        # 4) THRESHOLD & RECOVER (i, j) PAIRS ABOVE text_thresh
        N = S_final.size(0)
        mask = torch.triu(torch.ones(N, N, device=S_final.device), diagonal=1).bool()
        vals = S_final[mask]
        idxs = (vals >= text_thresh).nonzero(as_tuple=False).squeeze(1)

        triN = torch.triu_indices(N, N, offset=1, device=S_final.device)
        sel_i = triN[0, idxs].tolist()
        sel_j = triN[1, idxs].tolist()

        # 5) ASSEMBLE RESULT DICTS
        results = []

        pairs = list(zip(sel_i, sel_j))
        for i, j in tqdm(pairs,
                         desc="CUDA pair comparisons",
                         total=len(pairs),
                         unit="pair"):
            results.append({
                'image1_path': paths[i],
                'image2_path': paths[j],
                'similarity_score': float(S_final[i, j].cpu()),
                'face_bonus': float((bonus_mat[i, j] if face_embeddings_list else 0.0)),
                'face_count': int(cnt.get(tuple(sorted((i, j))), 0) if face_embeddings_list else 0),
                'pair_key': f"{paths[i]},{paths[j]}"
            })




        '''
        for i, j in zip(sel_i, sel_j):
            results.append({
                'image1_path': paths[i],
                'image2_path': paths[j],
                'similarity_score': float(S_final[i, j].cpu()),
                'face_bonus': float((bonus_mat[i, j] if face_embeddings_list  else 0.0)),
                'face_count': int(cnt.get(tuple(sorted((i, j))), 0) if face_embeddings_list  else 0),
                'pair_key': f"{paths[i]},{paths[j]}"
            })'''

        # 6) SAVE & CHECKPOINT AS BEFORE

        if results:
            unique_sim_filename = f"sim_{os.getpid()}_{uuid.uuid4()}.parquet"
            self.storage.write_similarities(
                unique_sim_filename,
                [[r['image1_path'], r['image2_path'],
                  r['similarity_score'], r['face_bonus'], r['face_count']]
                 for r in results]
            )
            cp = self.storage.load_checkpoint()
            cp['compared_pairs'].extend(r['pair_key'] for r in results)
            self.storage.save_checkpoint(cp)

        #stop the monitor
        #print("stopping gpu monitor for compare_embeddings_batch_cuda")
        gpu_monitor.stop()

        return results

    @deprecated(version='22.0', reason="This method is deprecated parrelity is done with cuda now")
    def compare_embeddings_batch_cuda_old(self, batch_df, threshold=0.8):
        """
        GPU‐accelerated batch comparison:
        - Loads each row['embedding'] JSON → float32 vector
        - Builds an N×D tensor, moves it to cuda
        - Computes full N×N cosine matrix on GPU
        - Extracts i<j pairs where sim ≥ threshold
        - Applies face‐bonus & returns results
        """
        self.face_comparer = FaceComparer(self.storage)

        # 1) parse embeddings into a single float32 tensor on the CPU
        emb_list = []
        paths = []
        for _, row in batch_df.iterrows():
            arr = np.array(json.loads(row['embedding']), dtype=np.float32)
            emb_list.append(arr)
            paths.append(row['image_path'])
        emb_cpu = np.stack(emb_list, axis=0)  # shape (N, D)

        # 2) move to GPU
        emb = torch.from_numpy(emb_cpu).to(DEVICE)  # (N, D)
        emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)

        # 3) full cosine‐similarity: sim_matrix[i,j] = emb_norm[i]·emb_norm[j]
        #    note: this builds a (N x N) matrix — watch your batch sizes!
        sim_matrix = torch.matmul(emb_norm, emb_norm.t())  # (N, N)




        # 4) mask out the diagonal & lower triangle, threshold
        N = sim_matrix.size(0)

        # 1) build GPU masks & indices
        mask = torch.triu(torch.ones(N, N, device=sim_matrix.device), diagonal=1).bool()
        sim_vals = sim_matrix[mask]
        idxs = torch.nonzero(sim_vals >= threshold, as_tuple=False).squeeze(1)

        # 2) build GPU-resident triangle indices
        tri_i, tri_j = torch.triu_indices(N, N, offset=1, device=sim_matrix.device)

        # 3) pick out only the i<j pairs above threshold
        sel_i = tri_i[idxs].tolist()
        sel_j = tri_j[idxs].tolist()

        results = []
        for i, j in zip(sel_i, sel_j):
            path_i, path_j = paths[i], paths[j]
            base_sim = float(sim_matrix[i, j].cpu().item())
            # 5) apply your existing face‐bonus logic
            bonus, faces = self.face_comparer.calculate_face_similarity_bonus(path_i, path_j)
            final_sim = min(base_sim + bonus, 1.0)
            pair_key = f"{path_i},{path_j}"
            results.append({
                'image1_path': path_i,
                'image2_path': path_j,
                'similarity_score': final_sim,
                'face_bonus': bonus,
                'face_count': faces,
                'pair_key': pair_key
            })

        # save & checkpoint just like before
        if results:
            unique_sim_filename = f"sim_{os.getpid()}_{uuid.uuid4()}.parquet"
            self.storage.write_similarities(unique_sim_filename, [
                [r['image1_path'], r['image2_path'], r['similarity_score'], r['face_bonus'], r['face_count']]
                for r in results
            ])
            cp = self.storage.load_checkpoint()
            cp['compared_pairs'].extend(r['pair_key'] for r in results)
            self.storage.save_checkpoint(cp)

        return results


    @deprecated(version='22.0', reason="This method is deprecated parrelity is done with cuda now")
    def compare_embeddings_batch(self, batch_df):
        checkpoint = self.storage.load_checkpoint()
        comparison_tasks = []
        for i, row1 in batch_df.iterrows():
            for j, row2 in batch_df.iloc[i + 1:].iterrows():
                pair_key = f"{row1['image_path']},{row2['image_path']}"
                if pair_key in checkpoint['compared_pairs']:
                    self.logger.debug(f"Skipping already compared pair: {pair_key}")
                    continue
                comparison_tasks.append((row1, row2))
        if not comparison_tasks:
            self.logger.info("No new pairs to compare in this batch")
            return []
        self.logger.info(f"Starting parallel comparison of {len(comparison_tasks)} image pairs")
        num_processes = max(1, os.cpu_count() //3)
        try:
            with Pool(num_processes, initializer=worker_init, initargs=(log_queue,)) as pool:
                results = list(pool.map(self.compare_image_pair, comparison_tasks))
        except Exception as e:
            self.logger.error(f"Error in parallel processing: {e}")
            self.logger.info("Falling back to sequential processing")
            results = [self.compare_image_pair(task) for task in tqdm(comparison_tasks, desc="Comparing sequentially")]
        valid_results = [r for r in results if r is not None]
        self.logger.info(f"Successfully compared {len(valid_results)} pairs")
        unique_sim_filename = f"sim_{os.getpid()}_{uuid.uuid4()}.parquet"
        self.storage.write_similarities(unique_sim_filename, [
            [res['image1_path'], res['image2_path'], res['similarity_score'], res['face_bonus'], res['face_count']]
            for res in valid_results
        ])
        for result in valid_results:
            checkpoint['compared_pairs'].append(result['pair_key'])
        self.storage.save_checkpoint(checkpoint)
        self.logger.info("Batch comparison complete")
        import gc
        gc.collect()
        return valid_results

    def compare_all_images(self):
        self.logger.info("Starting similarity comparisons")
        try:
            merged_results_file = self.storage.merge_results()
            if merged_results_file is None:
                self.logger.error("No results file to process for comparisons")
                return
            df = pd.read_parquet(merged_results_file)
            for chunk_num, chunk in enumerate(np.array_split(df, max(1, len(df) // self.batch_size))):
                self.logger.info(f"Processing comparison batch {chunk_num + 1}")
                self.compare_descriptions_textsim(chunk)
            self.logger.info("Completed similarity comparisons")
        except Exception as e:
            self.logger.error(f"Error during similarity comparisons: {e}")

    def find_similar_images(self, threshold=0.8):
        similar_pairs = []
        self.logger.info(f"Finding similar images with threshold {threshold}")
        try:
            merged_sim_file = self.storage.merge_similarities()
            if not merged_sim_file:
                self.logger.error("No similarities file available")
                return similar_pairs
            df = pd.read_parquet(merged_sim_file)
            df = df[df['similarity_score'] >= threshold]
            self.logger.info(f"Found {len(df)} similar pairs above threshold")
            for _, row in df.iterrows():
                similar_pairs.append({
                    'image1': row['image1_path'],
                    'image2': row['image2_path'],
                    'similarity': row['similarity_score'],
                    'face_bonus': row['face_bonus'],
                    'matching_faces': row['face_count']
                })
        except Exception as e:
            self.logger.error(f"Error finding similar images: {e}")
        return similar_pairs


def main():
    parser = argparse.ArgumentParser(description="Process images with resume functionality")
    parser.add_argument("--resume", action="store_true", help="Resume processing from an existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Checkpoint timestamp to resume from (e.g. 20230428_123456)")
    args = parser.parse_args()
    listener, log_filename = setup_root_logger()
    logger = configure_process_logger()
    logger.info(f"Main process started with PID: {os.getpid()}")
    try:


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = "C:/Users/user/PycharmProjects/memoline/test_album/album"
        analyzer = ImageAnalyzer(folder_path)
        analyzer.timestamp = timestamp
        analyzer.storage = ResultStorageParquet(timestamp)
        result_files = analyzer.process_images_parallel()
        logger.info(f"Successfully processed {len(result_files)} images")
        analyzer.compare_all_images()
        analyzer.storage.merge_similarities()
        #similar_pairs = analyzer.find_similar_images(threshold=0.8)
        graph_generator = CentralityGraphGenerator(analyzer.storage, analyzer.timestamp)
        graph_path = f"centrality_graph_{analyzer.timestamp}.png"
        centrality_metrics = graph_generator.create_graph(threshold=0.8, save_path=graph_path)
        logger.info("Processing Complete!")
        """
        final_results_file = analyzer.storage.merge_results()
        final_similarities_file = analyzer.storage.merge_similarities()
        logger.info(f"Final results merged into: {final_results_file}")
        logger.info(f"Final similarities merged into: {final_similarities_file}")
        print(f"\nProcessing Complete!")
        print(f"Final results merged into: {final_results_file}")
        print(f"Final similarities merged into: {final_similarities_file}")
        print(f"Centrality graph saved to: {graph_path}")
        if centrality_metrics:
            print("\nTop 5 Most Central Images:")
            eigenvector_central = sorted(
                centrality_metrics['eigenvector'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for img, score in eigenvector_central:
                print(f"{img}: {score:.4f}")
        """


    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        logger.info("Shutting down logging system")
        listener.stop()
        print("Logging shutdown complete")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed on Windows.
    main()
