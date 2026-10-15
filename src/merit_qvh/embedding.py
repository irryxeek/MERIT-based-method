"""Frozen Qwen3-Embedding-4B with the upstream no-prefix cosine protocol."""

import os
import tempfile
from pathlib import Path

from .core import digest


class Embedder:
    def __init__(self, config, root):
        self.config = config
        self.root = Path(root) / "embeddings"
        self.model = None

    def encode(self, texts):
        import numpy as np

        identity = digest({"config": self.config, "texts": texts})
        path = self.root / (identity + ".npy")
        if path.exists():
            return np.load(path, allow_pickle=False)
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.config["cpu_threads"])
            self.model = SentenceTransformer(
                self.config["weight_path"],
                device=self.config["device"],
                local_files_only=True,
                model_kwargs={
                    "torch_dtype": getattr(torch, self.config["dtype"]),
                    "attn_implementation": "sdpa",
                },
            )
            self.model.max_seq_length = self.config["max_seq_length"]
        result = self.model.encode(
            texts,
            batch_size=self.config["batch_size"],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        if not np.isfinite(result).all() or (np.linalg.norm(result, axis=-1) <= 0).any():
            raise ValueError("Embedding model returned invalid vectors")
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.root, prefix=".writing-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, result, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return result
