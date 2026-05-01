import torch
from sentence_transformers import SentenceTransformer

from .config import NeuralMemoryConfig


class ExperienceEncoder:
    """
    에이전트의 경험(텍스트)을 임베딩 벡터로 변환.
    sentence-transformers의 사전학습 모델을 사용하며, freeze 상태로 유지.
    """

    def __init__(self, config: NeuralMemoryConfig):
        self.model = SentenceTransformer(config.encoder_model)
        self.model.eval()
        self.device = config.device

    def encode(self, text: str) -> torch.Tensor:
        """단일 텍스트를 임베딩 벡터로 변환"""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
        return embedding.to(self.device)

    def encode_batch(self, texts: list) -> torch.Tensor:
        """복수 텍스트를 일괄 변환"""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)
