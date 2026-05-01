import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from .config import NeuralMemoryConfig


class NeuralMemoryModule(nn.Module):
    """
    Titans 스타일 Linear Associative Memory.

    Write: 새 임베딩이 들어오면 surprise 기반으로 Wk/Wv 파라미터 업데이트
    Read: 쿼리 임베딩으로 context vector를 읽어 반환

    수정 사항 (원본 설계 대비):
    1. Wk_prev/Wv_prev를 N step마다 갱신 → 정규화 항 실제 작동
    2. lr_gate gradient를 Wk/Wv와 분리
    """

    def __init__(self, config: NeuralMemoryConfig):
        super().__init__()
        self.config = config
        d = config.embed_dim
        m = config.memory_dim

        self.Wk = nn.Parameter(torch.randn(m, d) * config.init_scale)
        self.Wv = nn.Parameter(torch.randn(d, m) * config.init_scale)

        self.register_buffer('Wk_prev', self.Wk.data.clone())
        self.register_buffer('Wv_prev', self.Wv.data.clone())

        self.lr_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self._lr_gate_optimizer = torch.optim.Adam(self.lr_gate.parameters(), lr=1e-3)

        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.ReLU()

        self.write_count: int = 0
        self.total_surprise: float = 0.0

        self.to(config.device)

    def write(self, embedding: torch.Tensor) -> float:
        """
        새 임베딩을 메모리에 기록.

        Args:
            embedding: shape (d,) 임베딩 벡터
        Returns:
            surprise 값
        """
        e = embedding.detach().to(self.config.device)

        # Surprise 계산
        with torch.no_grad():
            e_hat = self._read_raw(e)
            surprise = torch.sum((e - e_hat) ** 2).item()

        # Adaptive learning rate
        surprise_tensor = torch.tensor([surprise], device=self.config.device)
        eta_scale = self.lr_gate(surprise_tensor)
        eta = self.config.lr * eta_scale.item()

        # Wk/Wv gradient 계산
        e_for_grad = e.detach().requires_grad_(False)
        e_hat = self._read_raw(e_for_grad)
        loss = torch.sum((e_for_grad - e_hat) ** 2)

        if self.config.reg_lambda > 0:
            reg = self.config.reg_lambda * (
                torch.sum((self.Wk - self.Wk_prev) ** 2) +
                torch.sum((self.Wv - self.Wv_prev) ** 2)
            )
            loss = loss + reg

        loss.backward()

        with torch.no_grad():
            self.Wk.data -= eta * self.Wk.grad
            self.Wv.data -= eta * self.Wv.grad
            self.Wk.grad.zero_()
            self.Wv.grad.zero_()

        # lr_gate 별도 학습
        self._update_lr_gate(surprise)

        # Wk_prev/Wv_prev 주기적 갱신
        self.write_count += 1
        self.total_surprise += surprise
        if self.write_count % self.config.reg_snapshot_interval == 0:
            with torch.no_grad():
                self.Wk_prev.copy_(self.Wk.data)
                self.Wv_prev.copy_(self.Wv.data)

        return surprise

    def read(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """쿼리로 context vector를 읽어 반환. shape (d,)"""
        q = query_embedding.to(self.config.device)
        with torch.no_grad():
            return self._read_raw(q)

    def _read_raw(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.Wk @ x)
        return self.Wv @ hidden

    def _update_lr_gate(self, surprise: float):
        avg_surprise = self.total_surprise / max(self.write_count, 1)
        if avg_surprise < 1e-8:
            return
        normalized = surprise / avg_surprise
        target = torch.sigmoid(torch.tensor(normalized - 1.0))
        target = target.to(self.config.device).unsqueeze(0)

        surprise_input = torch.tensor([surprise], device=self.config.device)
        prediction = self.lr_gate(surprise_input)

        gate_loss = nn.functional.mse_loss(prediction, target)
        self._lr_gate_optimizer.zero_grad()
        gate_loss.backward()
        self._lr_gate_optimizer.step()

    def get_stats(self) -> Dict:
        return {
            "write_count": self.write_count,
            "avg_surprise": self.total_surprise / max(self.write_count, 1),
            "Wk_norm": torch.norm(self.Wk).item(),
            "Wv_norm": torch.norm(self.Wv).item(),
            "Wk_drift": torch.norm(self.Wk - self.Wk_prev).item(),
            "Wv_drift": torch.norm(self.Wv - self.Wv_prev).item(),
        }
