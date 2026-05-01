from dataclasses import dataclass


@dataclass
class NeuralMemoryConfig:
    # 원본 Smallville은 OpenAI text-embedding-ada-002 사용 (dim=1536)
    embed_dim: int = 1536             # 임베딩 차원
    memory_dim: int = 512             # 메모리 내부 차원
    lr: float = 0.01                  # 메모리 쓰기 학습률
    reg_lambda: float = 0.05          # 정규화 계수
    activation: str = "relu"          # 활성화 함수 ("relu" or "softmax")
    device: str = "cpu"               # "cuda" or "cpu"

    # [FIX] Wk_prev 주기적 갱신 간격
    reg_snapshot_interval: int = 10

    # [FIX] Decoder 검색 효율화 — sliding window
    decoder_search_window: int = 200

    # Weight 초기화 스케일
    init_scale: float = 0.01
