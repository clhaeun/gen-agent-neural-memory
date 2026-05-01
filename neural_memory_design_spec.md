# Neural Memory Module for Generative Embodied Agents: Implementation Design Specification

## 문서 목적
이 문서는 VS Code의 Claude Code가 실험 플랫폼을 세팅하고, Neural Memory 아키텍처를 설계·구현하며, 초기 실험을 돌리기 위한 **구체적인 실행 계획서**입니다. 모든 단계는 순서대로 실행되어야 하며, 각 단계의 완료 조건이 명시되어 있습니다.

---

## 1. 프로젝트 개요

### 1.1 연구 목적
Generative Embodied Agent(Stanford Generative Agents / Smallville 기반)의 메모리 시스템을 Neural Memory Module로 교체하고, 기존 RAG 기반 메모리와 유사한 성능이 나오는지 초기 검증하는 것.

### 1.2 핵심 가설
Titans 스타일의 Neural Memory Module이 에이전트의 과거 경험을 파라미터에 압축하여, 명시적 검색(RAG) 없이도 과거 맥락을 현재 의사결정에 반영할 수 있다.

### 1.3 이번 실험의 범위 (Scope)
- **In scope**: Neural Memory Module 구현, Smallville 메모리 시스템 교체, RAG 대비 기본 동작 비교
- **Out of scope**: 엄밀한 정량 평가, 인간 평가, 논문용 ablation study

---

## 2. 환경 세팅

### 2.1 Smallville (Generative Agents) 코드베이스 세팅

```bash
# Step 1: 저장소 클론
git clone https://github.com/joonspk-research/generative_agents.git
cd generative_agents

# Step 2: Python 환경 구성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 3: 추가 의존성 설치
pip install torch torchvision  # Neural Memory 구현용
pip install sentence-transformers  # 임베딩 생성용
pip install openai  # LLM API 호출용
pip install wandb  # 실험 로깅용 (선택)
```

### 2.2 디렉토리 구조 설계

```
generative_agents/
├── reverie/                          # 원본 Smallville 코드
│   └── backend_server/
│       ├── persona/
│       │   ├── memory_structures/    # ★ 여기를 수정
│       │   │   ├── associative_memory.py  # 원본 RAG 메모리
│       │   │   └── scratch.py
│       │   ├── cognitive_modules/    # ★ 여기에 neural memory 연결
│       │   │   ├── retrieve.py       # 원본 retrieval 로직
│       │   │   ├── plan.py
│       │   │   └── reflect.py
│       │   └── persona.py
│       └── ...
├── neural_memory/                    # ★ 새로 생성할 디렉토리
│   ├── __init__.py
│   ├── memory_module.py              # Neural Memory Module 핵심 구현
│   ├── encoder.py                    # 경험 → 임베딩 변환기
│   ├── decoder.py                    # 메모리 출력 → 텍스트 변환기
│   ├── config.py                     # 하이퍼파라미터 설정
│   └── utils.py                      # 유틸리티 함수
├── experiments/                      # ★ 새로 생성할 디렉토리
│   ├── run_baseline.py               # 원본 RAG로 실험 실행
│   ├── run_neural_memory.py          # Neural Memory로 실험 실행
│   ├── compare_results.py            # 결과 비교 스크립트
│   └── configs/
│       ├── baseline_config.yaml
│       └── neural_memory_config.yaml
└── logs/                             # 실험 로그 저장
```

### 2.3 환경 세팅 완료 조건
- [ ] Smallville 원본 코드가 에러 없이 실행됨
- [ ] 에이전트 2~3명이 시뮬레이션 1일차를 완료함
- [ ] 에이전트의 메모리 스트림 로그를 확인할 수 있음

---

## 3. 원본 Smallville 메모리 시스템 분석

### 3.1 먼저 반드시 확인할 파일들

Claude Code는 구현을 시작하기 전에 반드시 아래 파일들을 읽고 구조를 완전히 파악해야 합니다.

```
# 반드시 읽어야 할 파일 목록 (우선순위 순)
1. reverie/backend_server/persona/memory_structures/associative_memory.py
   → 메모리 저장/검색의 전체 구조 파악
   → 특히 add_memory(), retrieve() 메서드의 입출력 형식 확인

2. reverie/backend_server/persona/cognitive_modules/retrieve.py
   → retrieve 함수가 어떻게 호출되는지, 반환값이 무엇인지 확인
   → 중요도(importance), 최신성(recency), 관련성(relevance) 점수 계산 방식 확인

3. reverie/backend_server/persona/cognitive_modules/plan.py
   → 메모리에서 가져온 정보가 planning에 어떻게 사용되는지 확인

4. reverie/backend_server/persona/cognitive_modules/reflect.py
   → reflection이 메모리를 어떻게 읽고 쓰는지 확인

5. reverie/backend_server/persona/persona.py
   → 위 모듈들이 어떤 순서로 호출되는지 전체 파이프라인 확인
```

### 3.2 원본 메모리 시스템의 핵심 인터페이스 (예상 구조)

Claude Code가 코드를 읽은 후 확인해야 할 핵심 인터페이스:

```python
# 원본 associative_memory.py의 핵심 메서드 (정확한 시그니처는 코드에서 확인)

class AssociativeMemory:
    def add_memory(self, memory_node):
        """새로운 경험을 메모리에 저장"""
        # memory_node는 보통 다음을 포함:
        # - description: str (경험의 텍스트 설명)
        # - timestamp: datetime (발생 시각)
        # - importance: float (중요도 점수)
        # - embedding: List[float] (텍스트 임베딩 벡터)
        pass
    
    def retrieve(self, query, top_k=10):
        """쿼리에 관련된 메모리를 검색하여 반환"""
        # 반환값: List[memory_node]
        # 점수 = importance * recency * relevance 의 가중합
        pass
```

**중요**: 위 구조는 예상이므로, 실제 코드의 시그니처와 다를 수 있습니다. Claude Code는 반드시 실제 코드를 읽고 정확한 인터페이스를 파악한 후 구현해야 합니다.

---

## 4. Neural Memory Module 아키텍처 설계

### 4.1 설계 원칙

1. **인터페이스 호환성**: 원본 AssociativeMemory와 동일한 입출력 인터페이스를 유지. 나머지 시스템(plan, reflect 등)은 수정하지 않음.
2. **최소 구현**: 첫 실험에서는 가장 단순한 linear associative memory로 시작. 복잡한 구조는 이후 추가.
3. **하이브리드 가능**: neural memory 출력을 텍스트로 변환하여 기존 LLM 프롬프트에 주입하는 방식. LLM 자체는 수정하지 않음.

### 4.2 핵심 아키텍처: Linear Associative Memory

```
┌─────────────────────────────────────────────────┐
│              NeuralMemoryModule                  │
│                                                  │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐ │
│  │ Encoder  │───▶│ Memory φ   │───▶│ Decoder  │ │
│  │ (frozen) │    │ (Wk, Wv)   │    │ (frozen) │ │
│  └──────────┘    └────────────┘    └──────────┘ │
│                                                  │
│  Write: 경험 텍스트 → 임베딩 → φ 업데이트        │
│  Read:  쿼리 텍스트 → 임베딩 → φ 조회 → 텍스트   │
└─────────────────────────────────────────────────┘
```

### 4.3 수학적 정의

**파라미터 정의:**
```
d = 384  (임베딩 차원, all-MiniLM-L6-v2 기준)
m = 512  (메모리 내부 차원)

Wk ∈ R^{m × d}    # Key 매핑 행렬 (학습됨)
Wv ∈ R^{d × m}    # Value 매핑 행렬 (학습됨)
η = 0.01           # 메모리 쓰기 학습률
λ = 0.05           # 망각 방지 정규화 계수
```

**Write 연산 (새 경험 기록):**
```
입력: e ∈ R^d (경험 임베딩 벡터)

1. Surprise 계산:
   ê = Wv · σ(Wk · e)                    # 메모리에서 예측
   L_surprise = || e - ê ||²              # 예측과 현실의 차이

2. 파라미터 업데이트:
   L_total = L_surprise + λ·(||Wk - Wk_prev||² + ||Wv - Wv_prev||²)
   Wk ← Wk - η · ∂L_total/∂Wk
   Wv ← Wv - η · ∂L_total/∂Wv

3. Adaptive learning rate (선택적):
   η_adaptive = η₀ · sigmoid(MLP(L_surprise))
   → surprise가 극단적일 때 학습률 자동 조절
```

**Read 연산 (기억 조회):**
```
입력: q ∈ R^d (쿼리 임베딩 벡터)

1. 메모리 읽기:
   c = Wv · σ(Wk · q)                    # 맥락 벡터 출력

2. Top-k 관련 기억 복원:
   → c와 과거 기록된 경험 임베딩들의 유사도 계산
   → 또는 c 자체를 디코더에 통과시켜 텍스트로 변환
```

### 4.4 PyTorch 구현 명세

```python
# neural_memory/config.py

from dataclasses import dataclass

@dataclass
class NeuralMemoryConfig:
    embed_dim: int = 384          # sentence-transformers 임베딩 차원
    memory_dim: int = 512         # 메모리 내부 차원
    lr: float = 0.01              # 메모리 쓰기 학습률
    reg_lambda: float = 0.05      # 정규화 계수
    activation: str = "relu"      # 활성화 함수 ("relu" or "softmax")
    device: str = "cuda"          # "cuda" or "cpu"
    
    # Encoder 설정
    encoder_model: str = "all-MiniLM-L6-v2"  # sentence-transformers 모델
    
    # 하이브리드 모드 설정 (neural memory + 소규모 버퍼)
    buffer_size: int = 50         # 최근 경험 버퍼 크기 (fallback용)
    use_buffer: bool = True       # 버퍼 사용 여부
```

```python
# neural_memory/memory_module.py

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy

@dataclass
class MemoryEntry:
    """단일 기억 항목"""
    description: str
    embedding: torch.Tensor       # shape: (d,)
    timestamp: str                # 시뮬레이션 내 시각
    importance: float
    node_id: int                  # 원본 시스템과의 호환용 ID

class NeuralMemoryModule(nn.Module):
    """
    Titans 스타일 Linear Associative Memory.
    
    핵심 메커니즘:
    - Write: 새 경험이 들어오면 surprise 기반으로 파라미터 업데이트
    - Read: 쿼리가 들어오면 파라미터에서 맥락 벡터를 읽어 반환
    
    원본 Smallville의 AssociativeMemory와 동일한 인터페이스를 제공해야 함.
    """
    
    def __init__(self, config: NeuralMemoryConfig):
        super().__init__()
        self.config = config
        d = config.embed_dim
        m = config.memory_dim
        
        # === 핵심 메모리 파라미터 ===
        self.Wk = nn.Parameter(torch.randn(m, d) * 0.01)  # Key 행렬
        self.Wv = nn.Parameter(torch.randn(d, m) * 0.01)  # Value 행렬
        
        # === 이전 파라미터 스냅샷 (정규화용) ===
        self.register_buffer('Wk_prev', self.Wk.data.clone())
        self.register_buffer('Wv_prev', self.Wv.data.clone())
        
        # === Adaptive learning rate MLP (선택적) ===
        self.lr_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # === 활성화 함수 ===
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "softmax":
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.ReLU()
        
        # === 최근 경험 버퍼 (하이브리드 모드) ===
        self.recent_buffer: List[MemoryEntry] = []
        
        # === 전체 기록 (디버깅 및 비교용, 실제 추론에는 사용하지 않음) ===
        self.all_entries: List[MemoryEntry] = []
        
        # === 메모리 통계 ===
        self.write_count: int = 0
        self.total_surprise: float = 0.0
        
        self.to(config.device)
    
    def write(self, entry: MemoryEntry) -> float:
        """
        새 경험을 메모리에 기록 (파라미터 업데이트).
        
        Args:
            entry: 기록할 경험
        Returns:
            surprise: 이번 경험의 surprise 값 (디버깅용)
        """
        e = entry.embedding.to(self.config.device)
        
        # Step 1: Surprise 계산 (현재 메모리의 예측과 실제의 차이)
        with torch.no_grad():
            e_hat = self._read_raw(e)
            surprise = torch.sum((e - e_hat) ** 2).item()
        
        # Step 2: 파라미터 업데이트
        # 학습률 결정
        if self.config.reg_lambda > 0:
            surprise_tensor = torch.tensor([surprise], device=self.config.device)
            eta = self.config.lr * self.lr_gate(surprise_tensor).item()
        else:
            eta = self.config.lr
        
        # Gradient 계산 및 업데이트
        e_hat = self._read_raw(e)
        loss = torch.sum((e - e_hat) ** 2)
        
        # 정규화 항 추가
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
            
            # 이전 파라미터 스냅샷 갱신
            self.Wk_prev.copy_(self.Wk.data)
            self.Wv_prev.copy_(self.Wv.data)
        
        # Step 3: 버퍼 및 기록 업데이트
        self.all_entries.append(entry)
        if self.config.use_buffer:
            self.recent_buffer.append(entry)
            if len(self.recent_buffer) > self.config.buffer_size:
                self.recent_buffer.pop(0)
        
        self.write_count += 1
        self.total_surprise += surprise
        
        return surprise
    
    def read(self, query_embedding: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, List[MemoryEntry]]:
        """
        쿼리를 기반으로 메모리에서 맥락 정보를 읽어옴.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터, shape (d,)
            top_k: 반환할 관련 기억 수 (버퍼에서)
        Returns:
            context_vector: 메모리에서 읽은 맥락 벡터, shape (d,)
            buffer_entries: 버퍼에서 유사도 기준으로 선택된 최근 기억들
        """
        q = query_embedding.to(self.config.device)
        
        # Neural memory에서 맥락 벡터 읽기
        with torch.no_grad():
            context = self._read_raw(q)
        
        # 하이브리드: 버퍼에서도 유사한 기억 검색
        buffer_entries = []
        if self.config.use_buffer and len(self.recent_buffer) > 0:
            similarities = []
            for entry in self.recent_buffer:
                sim = torch.cosine_similarity(
                    q.unsqueeze(0), 
                    entry.embedding.unsqueeze(0).to(self.config.device)
                ).item()
                similarities.append(sim)
            
            # 상위 top_k개 선택
            indices = sorted(range(len(similarities)), 
                           key=lambda i: similarities[i], reverse=True)[:top_k]
            buffer_entries = [self.recent_buffer[i] for i in indices]
        
        return context, buffer_entries
    
    def _read_raw(self, x: torch.Tensor) -> torch.Tensor:
        """내부 읽기 연산: x → Wv · σ(Wk · x)"""
        hidden = self.activation(self.Wk @ x)
        output = self.Wv @ hidden
        return output
    
    def get_stats(self) -> Dict:
        """현재 메모리 상태 통계 반환 (디버깅용)"""
        return {
            "write_count": self.write_count,
            "avg_surprise": self.total_surprise / max(self.write_count, 1),
            "buffer_size": len(self.recent_buffer),
            "Wk_norm": torch.norm(self.Wk).item(),
            "Wv_norm": torch.norm(self.Wv).item(),
        }
```

```python
# neural_memory/encoder.py

from sentence_transformers import SentenceTransformer
import torch
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
```

```python
# neural_memory/decoder.py

from typing import List, Tuple
import torch
from .memory_module import NeuralMemoryModule, MemoryEntry

class MemoryDecoder:
    """
    Neural Memory의 출력(맥락 벡터)을 텍스트로 변환하여
    LLM 프롬프트에 주입할 수 있는 형태로 만듦.
    
    첫 번째 구현은 단순 방식:
    - 맥락 벡터와 가장 유사한 과거 기억의 텍스트를 반환
    - 버퍼의 기억도 함께 반환
    """
    
    def __init__(self):
        pass
    
    def decode(
        self, 
        context_vector: torch.Tensor,
        buffer_entries: List[MemoryEntry],
        all_entries: List[MemoryEntry],
        top_k: int = 5
    ) -> List[str]:
        """
        맥락 벡터를 기반으로 관련 기억 텍스트를 반환.
        
        전략:
        1. context_vector와 전체 기록에서 코사인 유사도 기준 top-k 선택
        2. buffer_entries에서 추가 기억 포함
        3. 중복 제거 후 반환
        
        Args:
            context_vector: neural memory에서 읽은 벡터
            buffer_entries: 최근 버퍼에서 선택된 기억들
            all_entries: 전체 기억 기록
            top_k: 반환할 기억 수
        Returns:
            descriptions: 관련 기억 텍스트 리스트
        """
        results = []
        seen_ids = set()
        
        # 1. Context vector 기반 검색
        if len(all_entries) > 0:
            similarities = []
            for entry in all_entries:
                sim = torch.cosine_similarity(
                    context_vector.unsqueeze(0),
                    entry.embedding.unsqueeze(0).to(context_vector.device)
                ).item()
                similarities.append((sim, entry))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            for sim, entry in similarities[:top_k]:
                if entry.node_id not in seen_ids:
                    results.append(entry.description)
                    seen_ids.add(entry.node_id)
        
        # 2. Buffer entries 추가
        for entry in buffer_entries:
            if entry.node_id not in seen_ids:
                results.append(entry.description)
                seen_ids.add(entry.node_id)
        
        return results[:top_k]
```

### 4.5 구현 시 주의사항

1. **Gradient 관리**: write() 시 backward()를 호출하므로, computation graph가 누적되지 않도록 매 write 후 grad를 반드시 zero로 초기화.
2. **디바이스 일관성**: 임베딩, 메모리 파라미터, 쿼리가 모두 같은 디바이스에 있어야 함.
3. **수치 안정성**: Wk, Wv 초기화를 작은 값(0.01 스케일)으로 해야 초기 surprise 폭발을 방지.
4. **메모리 누수**: all_entries 리스트가 무한히 커지므로, 디버깅 후에는 비활성화하거나 주기적으로 정리하는 로직 필요.

---

## 5. Smallville 통합 (Integration)

### 5.1 통합 전략: Adapter Pattern

원본 코드를 최소한으로 수정하기 위해 **Adapter 클래스**를 만들어 원본 AssociativeMemory의 인터페이스를 그대로 구현합니다.

```python
# neural_memory/adapter.py

"""
Smallville의 AssociativeMemory를 대체하는 어댑터.

중요: 이 파일의 구현은 원본 associative_memory.py의 실제 인터페이스를 확인한 후
정확히 맞춰야 합니다. 아래는 예상 인터페이스 기반의 스켈레톤입니다.

Claude Code가 해야 할 일:
1. 원본 associative_memory.py를 읽고 모든 public 메서드의 시그니처를 확인
2. 아래 어댑터가 동일한 시그니처를 갖도록 수정
3. 원본에서 사용하는 데이터 구조(ConceptNode 등)를 그대로 활용
"""

from .memory_module import NeuralMemoryModule, MemoryEntry
from .encoder import ExperienceEncoder  
from .decoder import MemoryDecoder
from .config import NeuralMemoryConfig

class NeuralMemoryAdapter:
    """
    원본 AssociativeMemory와 동일한 인터페이스를 제공하면서
    내부적으로 NeuralMemoryModule을 사용.
    """
    
    def __init__(self, config: NeuralMemoryConfig):
        self.memory = NeuralMemoryModule(config)
        self.encoder = ExperienceEncoder(config)
        self.decoder = MemoryDecoder()
        self.config = config
        self._node_counter = 0
    
    def add_memory(self, description, timestamp, importance, **kwargs):
        """
        원본 AssociativeMemory.add_memory()와 동일한 시그니처.
        
        ★ 실제 시그니처는 원본 코드를 확인 후 맞출 것.
        """
        # 1. 텍스트를 임베딩으로 변환
        embedding = self.encoder.encode(description)
        
        # 2. MemoryEntry 생성
        entry = MemoryEntry(
            description=description,
            embedding=embedding,
            timestamp=str(timestamp),
            importance=importance,
            node_id=self._node_counter
        )
        self._node_counter += 1
        
        # 3. Neural Memory에 기록
        surprise = self.memory.write(entry)
        
        return entry  # 원본이 반환하는 형식에 맞출 것
    
    def retrieve(self, query, top_k=10, **kwargs):
        """
        원본 AssociativeMemory.retrieve()와 동일한 시그니처.
        
        ★ 실제 시그니처는 원본 코드를 확인 후 맞출 것.
        ★ 특히 반환값의 형식(ConceptNode 리스트 등)을 정확히 맞춰야 함.
        """
        # 1. 쿼리를 임베딩으로 변환
        query_embedding = self.encoder.encode(query)
        
        # 2. Neural Memory에서 읽기
        context_vector, buffer_entries = self.memory.read(query_embedding, top_k)
        
        # 3. 텍스트로 디코딩
        descriptions = self.decoder.decode(
            context_vector, 
            buffer_entries,
            self.memory.all_entries,
            top_k
        )
        
        # 4. 원본 반환 형식에 맞춰 변환
        # ★ 원본이 ConceptNode 객체를 반환한다면, 여기서 변환해야 함
        return descriptions
```

### 5.2 원본 코드 수정 지점

```python
# 수정할 파일: reverie/backend_server/persona/persona.py (또는 해당 위치)
# 
# 수정 방법: 메모리 초기화 부분에서 조건 분기 추가
#
# ★ 실제 수정은 원본 코드의 정확한 구조를 확인한 후 진행할 것

# 원본:
# self.memory = AssociativeMemory(...)

# 수정 후:
# if USE_NEURAL_MEMORY:
#     from neural_memory.adapter import NeuralMemoryAdapter
#     from neural_memory.config import NeuralMemoryConfig
#     config = NeuralMemoryConfig()
#     self.memory = NeuralMemoryAdapter(config)
# else:
#     self.memory = AssociativeMemory(...)

# USE_NEURAL_MEMORY는 experiments/configs/ 의 설정 파일에서 제어
```

### 5.3 통합 시 예상되는 문제점과 해결 방안

**문제 1: 원본이 메모리 객체의 내부 속성에 직접 접근하는 경우**
- 예: `persona.memory.some_internal_list` 같은 접근이 코드 곳곳에 있을 수 있음
- 해결: `NeuralMemoryAdapter`에 해당 속성을 `@property`로 추가하여 호환성 유지

**문제 2: 원본의 retrieve()가 단순 텍스트가 아닌 복잡한 객체를 반환하는 경우**
- 예: `ConceptNode` 객체 리스트를 반환하고, 호출자가 `.description`, `.importance` 등에 접근
- 해결: `MemoryEntry`를 `ConceptNode`와 동일한 인터페이스를 갖도록 확장하거나, 래퍼 객체 생성

**문제 3: Reflection 모듈이 메모리를 특수한 방식으로 읽고 쓰는 경우**
- 예: 최근 N개 메모리를 가져와서 요약 후 다시 저장하는 reflect 로직
- 해결: `NeuralMemoryAdapter`에 `get_recent(n)` 같은 메서드 추가, 내부 버퍼에서 반환

---

## 6. 실험 실행 계획

### 6.1 Phase 1: Smoke Test (1일, 최우선)

**목적**: Neural Memory가 에러 없이 Smallville과 함께 돌아가는지 확인

```python
# experiments/smoke_test.py

"""
Smoke Test: 시뮬레이션 1시간(게임 내 시간)을 돌려서
에이전트가 에러 없이 행동하는지만 확인.

실행 방법:
1. Smallville 원본으로 1시간 실행 → 로그 저장
2. Neural Memory로 1시간 실행 → 로그 저장
3. 두 로그를 비교하여 에이전트가 비정상 행동(무한 루프, 빈 응답 등)을 하지 않는지 확인

확인 사항:
- [ ] 에이전트가 매 스텝 유효한 행동을 생성하는가
- [ ] 메모리 write가 에러 없이 수행되는가
- [ ] 메모리 read가 에러 없이 수행되는가
- [ ] LLM에 전달되는 프롬프트에 기억이 포함되어 있는가
- [ ] GPU 메모리가 시간이 지나도 폭발하지 않는가 (메모리 누수 체크)
"""
```

### 6.2 Phase 2: Baseline 비교 실행 (3~5일)

**목적**: 동일한 시나리오에서 원본 RAG vs Neural Memory의 행동 비교

```yaml
# experiments/configs/comparison_config.yaml

experiment:
  name: "neural_memory_vs_rag_v1"
  simulation_days: 3          # 시뮬레이션 게임 내 3일
  num_agents: 3               # 에이전트 3명 (비용 제어)
  random_seed: 42             # 재현성

baseline:
  type: "original_rag"        # Smallville 원본 메모리

neural_memory:
  embed_dim: 384
  memory_dim: 512
  lr: 0.01
  reg_lambda: 0.05
  activation: "relu"
  buffer_size: 50
  use_buffer: true

logging:
  save_memory_states: true     # 매 시간마다 메모리 상태 스냅샷 저장
  save_agent_actions: true     # 모든 에이전트 행동 로그 저장
  save_llm_prompts: true       # LLM에 보내는 프롬프트 저장 (디버깅용)
  log_dir: "logs/comparison_v1"
```

```python
# experiments/compare_results.py

"""
비교 분석 스크립트.

이 단계에서는 엄밀한 정량 평가가 아니라, 다음을 정성적으로 확인:

1. 행동 유사성 체크:
   - 두 시스템의 에이전트가 대략적으로 비슷한 일과(routine)를 보이는가
   - Neural Memory 에이전트가 비정상적으로 반복적이거나 무의미한 행동을 하지 않는가

2. 기억 활용 체크:
   - 시뮬레이션 2일차에 1일차의 경험을 언급하는 행동이 있는가
   - Neural Memory의 surprise 값 추이 그래프 (안정적으로 감소하는지)
   
3. 비용 체크:
   - 각 시스템의 LLM API 호출 횟수와 토큰 사용량
   - Neural Memory의 추론 시간 (write/read 각각)

출력:
   - logs/comparison_v1/summary_report.md
   - logs/comparison_v1/surprise_curve.png
   - logs/comparison_v1/action_logs_side_by_side.txt
"""
```

### 6.3 Phase 3: 하이퍼파라미터 탐색 (선택적)

Phase 2에서 문제가 발견되면 다음을 조정:

```
조정 우선순위:
1. lr (0.001, 0.01, 0.1) — surprise 곡선이 불안정하면 낮추기
2. reg_lambda (0.01, 0.05, 0.1) — 기억이 빠르게 사라지면 올리기  
3. memory_dim (256, 512, 1024) — 기억 용량이 부족하면 올리기
4. buffer_size (0, 25, 50, 100) — 버퍼 의존도 확인
5. use_buffer (true/false) — 버퍼 없이 순수 neural memory만으로 가능한지 확인
```

---

## 7. 로깅 및 디버깅 체계

### 7.1 필수 로그 항목

```python
# neural_memory/utils.py

import json
import os
from datetime import datetime

class MemoryLogger:
    """Neural Memory의 동작을 추적하는 로거"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "memory_log.jsonl")
    
    def log_write(self, agent_name: str, description: str, 
                  surprise: float, timestamp: str):
        """메모리 쓰기 이벤트 로깅"""
        entry = {
            "event": "write",
            "agent": agent_name,
            "description": description,
            "surprise": surprise,
            "sim_timestamp": timestamp,
            "real_timestamp": datetime.now().isoformat()
        }
        self._append(entry)
    
    def log_read(self, agent_name: str, query: str, 
                 retrieved_descriptions: list, timestamp: str):
        """메모리 읽기 이벤트 로깅"""
        entry = {
            "event": "read",
            "agent": agent_name,
            "query": query,
            "retrieved": retrieved_descriptions,
            "sim_timestamp": timestamp,
            "real_timestamp": datetime.now().isoformat()
        }
        self._append(entry)
    
    def log_stats(self, agent_name: str, stats: dict, timestamp: str):
        """메모리 상태 통계 로깅"""
        entry = {
            "event": "stats",
            "agent": agent_name,
            "stats": stats,
            "sim_timestamp": timestamp,
            "real_timestamp": datetime.now().isoformat()
        }
        self._append(entry)
    
    def _append(self, entry: dict):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

### 7.2 디버깅 체크포인트

시뮬레이션 중 다음 시점에서 자동으로 상태를 저장:
- 매 게임 내 1시간마다 메모리 파라미터 스냅샷 (torch.save)
- 매 에이전트 행동마다 surprise 값 기록
- 매 LLM 호출마다 프롬프트 전문과 응답 저장

---

## 8. 실행 순서 체크리스트 (Claude Code용)

Claude Code는 아래 순서를 **엄격하게** 따라야 합니다. 각 단계의 완료 조건이 충족되지 않으면 다음 단계로 넘어가지 마십시오.

### Phase 0: 사전 준비
- [ ] Smallville 저장소 클론 및 의존성 설치
- [ ] Smallville 원본 코드 실행 확인 (에이전트 1명, 시뮬레이션 30분)
- [ ] **원본 코드의 메모리 관련 파일 5개를 모두 읽고 구조를 완전히 파악**
  - associative_memory.py의 모든 public 메서드 시그니처 기록
  - retrieve.py의 호출 패턴 및 반환값 형식 기록
  - plan.py, reflect.py에서 메모리를 사용하는 모든 지점 기록
  - persona.py에서 메모리 초기화 및 호출 흐름 기록

### Phase 1: Neural Memory 구현
- [ ] neural_memory/ 디렉토리 생성 및 config.py 작성
- [ ] memory_module.py 구현 (NeuralMemoryModule 클래스)
- [ ] encoder.py 구현 (ExperienceEncoder 클래스)  
- [ ] decoder.py 구현 (MemoryDecoder 클래스)
- [ ] **단위 테스트**: NeuralMemoryModule의 write/read가 에러 없이 작동하는지 확인
  ```python
  # 최소 단위 테스트
  config = NeuralMemoryConfig(device="cpu")
  memory = NeuralMemoryModule(config)
  encoder = ExperienceEncoder(config)
  
  # Write 테스트
  emb = encoder.encode("Isabella went to the cafe")
  entry = MemoryEntry("Isabella went to the cafe", emb, "2023-02-13 08:00", 5.0, 0)
  surprise = memory.write(entry)
  assert surprise >= 0
  
  # Read 테스트
  query_emb = encoder.encode("Where did Isabella go?")
  context, buffer = memory.read(query_emb, top_k=3)
  assert context.shape[0] == config.embed_dim
  ```

### Phase 2: Smallville 통합
- [ ] adapter.py 작성 (원본 인터페이스에 정확히 맞춤)
- [ ] persona.py 수정 (메모리 시스템 전환 플래그 추가)
- [ ] **통합 테스트**: Neural Memory로 Smallville 시뮬레이션 10분 실행, 에러 없음 확인
- [ ] 로깅 시스템 연결 확인

### Phase 3: 비교 실험
- [ ] 원본 RAG로 시뮬레이션 3일 실행, 로그 저장
- [ ] Neural Memory로 동일 조건에서 시뮬레이션 3일 실행, 로그 저장
- [ ] 비교 분석 스크립트 실행, 결과 리포트 생성

---

## 9. 예상되는 실패 모드 및 대응 방안

| 증상 | 가능한 원인 | 대응 방안 |
|------|------------|-----------|
| 에이전트가 모든 상황에서 같은 행동만 반복 | Neural memory가 모든 쿼리에 비슷한 context를 반환 | memory_dim 증가, lr 감소, 활성화 함수를 softmax로 변경 |
| Surprise가 전혀 감소하지 않음 | lr이 너무 낮거나, memory_dim이 너무 작아서 용량 부족 | lr 10배 증가 시도, memory_dim 2배로 증가 |
| Surprise가 0 근처에서 시작하여 변화 없음 | Wk, Wv 초기화가 너무 작아 항상 0에 가까운 출력 | 초기화 스케일을 0.1로 변경, Xavier initialization 적용 |
| GPU 메모리가 시간에 따라 증가 | computation graph 누적 또는 all_entries 무한 증가 | backward 후 detach 확인, all_entries 주기적 정리 |
| LLM이 기억 관련 발언을 전혀 하지 않음 | decoder가 관련 기억을 제대로 전달하지 못함 | LLM에 전달되는 프롬프트를 직접 확인, decoder 출력 검증 |
| 원본 코드에서 AttributeError 발생 | adapter가 원본 인터페이스를 완전히 구현하지 않음 | 에러 메시지의 attribute를 adapter에 추가 |

---

## 10. OpenAI API 키 관련

Smallville은 LLM 호출에 OpenAI API를 사용합니다. 실험 전에 다음을 확인:

1. `.env` 파일에 `OPENAI_API_KEY` 설정
2. 비용 추정: 에이전트 3명 × 3일 시뮬레이션 ≈ 약 $5~15 (모델에 따라 다름)
3. 가능하면 `gpt-3.5-turbo` 등 저비용 모델로 먼저 테스트한 후 `gpt-4`로 전환

---

## 부록 A: 참고 논문 및 코드

- Generative Agents (Smallville): https://github.com/joonspk-research/generative_agents
- Titans (Neural Memory): https://arxiv.org/abs/2501.00663
- sentence-transformers: https://www.sbert.net/

## 부록 B: 이 문서의 사용 방법

이 문서를 Claude Code에 전달할 때 다음과 같이 지시하세요:

```
이 설계 계획서를 읽고, Section 8의 체크리스트를 순서대로 실행해줘.
각 단계의 완료 조건을 반드시 확인하고, 문제가 있으면 Section 9의 대응 방안을 참고해.
원본 Smallville 코드를 먼저 완전히 읽고 이해한 후에 구현을 시작해.
```
