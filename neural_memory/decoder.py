import torch
from typing import List

from .memory_module import MemoryEntry


class MemoryDecoder:
    """
    Neural Memory의 출력(맥락 벡터)을 텍스트로 변환.

    원본 설계 대비 수정 사항:
    - [FIX] all_entries 전체 스캔(O(n)) 대신 search_entries(sliding window)만 검색
    - [FIX] 임베딩을 미리 stack하여 배치 cosine similarity 계산 → 반복 unsqueeze 제거
    """

    def decode(
        self,
        context_vector: torch.Tensor,
        buffer_entries: List[MemoryEntry],
        search_entries: List[MemoryEntry],
        top_k: int = 5
    ) -> List[str]:
        """
        맥락 벡터를 기반으로 관련 기억 텍스트를 반환.

        Args:
            context_vector: neural memory에서 읽은 벡터
            buffer_entries: 최근 버퍼에서 선택된 기억들
            search_entries: 검색 대상 엔트리 (sliding window로 제한됨)
            top_k: 반환할 기억 수
        """
        results = []
        seen_ids: set = set()

        # 1. Context vector 기반 검색 — 배치 cosine similarity
        if len(search_entries) > 0:
            device = context_vector.device
            # [FIX] 임베딩을 한번에 stack하여 배치 연산
            entry_embeddings = torch.stack(
                [e.embedding.to(device) for e in search_entries]
            )  # (N, d)
            ctx = context_vector.unsqueeze(0)  # (1, d)
            similarities = torch.cosine_similarity(ctx, entry_embeddings, dim=1)  # (N,)

            # top_k 인덱스 추출
            k = min(top_k, len(search_entries))
            _, top_indices = torch.topk(similarities, k)

            for idx in top_indices.tolist():
                entry = search_entries[idx]
                if entry.node_id not in seen_ids:
                    results.append(entry.description)
                    seen_ids.add(entry.node_id)

        # 2. Buffer entries 추가
        for entry in buffer_entries:
            if entry.node_id not in seen_ids:
                results.append(entry.description)
                seen_ids.add(entry.node_id)

        return results[:top_k]
