"""
NeuralMemoryAdapter: AssociativeMemory의 완전한 drop-in replacement.

원본 AssociativeMemory의 모든 public 메서드와 속성을 동일하게 구현하면서,
내부적으로 NeuralMemoryModule을 사용하여 neural memory에 경험을 기록하고
context vector 기반 검색을 추가로 수행.

전략:
- 원본의 모든 자료구조(seq_event, kw_to_event, embeddings 등)를 그대로 유지
  → retrieve.py, reflect.py, plan.py 등에서 직접 속성 접근하는 코드와 호환
- add_event/add_thought/add_chat 시 원본 로직 + neural memory write 병행
- new_retrieve()에서의 relevance 계산 시 neural memory context vector를
  추가 시그널로 활용 (retrieve.py 수정 시)
"""

import json
import datetime
import os
import torch
from typing import Optional, List, Tuple, Any, Set

from .config import NeuralMemoryConfig
from .memory_module import NeuralMemoryModule


class ConceptNode:
    """원본 ConceptNode와 동일한 구조."""
    def __init__(self,
                 node_id, node_count, type_count, node_type, depth,
                 created, expiration,
                 s, p, o,
                 description, embedding_key, poignancy, keywords, filling):
        self.node_id = node_id
        self.node_count = node_count
        self.type_count = type_count
        self.type = node_type  # thought / event / chat
        self.depth = depth

        self.created = created
        self.expiration = expiration
        self.last_accessed = self.created

        self.subject = s
        self.predicate = p
        self.object = o

        self.description = description
        self.embedding_key = embedding_key
        self.poignancy = poignancy
        self.keywords = keywords
        self.filling = filling

    def spo_summary(self):
        return (self.subject, self.predicate, self.object)


class NeuralMemoryAdapter:
    """
    AssociativeMemory와 완전히 동일한 인터페이스를 제공.

    내부 동작:
    1. 원본과 동일한 자료구조(seq_event, kw_to_event, embeddings 등) 유지
    2. add_event/thought/chat 시 neural memory에도 임베딩을 write
    3. retrieve 시 neural memory의 context vector를 활용한 보조 점수 제공 가능
    """

    def __init__(self, f_saved: str, neural_config: Optional[NeuralMemoryConfig] = None):
        """
        원본 AssociativeMemory.__init__과 동일한 시그니처 + neural_config 추가.

        Args:
            f_saved: 저장된 메모리 파일 경로 (nodes.json, embeddings.json 등)
            neural_config: NeuralMemoryConfig (None이면 기본값 사용)
        """
        # === 원본과 동일한 자료구조 ===
        self.id_to_node = dict()
        self.seq_event = []
        self.seq_thought = []
        self.seq_chat = []
        self.kw_to_event = dict()
        self.kw_to_thought = dict()
        self.kw_to_chat = dict()
        self.kw_strength_event = dict()
        self.kw_strength_thought = dict()
        self.embeddings = json.load(open(f_saved + "/embeddings.json"))

        # === Neural Memory 초기화 ===
        if neural_config is None:
            neural_config = NeuralMemoryConfig()
        self.neural_config = neural_config
        self.neural_memory = NeuralMemoryModule(neural_config)
        self.surprise_log: List[float] = []

        # === 원본과 동일한 노드 로딩 ===
        nodes_load = json.load(open(f_saved + "/nodes.json"))
        for count in range(len(nodes_load.keys())):
            node_id = f"node_{str(count+1)}"
            node_details = nodes_load[node_id]

            node_count = node_details["node_count"]
            type_count = node_details["type_count"]
            node_type = node_details["type"]
            depth = node_details["depth"]

            created = datetime.datetime.strptime(node_details["created"],
                                                 '%Y-%m-%d %H:%M:%S')
            expiration = None
            if node_details["expiration"]:
                expiration = datetime.datetime.strptime(
                    node_details["expiration"], '%Y-%m-%d %H:%M:%S')

            s = node_details["subject"]
            p = node_details["predicate"]
            o = node_details["object"]

            description = node_details["description"]
            embedding_pair = (node_details["embedding_key"],
                              self.embeddings[node_details["embedding_key"]])
            poignancy = node_details["poignancy"]
            keywords = set(node_details["keywords"])
            filling = node_details["filling"]

            if node_type == "event":
                self.add_event(created, expiration, s, p, o,
                               description, keywords, poignancy,
                               embedding_pair, filling)
            elif node_type == "chat":
                self.add_chat(created, expiration, s, p, o,
                              description, keywords, poignancy,
                              embedding_pair, filling)
            elif node_type == "thought":
                self.add_thought(created, expiration, s, p, o,
                                 description, keywords, poignancy,
                                 embedding_pair, filling)

        kw_strength_load = json.load(open(f_saved + "/kw_strength.json"))
        if kw_strength_load["kw_strength_event"]:
            self.kw_strength_event = kw_strength_load["kw_strength_event"]
        if kw_strength_load["kw_strength_thought"]:
            self.kw_strength_thought = kw_strength_load["kw_strength_thought"]

    def save(self, out_json):
        """원본과 동일한 save 로직."""
        r = dict()
        for count in range(len(self.id_to_node.keys()), 0, -1):
            node_id = f"node_{str(count)}"
            node = self.id_to_node[node_id]

            r[node_id] = dict()
            r[node_id]["node_count"] = node.node_count
            r[node_id]["type_count"] = node.type_count
            r[node_id]["type"] = node.type
            r[node_id]["depth"] = node.depth

            r[node_id]["created"] = node.created.strftime('%Y-%m-%d %H:%M:%S')
            r[node_id]["expiration"] = None
            if node.expiration:
                r[node_id]["expiration"] = node.expiration.strftime(
                    '%Y-%m-%d %H:%M:%S')

            r[node_id]["subject"] = node.subject
            r[node_id]["predicate"] = node.predicate
            r[node_id]["object"] = node.object

            r[node_id]["description"] = node.description
            r[node_id]["embedding_key"] = node.embedding_key
            r[node_id]["poignancy"] = node.poignancy
            r[node_id]["keywords"] = list(node.keywords)
            r[node_id]["filling"] = node.filling

        with open(out_json + "/nodes.json", "w") as outfile:
            json.dump(r, outfile)

        r = dict()
        r["kw_strength_event"] = self.kw_strength_event
        r["kw_strength_thought"] = self.kw_strength_thought
        with open(out_json + "/kw_strength.json", "w") as outfile:
            json.dump(r, outfile)

        with open(out_json + "/embeddings.json", "w") as outfile:
            json.dump(self.embeddings, outfile)

        # Neural memory 상태도 저장
        neural_path = out_json + "/neural_memory.pt"
        torch.save({
            'model_state': self.neural_memory.state_dict(),
            'write_count': self.neural_memory.write_count,
            'total_surprise': self.neural_memory.total_surprise,
            'surprise_log': self.surprise_log,
        }, neural_path)

    def add_event(self, created, expiration, s, p, o,
                  description, keywords, poignancy,
                  embedding_pair, filling):
        """원본과 동일 + neural memory write."""
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_event) + 1
        node_type = "event"
        node_id = f"node_{str(node_count)}"
        depth = 0

        if "(" in description:
            description = (" ".join(description.split()[:3])
                           + " "
                           + description.split("(")[-1][:-1])

        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                           created, expiration,
                           s, p, o,
                           description, embedding_pair[0],
                           poignancy, keywords, filling)

        self.seq_event[0:0] = [node]
        kw_lower = [i.lower() for i in keywords]
        for kw in kw_lower:
            if kw in self.kw_to_event:
                self.kw_to_event[kw][0:0] = [node]
            else:
                self.kw_to_event[kw] = [node]
        self.id_to_node[node_id] = node

        if f"{p} {o}" != "is idle":
            for kw in kw_lower:
                if kw in self.kw_strength_event:
                    self.kw_strength_event[kw] += 1
                else:
                    self.kw_strength_event[kw] = 1

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        # Neural memory write
        self._neural_write(embedding_pair[1])

        return node

    def add_thought(self, created, expiration, s, p, o,
                    description, keywords, poignancy,
                    embedding_pair, filling):
        """원본과 동일 + neural memory write."""
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_thought) + 1
        node_type = "thought"
        node_id = f"node_{str(node_count)}"
        depth = 1
        try:
            if filling:
                depth += max([self.id_to_node[i].depth for i in filling])
        except:
            pass

        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                           created, expiration,
                           s, p, o,
                           description, embedding_pair[0],
                           poignancy, keywords, filling)

        self.seq_thought[0:0] = [node]
        kw_lower = [i.lower() for i in keywords]
        for kw in kw_lower:
            if kw in self.kw_to_thought:
                self.kw_to_thought[kw][0:0] = [node]
            else:
                self.kw_to_thought[kw] = [node]
        self.id_to_node[node_id] = node

        if f"{p} {o}" != "is idle":
            for kw in kw_lower:
                if kw in self.kw_strength_thought:
                    self.kw_strength_thought[kw] += 1
                else:
                    self.kw_strength_thought[kw] = 1

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        # Neural memory write
        self._neural_write(embedding_pair[1])

        return node

    def add_chat(self, created, expiration, s, p, o,
                 description, keywords, poignancy,
                 embedding_pair, filling):
        """원본과 동일 + neural memory write."""
        node_count = len(self.id_to_node.keys()) + 1
        type_count = len(self.seq_chat) + 1
        node_type = "chat"
        node_id = f"node_{str(node_count)}"
        depth = 0

        node = ConceptNode(node_id, node_count, type_count, node_type, depth,
                           created, expiration,
                           s, p, o,
                           description, embedding_pair[0],
                           poignancy, keywords, filling)

        self.seq_chat[0:0] = [node]
        kw_lower = [i.lower() for i in keywords]
        for kw in kw_lower:
            if kw in self.kw_to_chat:
                self.kw_to_chat[kw][0:0] = [node]
            else:
                self.kw_to_chat[kw] = [node]
        self.id_to_node[node_id] = node

        self.embeddings[embedding_pair[0]] = embedding_pair[1]

        # Neural memory write
        self._neural_write(embedding_pair[1])

        return node

    def _neural_write(self, embedding_list):
        """임베딩 리스트를 tensor로 변환하여 neural memory에 기록."""
        emb_tensor = torch.tensor(embedding_list, dtype=torch.float32)
        surprise = self.neural_memory.write(emb_tensor)
        self.surprise_log.append(surprise)

    def neural_read(self, query_embedding_list) -> torch.Tensor:
        """
        Neural memory에서 context vector를 읽어 반환.
        new_retrieve()에서 추가 relevance 시그널로 활용 가능.

        Args:
            query_embedding_list: list of floats (OpenAI embedding)
        Returns:
            context_vector: tensor shape (d,)
        """
        q = torch.tensor(query_embedding_list, dtype=torch.float32)
        return self.neural_memory.read(q)

    # === 원본과 동일한 검색/조회 메서드 ===

    def get_summarized_latest_events(self, retention):
        ret_set = set()
        for e_node in self.seq_event[:retention]:
            ret_set.add(e_node.spo_summary())
        return ret_set

    def get_str_seq_events(self):
        ret_str = ""
        for count, event in enumerate(self.seq_event):
            ret_str += f'{"Event", len(self.seq_event) - count, ": ", event.spo_summary(), " -- ", event.description}\n'
        return ret_str

    def get_str_seq_thoughts(self):
        ret_str = ""
        for count, event in enumerate(self.seq_thought):
            ret_str += f'{"Thought", len(self.seq_thought) - count, ": ", event.spo_summary(), " -- ", event.description}'
        return ret_str

    def get_str_seq_chats(self):
        ret_str = ""
        for count, event in enumerate(self.seq_chat):
            ret_str += f"with {event.object.content} ({event.description})\n"
            ret_str += f'{event.created.strftime("%B %d, %Y, %H:%M:%S")}\n'
            for row in event.filling:
                ret_str += f"{row[0]}: {row[1]}\n"
        return ret_str

    def retrieve_relevant_thoughts(self, s_content, p_content, o_content):
        contents = [s_content, p_content, o_content]
        ret = []
        for i in contents:
            if i in self.kw_to_thought:
                ret += self.kw_to_thought[i.lower()]
        return set(ret)

    def retrieve_relevant_events(self, s_content, p_content, o_content):
        contents = [s_content, p_content, o_content]
        ret = []
        for i in contents:
            if i in self.kw_to_event:
                ret += self.kw_to_event[i]
        return set(ret)

    def get_last_chat(self, target_persona_name):
        if target_persona_name.lower() in self.kw_to_chat:
            return self.kw_to_chat[target_persona_name.lower()][0]
        else:
            return False

    def get_neural_stats(self):
        """Neural memory 통계 반환 (디버깅용)."""
        return self.neural_memory.get_stats()
