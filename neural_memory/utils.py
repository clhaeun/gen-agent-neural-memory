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
