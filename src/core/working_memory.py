"""
WorkingMemory: Hippocampal-equivalent fast, limited memory
"""
from typing import List, Optional, Dict
from datetime import datetime
from collections import deque
import json

from .models import Episode, MemoryState, ImportanceVector
from .config import WORKING_MEMORY_CAPACITY
from .time_utils import ensure_utc, utc_now


class WorkingMemory:
    """
    Hippocampal-equivalent: Fast access, limited capacity, temporal encoding.
    Like human short-term memory, holds ~7 items (Miller's Law).
    """

    def __init__(self, capacity: int = WORKING_MEMORY_CAPACITY):
        self.capacity = capacity
        self.episodes: deque = deque(maxlen=capacity)
        self._access_times = {}  # Track access for LRU

    def store(self, episode: Episode) -> None:
        """
        Store a new episode in working memory.
        If at capacity, oldest episode is pushed out.
        """
        # Update importance based on current time
        episode.timestamp = utc_now()
        if not episode.state:
            episode.state = MemoryState.ACTIVE

        self.episodes.append(episode)
        self._access_times[episode.id] = utc_now()

    def retrieve(self, query: str = None, limit: int = 5) -> List[Episode]:
        """
        Retrieve episodes from working memory.

        Args:
            query: Optional query to filter episodes
            limit: Maximum episodes to return

        Returns:
            List of most relevant episodes
        """
        if not self.episodes:
            return []

        episodes = list(self.episodes)

        # Sort by recency and importance
        episodes.sort(
            key=lambda e: (
                e.timestamp,
                e.importance.overall
            ),
            reverse=True
        )

        return episodes[:limit]

    def get_recent(self, n: int = 3) -> List[Episode]:
        """Get the n most recent episodes"""
        return list(self.episodes)[-n:]

    def get_all(self) -> List[Episode]:
        """Get all episodes in working memory"""
        return list(self.episodes)

    def is_full(self) -> bool:
        """Check if working memory is at capacity"""
        return len(self.episodes) >= self.capacity

    def size(self) -> int:
        """Current number of episodes"""
        return len(self.episodes)

    def update_episode(self, episode_id: str, updates: Dict) -> bool:
        """Update an existing episode"""
        for i, episode in enumerate(self.episodes):
            if episode.id == episode_id:
                # Update fields
                for key, value in updates.items():
                    if hasattr(episode, key):
                        setattr(episode, key, value)
                return True
        return False

    def remove_episode(self, episode_id: str) -> bool:
        """Remove specific episode from working memory"""
        for i, episode in enumerate(self.episodes):
            if episode.id == episode_id:
                del self.episodes[i]
                self._access_times.pop(episode_id, None)
                return True
        return False

    def mark_consolidating(self, episode_ids: List[str]):
        """Mark episodes as being consolidated to long-term memory"""
        for episode in self.episodes:
            if episode.id in episode_ids:
                episode.state = MemoryState.CONSOLIDATING

    def get_by_state(self, state: MemoryState) -> List[Episode]:
        """Get all episodes in a specific state"""
        return [e for e in self.episodes if e.state == state]

    def clear(self):
        """Clear all episodes (used during sleep)"""
        self.episodes.clear()
        self._access_times.clear()

    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            'episodes': [
                {
                    'id': e.id,
                    'timestamp': e.timestamp.isoformat(),
                    'concept_ids': e.concept_ids,
                    'raw_content': e.raw_content,
                    'context': e.context,
                    'importance': e.importance.model_dump(),
                    'state': e.state.value,
                    'source': e.source
                }
                for e in self.episodes
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkingMemory':
        """Deserialize from storage"""
        wm = cls()
        for ep_data in data.get('episodes', []):
            ep = Episode(
                id=ep_data['id'],
                timestamp=ensure_utc(ep_data['timestamp']) or utc_now(),
                concept_ids=ep_data.get('concept_ids', []),
                raw_content=ep_data['raw_content'],
                context=ep_data.get('context', {}),
                importance=ImportanceVector(**ep_data.get('importance', {})),
                state=MemoryState(ep_data.get('state', 'active')),
                source=ep_data.get('source', 'user')
            )
            wm.episodes.append(ep)
        return wm
