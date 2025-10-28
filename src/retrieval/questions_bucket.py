"""Questions bucket for collecting and managing pending questions.

Handles local storage of questions and coordinates with dashboard uploader for site commits.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from ..dashboard.uploader import DashboardUploader, DashboardConfig

logger = logging.getLogger(__name__)


@dataclass
class PendingQuestion:
    """A question pending external retrieval."""
    question: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    shallow_hits: int = 0  # Number of on-device hits found
    gate_tokens_used: int = 0  # Number of gate tokens consumed
    resolved: bool = False
    resolution_timestamp: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'question': self.question,
            'timestamp': self.timestamp,
            'context': self.context,
            'shallow_hits': self.shallow_hits,
            'gate_tokens_used': self.gate_tokens_used,
            'resolved': self.resolved,
            'resolution_timestamp': self.resolution_timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PendingQuestion':
        """Create from dictionary."""
        return cls(
            question=data['question'],
            timestamp=data['timestamp'],
            context=data.get('context', {}),
            shallow_hits=data.get('shallow_hits', 0),
            gate_tokens_used=data.get('gate_tokens_used', 0),
            resolved=data.get('resolved', False),
            resolution_timestamp=data.get('resolution_timestamp')
        )


class QuestionsBucket:
    """Manages pending questions and coordinates with dashboard."""

    def __init__(self, cache_dir: Path, dashboard_uploader: Optional[DashboardUploader] = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bucket_file = self.cache_dir / 'pending_questions.json'
        self.dashboard_uploader = dashboard_uploader

        # In-memory storage
        self.pending_questions: Dict[str, PendingQuestion] = {}
        self._load_bucket()

        # Gate policy thresholds
        self.min_shallow_hits = 3  # Require ≥3 shallow hits
        self.max_gate_burst = 2    # Max 2 content calls per burst

        logger.info(f"QuestionsBucket initialized with {len(self.pending_questions)} pending questions")

    def _load_bucket(self):
        """Load pending questions from disk."""
        try:
            if self.bucket_file.exists():
                with open(self.bucket_file, 'r') as f:
                    data = json.load(f)
                    for qid, qdata in data.items():
                        self.pending_questions[qid] = PendingQuestion.from_dict(qdata)
        except Exception as e:
            logger.warning(f"Failed to load questions bucket: {e}")

    def _save_bucket(self):
        """Save pending questions to disk."""
        try:
            data = {qid: q.to_dict() for qid, q in self.pending_questions.items()}
            with open(self.bucket_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save questions bucket: {e}")

    def add_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Add a new pending question. Returns question ID."""
        import hashlib
        qid = hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]

        if qid in self.pending_questions:
            # Update existing question
            existing = self.pending_questions[qid]
            existing.context.update(context or {})
            existing.timestamp = time.time()  # Refresh timestamp
        else:
            # Create new question
            self.pending_questions[qid] = PendingQuestion(
                question=question,
                context=context or {}
            )

        self._save_bucket()
        logger.info(f"Added/updated question: {qid}")
        return qid

    def record_shallow_hit(self, question_id: str) -> bool:
        """Record a shallow hit for a question. Returns True if threshold reached."""
        if question_id not in self.pending_questions:
            return False

        question = self.pending_questions[question_id]
        question.shallow_hits += 1

        threshold_reached = question.shallow_hits >= self.min_shallow_hits
        if threshold_reached:
            logger.info(f"Question {question_id} reached shallow hit threshold ({question.shallow_hits})")

        self._save_bucket()
        return threshold_reached

    def can_gate_burst(self, question_id: str) -> bool:
        """Check if question can trigger a gate burst."""
        if question_id not in self.pending_questions:
            return False

        question = self.pending_questions[question_id]
        return (
            question.shallow_hits >= self.min_shallow_hits and
            question.gate_tokens_used < self.max_gate_burst and
            not question.resolved
        )

    def record_gate_usage(self, question_id: str) -> bool:
        """Record usage of a gate token. Returns True if still can use more."""
        if question_id not in self.pending_questions:
            return False

        question = self.pending_questions[question_id]
        question.gate_tokens_used += 1

        can_use_more = question.gate_tokens_used < self.max_gate_burst
        if not can_use_more:
            logger.info(f"Question {question_id} exhausted gate burst limit ({question.gate_tokens_used})")

        self._save_bucket()
        return can_use_more

    def resolve_question(self, question_id: str):
        """Mark a question as resolved."""
        if question_id in self.pending_questions:
            question = self.pending_questions[question_id]
            question.resolved = True
            question.resolution_timestamp = time.time()
            self._save_bucket()
            logger.info(f"Resolved question: {question_id}")

    async def commit_to_dashboard(self):
        """Commit pending questions to dashboard site."""
        if not self.dashboard_uploader:
            logger.debug("No dashboard uploader configured")
            return

        # Prepare questions data for upload
        questions_data = {
            'timestamp': time.time(),
            'pending_count': len(self.pending_questions),
            'questions': [
                q.to_dict() for q in self.pending_questions.values()
                if not q.resolved
            ]
        }

        # Convert to JSON and upload
        json_content = json.dumps(questions_data, indent=2).encode('utf-8')
        await self.dashboard_uploader.queue_file(
            'faq/pending_questions.json',
            json_content
        )

        logger.info(f"Committed {len(questions_data['questions'])} pending questions to dashboard")

    def get_pending_questions(self) -> List[PendingQuestion]:
        """Get all pending (unresolved) questions."""
        return [q for q in self.pending_questions.values() if not q.resolved]

    def get_question(self, question_id: str) -> Optional[PendingQuestion]:
        """Get a specific question by ID."""
        return self.pending_questions.get(question_id)

    def cleanup_resolved(self, max_age_days: int = 30):
        """Clean up old resolved questions."""
        cutoff = time.time() - (max_age_days * 24 * 3600)

        to_remove = []
        for qid, question in self.pending_questions.items():
            if question.resolved and question.resolution_timestamp:
                if question.resolution_timestamp < cutoff:
                    to_remove.append(qid)

        for qid in to_remove:
            del self.pending_questions[qid]

        if to_remove:
            self._save_bucket()
            logger.info(f"Cleaned up {len(to_remove)} old resolved questions")

    def get_stats(self) -> Dict[str, Any]:
        """Get bucket statistics."""
        pending = self.get_pending_questions()
        resolved = [q for q in self.pending_questions.values() if q.resolved]

        return {
            'total_questions': len(self.pending_questions),
            'pending_count': len(pending),
            'resolved_count': len(resolved),
            'avg_shallow_hits': sum(q.shallow_hits for q in pending) / max(1, len(pending)),
            'avg_gate_usage': sum(q.gate_tokens_used for q in pending) / max(1, len(pending)),
            'threshold_ready': sum(1 for q in pending if q.shallow_hits >= self.min_shallow_hits)
        }