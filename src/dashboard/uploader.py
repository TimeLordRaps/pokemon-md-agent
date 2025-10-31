"""Dashboard uploader for PMD-Red Agent.

Handles batching, rate limiting, and uploading of dashboard artifacts to GitHub Pages.
Supports multiple upload modes: git push, GitHub Contents API, and no-op.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import base64
import hashlib

import requests

logger = logging.getLogger(__name__)


class UploadMode(Enum):
    """Upload mode for dashboard artifacts."""
    GIT_PUSH = "git_push"
    GITHUB_API = "github_api"
    NO_OP = "no_op"


class UploadPriority(Enum):
    """Upload priority levels."""
    CRITICAL = 0  # Highest priority, immediate upload
    NORMAL = 1    # Standard priority
    BACKGROUND = 2  # Lowest priority, batched with others


@dataclass
class FileBatch:
    """A batch of files to upload."""
    files: Dict[str, bytes] = field(default_factory=dict)
    total_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    priority: UploadPriority = UploadPriority.NORMAL

    def add_file(self, path: str, content: bytes, priority: UploadPriority = UploadPriority.NORMAL) -> bool:
        """Add a file to the batch. Returns True if added, False if would exceed limits."""
        file_size = len(content)
        if self.total_bytes + file_size > 8 * 1024 * 1024:  # 8 MB limit
            return False
        self.files[path] = content
        self.total_bytes += file_size
        # Update batch priority to highest priority of files in batch
        if priority.value < self.priority.value:
            self.priority = priority
        return True

    def is_empty(self) -> bool:
        return len(self.files) == 0

    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class TrajectoryBatch:
    """A batch of trajectory data for upload."""
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    batch_id: str = field(default_factory=lambda: f"traj_{int(time.time())}")

    def add_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """Add a trajectory to the batch. Returns True if added."""
        traj_bytes = len(json.dumps(trajectory).encode())
        if self.total_bytes + traj_bytes > 50 * 1024 * 1024:  # 50 MB limit
            return False
        self.trajectories.append(trajectory)
        self.total_bytes += traj_bytes
        return True

    def is_empty(self) -> bool:
        return len(self.trajectories) == 0

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary for persistence."""
        return {
            'trajectories': self.trajectories,
            'total_bytes': self.total_bytes,
            'created_at': self.created_at,
            'batch_id': self.batch_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryBatch':
        """Create batch from dictionary."""
        batch = cls()
        batch.trajectories = data['trajectories']
        batch.total_bytes = data['total_bytes']
        batch.created_at = data['created_at']
        batch.batch_id = data.get('batch_id', f"recovered_{int(time.time())}")
        return batch


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""
    capacity: int  # Max tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.time()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_tokens(self, tokens: int) -> float:
        """Time in seconds until we have enough tokens."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class DashboardConfig:
    """Configuration for dashboard uploading."""
    enabled: bool = True
    branch: str = "pages"
    site_root: str = "docs"
    flush_seconds: float = 30.0
    max_batch_bytes: int = 8 * 1024 * 1024  # 8 MB
    max_files_per_minute: int = 30
    github_token: Optional[str] = None
    github_repo: Optional[str] = None  # format: "owner/repo"
    
    # Trajectory batch settings
    trajectory_flush_seconds: float = 60.0  # Flush trajectories every minute
    max_trajectory_batch_bytes: int = 10 * 1024 * 1024  # 10 MB for trajectories
    max_trajectory_batch_count: int = 100  # Max trajectories per batch


class DashboardUploader:
    """Handles uploading dashboard artifacts with batching and rate limiting."""

    def __init__(self, config: DashboardConfig, cache_dir: Path):
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Upload mode detection
        self.upload_mode = self._detect_upload_mode()

        # Rate limiters
        self.file_limiter = RateLimiter(
            capacity=self.config.max_files_per_minute,
            refill_rate=self.config.max_files_per_minute / 60.0
        )
        self.build_limiter = RateLimiter(
            capacity=10,  # 10 builds per hour
            refill_rate=10 / 3600.0
        )

        # Priority queues for different upload priorities
        self.priority_batches = {
            UploadPriority.CRITICAL: FileBatch(priority=UploadPriority.CRITICAL),
            UploadPriority.NORMAL: FileBatch(priority=UploadPriority.NORMAL),
            UploadPriority.BACKGROUND: FileBatch(priority=UploadPriority.BACKGROUND),
        }
        self.last_flush = time.time()

        # Trajectory batching
        self.current_trajectory_batch = TrajectoryBatch()
        self.last_trajectory_flush = time.time()
        self.pending_trajectory_batches: List[TrajectoryBatch] = []
        self._load_pending_batches()

        # Stats
        self.stats = {
            'files_uploaded': 0,
            'bytes_uploaded': 0,
            'batches_flushed': 0,
            'rate_limit_hits': 0,
            'builds_triggered': 0,
            'trajectories_uploaded': 0,
            'trajectory_batches_flushed': 0,
            'trajectory_bytes_uploaded': 0
        }

        logger.info(f"Dashboard uploader initialized with mode: {self.upload_mode}")

    def _load_pending_batches(self):
        """Load any pending trajectory batches from disk after crash."""
        pending_dir = self.cache_dir / "pending_batches"
        if not pending_dir.exists():
            return

        try:
            for batch_file in pending_dir.glob("*.json"):
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                    batch = TrajectoryBatch.from_dict(batch_data)
                    self.pending_trajectory_batches.append(batch)
                    logger.info(f"Recovered pending batch: {batch.batch_id}")
                except Exception as e:
                    logger.warning(f"Failed to load pending batch {batch_file}: {e}")
                    # Remove corrupted file
                    batch_file.unlink()
        except Exception as e:
            logger.error(f"Failed to load pending batches: {e}")

    def _save_pending_batch(self, batch: TrajectoryBatch):
        """Save a pending batch to disk for crash recovery."""
        pending_dir = self.cache_dir / "pending_batches"
        pending_dir.mkdir(exist_ok=True)

        batch_file = pending_dir / f"{batch.batch_id}.json"
        try:
            with open(batch_file, 'w') as f:
                json.dump(batch.to_dict(), f)
        except Exception as e:
            logger.error(f"Failed to save pending batch {batch.batch_id}: {e}")

    def _remove_pending_batch(self, batch: TrajectoryBatch):
        """Remove a completed batch from disk."""
        pending_dir = self.cache_dir / "pending_batches"
        batch_file = pending_dir / f"{batch.batch_id}.json"
        try:
            if batch_file.exists():
                batch_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove pending batch file {batch_file}: {e}")

    async def queue_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """Queue a trajectory for upload. Returns True if queued successfully."""
        if self.upload_mode == UploadMode.NO_OP:
            return True

        # Try to add to current batch
        if not self.current_trajectory_batch.add_trajectory(trajectory):
            # Batch is full, flush it first
            await self._flush_trajectory_batch()
            # Try again with new batch
            if not self.current_trajectory_batch.add_trajectory(trajectory):
                logger.error("Trajectory too large for empty batch")
                return False

        # Check if we should flush based on time, size, or count
        should_flush = (
            self.current_trajectory_batch.age_seconds() >= self.config.trajectory_flush_seconds or
            self.current_trajectory_batch.total_bytes >= self.config.max_trajectory_batch_bytes or
            len(self.current_trajectory_batch.trajectories) >= self.config.max_trajectory_batch_count
        )

        if should_flush:
            await self._flush_trajectory_batch()

        return True

    async def _flush_trajectory_batch(self):
        """Flush the current trajectory batch to the dashboard."""
        if self.current_trajectory_batch.is_empty():
            return

        # Save batch for crash recovery
        self._save_pending_batch(self.current_trajectory_batch)
        self.pending_trajectory_batches.append(self.current_trajectory_batch)

        # Convert trajectories to JSONL format
        jsonl_content = "\n".join(json.dumps(traj) for traj in self.current_trajectory_batch.trajectories)

        # Create filename with timestamp
        timestamp = int(self.current_trajectory_batch.created_at)
        filename = f"trajectories_{timestamp}_{self.current_trajectory_batch.batch_id}.jsonl"

        try:
            # Queue as regular file for upload
            success = await self.queue_file(filename, jsonl_content.encode())

            if success:
                self.stats['trajectory_batches_flushed'] += 1
                self.stats['trajectories_uploaded'] += len(self.current_trajectory_batch.trajectories)
                self.stats['trajectory_bytes_uploaded'] += self.current_trajectory_batch.total_bytes

                logger.info(f"Flushed trajectory batch: {len(self.current_trajectory_batch.trajectories)} trajectories, {self.current_trajectory_batch.total_bytes} bytes")

                # Remove from pending batches
                self.pending_trajectory_batches.remove(self.current_trajectory_batch)
                self._remove_pending_batch(self.current_trajectory_batch)

            else:
                logger.error("Failed to queue trajectory batch file")

        except Exception as e:
            logger.error(f"Failed to flush trajectory batch: {e}")
            # Keep batch for retry
            return

        # Reset batch
        self.current_trajectory_batch = TrajectoryBatch()
        self.last_trajectory_flush = time.time()

    async def retry_pending_batches(self):
        """Retry uploading any pending trajectory batches after crash recovery."""
        if not self.pending_trajectory_batches:
            return

        logger.info(f"Retrying {len(self.pending_trajectory_batches)} pending trajectory batches")

        for batch in self.pending_trajectory_batches[:]:  # Copy to avoid modification during iteration
            try:
                # Convert trajectories to JSONL format
                jsonl_content = "\n".join(json.dumps(traj) for traj in batch.trajectories)
                timestamp = int(batch.created_at)
                filename = f"trajectories_{timestamp}_{batch.batch_id}.jsonl"

                success = await self.queue_file(filename, jsonl_content.encode())

                if success:
                    self.stats['trajectory_batches_flushed'] += 1
                    self.stats['trajectories_uploaded'] += len(batch.trajectories)
                    self.stats['trajectory_bytes_uploaded'] += batch.total_bytes

                    logger.info(f"Recovered trajectory batch: {batch.batch_id}")

                    # Remove from pending
                    self.pending_trajectory_batches.remove(batch)
                    self._remove_pending_batch(batch)

            except Exception as e:
                logger.error(f"Failed to retry pending batch {batch.batch_id}: {e}")

    def _detect_upload_mode(self) -> UploadMode:
        """Detect the best upload mode based on environment."""
        if not self.config.enabled:
            return UploadMode.NO_OP

        # Check for git repository
        if self._is_git_repo():
            return UploadMode.GIT_PUSH

        # Check for GitHub API access
        if self.config.github_token and self.config.github_repo:
            return UploadMode.GITHUB_API

        # Fallback to no-op
        logger.warning("No suitable upload mode detected, falling back to no-op")
        return UploadMode.NO_OP

    def _is_git_repo(self) -> bool:
        """Check if we're in a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True,
                text=True,
                cwd=self.cache_dir.parent
            )
            return result.returncode == 0
        except Exception:
            return False

    async def queue_file(self, relative_path: str, content: bytes, priority: UploadPriority = UploadPriority.NORMAL) -> bool:
        """Queue a file for upload. Returns True if queued successfully."""
        if self.upload_mode == UploadMode.NO_OP:
            # In NO_OP mode, still queue files for testing purposes
            pass

        # Check file size limits (avoid LFS)
        if len(content) > 50 * 1024 * 1024:  # 50 MB
            logger.warning(f"File {relative_path} too large ({len(content)} bytes), skipping")
            return False

        # Try to add to appropriate priority batch
        batch = self.priority_batches[priority]
        if not batch.add_file(relative_path, content, priority):
            # Batch is full, flush it first
            await self._flush_batch_for_priority(priority)
            # Try again with new batch
            batch = self.priority_batches[priority] = FileBatch(priority=priority)
            if not batch.add_file(relative_path, content, priority):
                logger.error(f"File {relative_path} too large for empty batch")
                return False

        # Check if we should flush based on time, size, or priority
        should_flush = (
            batch.age_seconds() >= self._get_flush_seconds_for_priority(priority) or
            batch.total_bytes >= self.config.max_batch_bytes
        )

        if should_flush:
            await self._flush_batch_for_priority(priority)

        return True

    def _get_flush_seconds_for_priority(self, priority: UploadPriority) -> float:
        """Get flush interval based on priority."""
        if priority == UploadPriority.CRITICAL:
            return 1.0  # Flush critical uploads quickly
        elif priority == UploadPriority.NORMAL:
            return self.config.flush_seconds
        else:  # BACKGROUND
            return self.config.flush_seconds * 4  # Flush background slower

    async def _flush_batch_for_priority(self, priority: UploadPriority):
        """Flush batch for a specific priority."""
        batch = self.priority_batches[priority]
        if batch.is_empty():
            return

        await self._flush_specific_batch(batch)

        # Reset batch
        self.priority_batches[priority] = FileBatch(priority=priority)
        self.last_flush = time.time()

    async def _flush_specific_batch(self, batch: FileBatch):
        """Flush a specific batch to the dashboard."""
        # Check rate limits
        file_count = len(batch.files)
        if not self.file_limiter.consume(file_count):
            wait_time = self.file_limiter.time_until_tokens(file_count)
            logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            self.stats['rate_limit_hits'] += 1

        # Check build budget
        if not self.build_limiter.consume(1):
            wait_time = self.build_limiter.time_until_tokens(1)
            logger.warning(f"Build budget exceeded, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

        try:
            if self.upload_mode == UploadMode.GIT_PUSH:
                await self._flush_via_git_batch(batch)
            elif self.upload_mode == UploadMode.GITHUB_API:
                await self._flush_via_api_batch(batch)
            else:
                # NO_OP - just clear batch
                pass

            self.stats['batches_flushed'] += 1
            self.stats['files_uploaded'] += file_count
            self.stats['bytes_uploaded'] += batch.total_bytes
            self.stats['builds_triggered'] += 1

            logger.info(f"Flushed {batch.priority.name} batch: {file_count} files, {batch.total_bytes} bytes")

        except Exception as e:
            logger.error(f"Failed to flush {batch.priority.name} batch: {e}")
            # Keep batch for retry on next flush
            return

    async def _flush_batch(self):
        """Flush all priority batches in order (critical first)."""
        for priority in [UploadPriority.CRITICAL, UploadPriority.NORMAL, UploadPriority.BACKGROUND]:
            await self._flush_batch_for_priority(priority)

    async def _flush_via_git_batch(self, batch: FileBatch):
        """Flush batch via git push to pages branch."""
        # Create temporary directory for batch
        batch_dir = self.cache_dir / f"batch_{int(time.time())}_{batch.priority.name.lower()}"
        batch_dir.mkdir()

        try:
            # Write files to batch directory
            for rel_path, content in batch.files.items():
                file_path = batch_dir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(content)

            # Copy to site root in repo
            repo_root = self._find_repo_root()
            site_dir = repo_root / self.config.site_root
            site_dir.mkdir(exist_ok=True)

            # Use rsync or similar to copy (simplified - just copy for now)
            import shutil
            for rel_path in batch.files:
                src = batch_dir / rel_path
                dst = site_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            # Git add, commit, push
            await self._run_git_command(['add', '.'], cwd=site_dir)
            await self._run_git_command(['commit', '-m', f'Dashboard update ({batch.priority.name}): {len(batch.files)} files'], cwd=site_dir)
            await self._run_git_command(['push', 'origin', self.config.branch], cwd=site_dir)

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(batch_dir, ignore_errors=True)

    async def _flush_via_git(self):
        """Flush batch via git push to pages branch (legacy method)."""
        await self._flush_via_git_batch(self.current_batch)

    async def _flush_via_api(self):
        """Flush batch via GitHub Contents API."""
        if not self.config.github_token or not self.config.github_repo:
            raise ValueError("GitHub token and repo required for API mode")

        headers = {
            'Authorization': f'token {self.config.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        base_url = f'https://api.github.com/repos/{self.config.github_repo}/contents'

        for rel_path, content in self.current_batch.files.items():
            file_path = f'{self.config.site_root}/{rel_path}'

            # Get current file SHA if it exists
            sha = await self._get_file_sha(file_path, headers, base_url)

            # Prepare request
            data = {
                'message': f'Update {rel_path}',
                'content': base64.b64encode(content).decode(),
                'branch': self.config.branch
            }
            if sha:
                data['sha'] = sha

            # Upload file
            url = f'{base_url}/{file_path}'
            response = requests.put(url, headers=headers, json=data)
            response.raise_for_status()

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

    async def _get_file_sha(self, file_path: str, headers: dict, base_url: str) -> Optional[str]:
        """Get SHA of existing file."""
        url = f'{base_url}/{file_path}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['sha']
        return None

    def _find_repo_root(self) -> Path:
        """Find the git repository root."""
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            cwd=self.cache_dir.parent
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
        raise RuntimeError("Not in a git repository")

    async def _run_git_command(self, args: List[str], cwd: Path) -> str:
        """Run a git command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            'git', *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"Git command failed: {stderr.decode()}")
        return stdout.decode()

    async def flush(self):
        """Force flush any pending batch."""
        await self._flush_batch()

    def get_stats(self) -> Dict[str, Any]:
        """Get uploader statistics."""
        return dict(self.stats)

    async def close(self):
        """Clean shutdown - flush any pending uploads."""
        await self.flush()
        await self._flush_trajectory_batch()  # Flush any pending trajectories