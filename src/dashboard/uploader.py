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


@dataclass
class FileBatch:
    """A batch of files to upload."""
    files: Dict[str, bytes] = field(default_factory=dict)
    total_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    def add_file(self, path: str, content: bytes) -> bool:
        """Add a file to the batch. Returns True if added, False if would exceed limits."""
        file_size = len(content)
        if self.total_bytes + file_size > 8 * 1024 * 1024:  # 8 MB limit
            return False
        self.files[path] = content
        self.total_bytes += file_size
        return True

    def is_empty(self) -> bool:
        return len(self.files) == 0

    def age_seconds(self) -> float:
        return time.time() - self.created_at


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

        # Current batch
        self.current_batch = FileBatch()
        self.last_flush = time.time()

        # Stats
        self.stats = {
            'files_uploaded': 0,
            'bytes_uploaded': 0,
            'batches_flushed': 0,
            'rate_limit_hits': 0,
            'builds_triggered': 0
        }

        logger.info(f"Dashboard uploader initialized with mode: {self.upload_mode}")

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

    async def queue_file(self, relative_path: str, content: bytes) -> bool:
        """Queue a file for upload. Returns True if queued successfully."""
        if self.upload_mode == UploadMode.NO_OP:
            # In NO_OP mode, still queue files for testing purposes
            pass

        # Check file size limits (avoid LFS)
        if len(content) > 50 * 1024 * 1024:  # 50 MB
            logger.warning(f"File {relative_path} too large ({len(content)} bytes), skipping")
            return False

        # Try to add to current batch
        if not self.current_batch.add_file(relative_path, content):
            # Batch is full, flush it first
            await self._flush_batch()
            # Try again with new batch
            if not self.current_batch.add_file(relative_path, content):
                logger.error(f"File {relative_path} too large for empty batch")
                return False

        # Check if we should flush based on time or size
        should_flush = (
            self.current_batch.age_seconds() >= self.config.flush_seconds or
            self.current_batch.total_bytes >= self.config.max_batch_bytes
        )

        if should_flush:
            await self._flush_batch()

        return True

    async def _flush_batch(self):
        """Flush the current batch to the dashboard."""
        if self.current_batch.is_empty():
            return

        # Check rate limits
        file_count = len(self.current_batch.files)
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
                await self._flush_via_git()
            elif self.upload_mode == UploadMode.GITHUB_API:
                await self._flush_via_api()
            else:
                # NO_OP - just clear batch
                pass

            self.stats['batches_flushed'] += 1
            self.stats['files_uploaded'] += file_count
            self.stats['bytes_uploaded'] += self.current_batch.total_bytes
            self.stats['builds_triggered'] += 1

            logger.info(f"Flushed batch: {file_count} files, {self.current_batch.total_bytes} bytes")

        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            # Keep batch for retry on next flush
            return

        # Reset batch
        self.current_batch = FileBatch()
        self.last_flush = time.time()

    async def _flush_via_git(self):
        """Flush batch via git push to pages branch."""
        # Create temporary directory for batch
        batch_dir = self.cache_dir / f"batch_{int(time.time())}"
        batch_dir.mkdir()

        try:
            # Write files to batch directory
            for rel_path, content in self.current_batch.files.items():
                file_path = batch_dir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(content)

            # Copy to site root in repo
            repo_root = self._find_repo_root()
            site_dir = repo_root / self.config.site_root
            site_dir.mkdir(exist_ok=True)

            # Use rsync or similar to copy (simplified - just copy for now)
            import shutil
            for rel_path in self.current_batch.files:
                src = batch_dir / rel_path
                dst = site_dir / rel_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            # Git add, commit, push
            await self._run_git_command(['add', '.'], cwd=site_dir)
            await self._run_git_command(['commit', '-m', f'Dashboard update: {len(self.current_batch.files)} files'], cwd=site_dir)
            await self._run_git_command(['push', 'origin', self.config.branch], cwd=site_dir)

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(batch_dir, ignore_errors=True)

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