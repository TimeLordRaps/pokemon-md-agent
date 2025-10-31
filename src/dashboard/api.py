"""Dashboard API server for PMD-Red Agent.

Provides REST endpoints for batch uploads and content retrieval with pagination/filtering.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger(__name__)


@dataclass
class UploadedContent:
    """Represents uploaded content with metadata."""
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    content_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'filename': self.filename,
            'content_type': self.content_type,
            'size_bytes': self.size_bytes,
            'uploaded_at': self.uploaded_at,
            'metadata': self.metadata,
            'tags': self.tags,
            'content_hash': self.content_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UploadedContent':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            filename=data['filename'],
            content_type=data['content_type'],
            size_bytes=data['size_bytes'],
            uploaded_at=data['uploaded_at'],
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            content_hash=data.get('content_hash')
        )


@dataclass
class ContentStore:
    """In-memory content store with persistence."""

    contents: Dict[str, UploadedContent] = field(default_factory=dict)
    storage_dir: Path = field(default_factory=lambda: Path.home() / '.cache' / 'pmd-red' / 'uploads')
    max_entries: int = 10000

    def __post_init__(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_content()

    def _load_persisted_content(self):
        """Load persisted content metadata."""
        index_file = self.storage_dir / 'content_index.json'
        if not index_file.exists():
            return

        try:
            with open(index_file, 'r') as f:
                data = json.load(f)
                for item_data in data.get('contents', []):
                    content = UploadedContent.from_dict(item_data)
                    self.contents[content.id] = content
            logger.info(f"Loaded {len(self.contents)} persisted content items")
        except Exception as e:
            logger.warning(f"Failed to load persisted content: {e}")

    def _save_index(self):
        """Save content index to disk."""
        index_file = self.storage_dir / 'content_index.json'
        try:
            data = {
                'contents': [content.to_dict() for content in self.contents.values()],
                'last_updated': time.time()
            }
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save content index: {e}")

    def add_content(self, content: UploadedContent, file_data: bytes) -> bool:
        """Add content to store. Returns True if successful."""
        if len(self.contents) >= self.max_entries:
            # Remove oldest entries
            sorted_items = sorted(self.contents.items(), key=lambda x: x[1].uploaded_at)
            to_remove = len(sorted_items) - self.max_entries + 1
            for i in range(to_remove):
                old_id = sorted_items[i][0]
                del self.contents[old_id]
                # Also remove file if it exists
                file_path = self.storage_dir / f"{old_id}.bin"
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass

        # Save file data
        file_path = self.storage_dir / f"{content.id}.bin"
        try:
            with open(file_path, 'wb') as f:
                f.write(file_data)
        except Exception as e:
            logger.error(f"Failed to save content file {content.id}: {e}")
            return False

        # Add to index
        self.contents[content.id] = content
        self._save_index()

        logger.info(f"Added content: {content.filename} ({content.size_bytes} bytes)")
        return True

    def get_content(self, content_id: str) -> Optional[UploadedContent]:
        """Get content by ID."""
        return self.contents.get(content_id)

    def list_contents(
        self,
        limit: int = 50,
        offset: int = 0,
        content_type_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        filename_pattern: Optional[str] = None
    ) -> List[UploadedContent]:
        """List contents with filtering and pagination."""
        # Start with all contents
        filtered = list(self.contents.values())

        # Apply filters
        if content_type_filter:
            filtered = [c for c in filtered if content_type_filter in c.content_type]

        if tag_filter:
            filtered = [c for c in filtered if tag_filter in c.tags]

        if date_from:
            filtered = [c for c in filtered if c.uploaded_at >= date_from]

        if date_to:
            filtered = [c for c in filtered if c.uploaded_at <= date_to]

        if filename_pattern:
            filtered = [c for c in filtered if filename_pattern.lower() in c.filename.lower()]

        # Sort by upload time (newest first)
        filtered.sort(key=lambda c: c.uploaded_at, reverse=True)

        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return filtered[start_idx:end_idx]

    def get_content_file(self, content_id: str) -> Optional[bytes]:
        """Get raw file data for content."""
        content = self.get_content(content_id)
        if not content:
            return None

        file_path = self.storage_dir / f"{content_id}.bin"
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read content file {content_id}: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_size = sum(c.size_bytes for c in self.contents.values())
        content_types = {}
        for content in self.contents.values():
            ct = content.content_type
            content_types[ct] = content_types.get(ct, 0) + 1

        return {
            'total_contents': len(self.contents),
            'total_size_bytes': total_size,
            'content_types': content_types,
            'oldest_upload': min((c.uploaded_at for c in self.contents.values()), default=None),
            'newest_upload': max((c.uploaded_at for c in self.contents.values()), default=None)
        }


# Pydantic models for API requests/responses
class BatchUploadRequest(BaseModel):
    """Request model for batch upload."""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class BatchUploadResponse(BaseModel):
    """Response model for batch upload."""
    uploaded_ids: List[str]
    failed_files: List[str]
    total_uploaded: int
    total_failed: int


class ContentItem(BaseModel):
    """Response model for content item."""
    id: str
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: float
    metadata: Dict[str, Any]
    tags: List[str]
    content_hash: Optional[str]


class FetchManyResponse(BaseModel):
    """Response model for fetch_many."""
    items: List[ContentItem]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class ContentStats(BaseModel):
    """Response model for content statistics."""
    total_contents: int
    total_size_bytes: int
    content_types: Dict[str, int]
    oldest_upload: Optional[float]
    newest_upload: Optional[float]


# Global content store instance
content_store = ContentStore()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="PMD-Red Agent Dashboard API",
        description="REST API for batch content uploads and retrieval",
        version="1.0.0"
    )

    @app.post("/batch-upload", response_model=BatchUploadResponse)
    async def batch_upload(
        files: List[UploadFile] = File(...),
        metadata: Optional[str] = Form(None),
        tags: Optional[str] = Form(None)
    ):
        """Batch upload multiple files with optional metadata and tags.

        Accepts multipart form data with:
        - files: List of files to upload
        - metadata: JSON string with shared metadata for all files
        - tags: JSON string with list of tags for all files
        """
        try:
            # Parse metadata and tags
            shared_metadata = json.loads(metadata) if metadata else {}
            shared_tags = json.loads(tags) if tags else []

            uploaded_ids = []
            failed_files = []

            for file in files:
                try:
                    # Read file content
                    content = await file.read()

                    # Generate unique ID
                    content_id = f"{int(time.time() * 1000000)}_{hash(file.filename)}"

                    # Create content object
                    uploaded_content = UploadedContent(
                        id=content_id,
                        filename=file.filename or "unknown",
                        content_type=file.content_type or "application/octet-stream",
                        size_bytes=len(content),
                        uploaded_at=time.time(),
                        metadata=shared_metadata.copy(),
                        tags=shared_tags.copy()
                    )

                    # Add to store
                    if content_store.add_content(uploaded_content, content):
                        uploaded_ids.append(content_id)
                    else:
                        failed_files.append(file.filename)

                except Exception as e:
                    logger.error(f"Failed to process file {file.filename}: {e}")
                    failed_files.append(file.filename)

            return BatchUploadResponse(
                uploaded_ids=uploaded_ids,
                failed_files=failed_files,
                total_uploaded=len(uploaded_ids),
                total_failed=len(failed_files)
            )

        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in metadata/tags: {e}")
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/fetch-many", response_model=FetchManyResponse)
    async def fetch_many(
        limit: int = Query(50, ge=1, le=1000, description="Maximum number of items to return"),
        offset: int = Query(0, ge=0, description="Number of items to skip"),
        content_type: Optional[str] = Query(None, description="Filter by content type substring"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
        date_from: Optional[float] = Query(None, description="Filter by minimum upload timestamp"),
        date_to: Optional[float] = Query(None, description="Filter by maximum upload timestamp"),
        filename: Optional[str] = Query(None, description="Filter by filename pattern")
    ):
        """Fetch multiple content items with pagination and filtering.

        Returns paginated list of content items with optional filtering by:
        - content_type: substring match in content type
        - tag: exact tag match
        - date_from/date_to: upload timestamp range
        - filename: substring match in filename (case-insensitive)
        """
        try:
            # Get filtered and paginated results
            items = content_store.list_contents(
                limit=limit,
                offset=offset,
                content_type_filter=content_type,
                tag_filter=tag,
                date_from=date_from,
                date_to=date_to,
                filename_pattern=filename
            )

            # Get total count for pagination info
            # Note: This is inefficient for large datasets - in production,
            # you'd want a separate count query or database indexing
            all_filtered = content_store.list_contents(
                limit=10000,  # Large limit to get all
                offset=0,
                content_type_filter=content_type,
                tag_filter=tag,
                date_from=date_from,
                date_to=date_to,
                filename_pattern=filename
            )
            total_count = len(all_filtered)

            # Convert to response model
            response_items = [
                ContentItem(
                    id=item.id,
                    filename=item.filename,
                    content_type=item.content_type,
                    size_bytes=item.size_bytes,
                    uploaded_at=item.uploaded_at,
                    metadata=item.metadata,
                    tags=item.tags,
                    content_hash=item.content_hash
                )
                for item in items
            ]

            return FetchManyResponse(
                items=response_items,
                total_count=total_count,
                limit=limit,
                offset=offset,
                has_more=(offset + limit) < total_count
            )

        except Exception as e:
            logger.error(f"Fetch many failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/content/{content_id}")
    async def get_content(content_id: str):
        """Get a specific content item by ID."""
        content = content_store.get_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Return file data
        file_data = content_store.get_content_file(content_id)
        if file_data is None:
            raise HTTPException(status_code=404, detail="Content file not found")

        return JSONResponse(
            content={
                'content': content.to_dict(),
                'file_data': file_data.hex()  # Return as hex string for JSON compatibility
            }
        )

    @app.get("/stats", response_model=ContentStats)
    async def get_stats():
        """Get content store statistics."""
        return ContentStats(**content_store.get_stats())

    @app.delete("/content/{content_id}")
    async def delete_content(content_id: str):
        """Delete a content item by ID."""
        content = content_store.get_content(content_id)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Remove from store
        if content_id in content_store.contents:
            del content_store.contents[content_id]
            content_store._save_index()

            # Remove file
            file_path = content_store.storage_dir / f"{content_id}.bin"
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete content file {content_id}: {e}")

        return {"message": f"Content {content_id} deleted"}

    return app


# For running the server directly
if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)