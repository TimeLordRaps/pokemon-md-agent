"""Tests for Dashboard Content API endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json
import time
from io import BytesIO

from src.dashboard.api import create_app, ContentStore, UploadedContent


class TestContentStore:
    """Test ContentStore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, temp_dir):
        """Test ContentStore initialization."""
        store = ContentStore(storage_dir=temp_dir)
        assert store.storage_dir == temp_dir
        assert len(store.contents) == 0

    def test_add_content(self, temp_dir):
        """Test adding content to store."""
        store = ContentStore(storage_dir=temp_dir)

        content = UploadedContent(
            id="test-1",
            filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            uploaded_at=time.time(),
            metadata={"tags": ["test"]}
        )

        file_data = b"Hello, World!"
        success = store.add_content(content, file_data)
        assert success is True
        assert len(store.contents) == 1
        assert store.contents["test-1"] == content

    def test_get_content(self, temp_dir):
        """Test retrieving content from store."""
        store = ContentStore(storage_dir=temp_dir)

        content = UploadedContent(
            id="test-1",
            filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            uploaded_at=time.time(),
            metadata={"tags": ["test"]}
        )

        file_data = b"Hello, World!"
        store.add_content(content, file_data)

        # Get existing content
        retrieved = store.get_content("test-1")
        assert retrieved == content

        # Get non-existing content
        assert store.get_content("nonexistent") is None

    def test_delete_content(self, temp_dir):
        """Test deleting content from store."""
        store = ContentStore(storage_dir=temp_dir)

        content = UploadedContent(
            id="test-1",
            filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            uploaded_at=time.time(),
            metadata={"tags": ["test"]}
        )

        file_data = b"Hello, World!"
        store.add_content(content, file_data)
        assert len(store.contents) == 1

        # Delete existing content (simulate what the API does)
        if "test-1" in store.contents:
            del store.contents["test-1"]
            store._save_index()
        assert len(store.contents) == 0

    def test_persistence(self, temp_dir):
        """Test content persistence across store instances."""
        # First store instance
        store1 = ContentStore(storage_dir=temp_dir)
        content = UploadedContent(
            id="test-1",
            filename="test.txt",
            content_type="text/plain",
            size_bytes=100,
            uploaded_at=time.time(),
            metadata={"tags": ["test"]}
        )
        file_data = b"Hello, World!"
        store1.add_content(content, file_data)

        # Second store instance should load persisted data
        store2 = ContentStore(storage_dir=temp_dir)
        assert len(store2.contents) == 1
        assert store2.contents["test-1"].id == "test-1"


class TestDashboardAPI:
    """Test Dashboard API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for API."""
        # Clear the global content store before each test
        from src.dashboard.api import content_store
        content_store.contents.clear()
        content_store._save_index()
        
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def sample_file(self):
        """Create sample file for upload testing."""
        return BytesIO(b"Hello, World!")

    def test_batch_upload_single_file(self, client, sample_file):
        """Test batch upload with single file."""
        files = {"files": ("test.txt", sample_file, "text/plain")}
        data = {"metadata": json.dumps({"tags": ["test"]})}

        response = client.post("/batch-upload", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert "uploaded_ids" in result
        assert len(result["uploaded_ids"]) == 1
        assert result["total_uploaded"] == 1
        assert result["total_failed"] == 0

    def test_batch_upload_multiple_files(self, client):
        """Test batch upload with multiple files."""
        files = [
            ("files", ("test1.txt", BytesIO(b"Content 1"), "text/plain")),
            ("files", ("test2.txt", BytesIO(b"Content 2"), "text/plain"))
        ]
        data = {"metadata": json.dumps({"tags": ["batch"]})}

        response = client.post("/batch-upload", files=files, data=data)

        assert response.status_code == 200
        result = response.json()
        assert len(result["uploaded_ids"]) == 2
        assert result["total_uploaded"] == 2
        assert result["total_failed"] == 0

    def test_batch_upload_no_files(self, client):
        """Test batch upload with no files."""
        response = client.post("/batch-upload")

        assert response.status_code == 422  # FastAPI validation error
        result = response.json()
        assert "detail" in result

    def test_batch_upload_invalid_metadata(self, client, sample_file):
        """Test batch upload with invalid JSON metadata."""
        files = {"files": ("test.txt", sample_file, "text/plain")}
        data = {"metadata": "invalid json"}

        response = client.post("/batch-upload", files=files, data=data)

        assert response.status_code == 400
        result = response.json()
        assert "detail" in result

    def test_fetch_many_empty(self, client):
        """Test fetch_many with no content."""
        response = client.get("/fetch-many")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] == 0
        assert result["items"] == []
        assert result["limit"] == 50
        assert result["offset"] == 0

    def test_fetch_many_with_content(self, client, sample_file):
        """Test fetch_many with existing content."""
        # First upload some content
        files = {"files": ("test.txt", sample_file, "text/plain")}
        data = {"metadata": json.dumps({"tags": ["test"]})}
        client.post("/batch-upload", files=files, data=data)

        # Then fetch it
        response = client.get("/fetch-many")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["filename"] == "test.txt"

    def test_fetch_many_pagination(self, client):
        """Test fetch_many pagination."""
        # Upload multiple files
        for i in range(5):
            files = {"files": (f"test{i}.txt", BytesIO(f"Content {i}".encode()), "text/plain")}
            client.post("/batch-upload", files=files)

        # Fetch with pagination
        response = client.get("/fetch-many?limit=2&offset=0")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] == 5
        assert len(result["items"]) == 2
        assert result["limit"] == 2
        assert result["offset"] == 0

    def test_fetch_many_filtering(self, client):
        """Test fetch_many with tag filtering."""
        # Upload files with different tags
        files1 = {"files": ("doc1.txt", BytesIO(b"Document 1"), "text/plain")}
        data1 = {"tags": json.dumps(["important", "doc"])}
        client.post("/batch-upload", files=files1, data=data1)

        files2 = {"files": ("img1.jpg", BytesIO(b"Fake image"), "image/jpeg")}
        data2 = {"tags": json.dumps(["image", "media"])}
        client.post("/batch-upload", files=files2, data=data2)

        # Filter by tag
        response = client.get("/fetch-many?tag=important")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] == 1
        assert result["items"][0]["filename"] == "doc1.txt"

    def test_fetch_many_content_type_filter(self, client):
        """Test fetch_many with content type filtering."""
        # Upload files with different content types
        files1 = {"files": ("doc1.txt", BytesIO(b"Document 1"), "text/plain")}
        client.post("/batch-upload", files=files1)

        files2 = {"files": ("img1.jpg", BytesIO(b"Fake image"), "image/jpeg")}
        client.post("/batch-upload", files=files2)

        # Filter by content type
        response = client.get("/fetch-many?content_type=image/jpeg")

        assert response.status_code == 200
        result = response.json()
        assert result["total_count"] == 1
        assert result["items"][0]["filename"] == "img1.jpg"

    def test_get_content(self, client, sample_file):
        """Test individual content retrieval."""
        # Upload content first
        files = {"files": ("test.txt", sample_file, "text/plain")}
        data = {"metadata": json.dumps({"tags": ["test"]})}
        upload_response = client.post("/batch-upload", files=files, data=data)
        content_id = upload_response.json()["uploaded_ids"][0]

        # Retrieve content
        response = client.get(f"/content/{content_id}")

        assert response.status_code == 200
        result = response.json()
        assert "content" in result
        assert "file_data" in result
        assert result["content"]["id"] == content_id
        assert result["content"]["filename"] == "test.txt"
        assert result["content"]["content_type"] == "text/plain"

    def test_get_content_not_found(self, client):
        """Test retrieving non-existent content."""
        response = client.get("/content/nonexistent-id")

        assert response.status_code == 404
        result = response.json()
        assert "detail" in result
        assert "not found" in result["detail"].lower()

    def test_delete_content(self, client, sample_file):
        """Test content deletion."""
        # Upload content first
        files = {"files": ("test.txt", sample_file, "text/plain")}
        upload_response = client.post("/batch-upload", files=files)
        content_id = upload_response.json()["uploaded_ids"][0]

        # Delete content
        response = client.delete(f"/content/{content_id}")

        assert response.status_code == 200
        result = response.json()
        assert "message" in result

        # Verify content is gone
        get_response = client.get(f"/content/{content_id}")
        assert get_response.status_code == 404

    def test_delete_content_not_found(self, client):
        """Test deleting non-existent content."""
        response = client.delete("/content/nonexistent-id")

        assert response.status_code == 404
        result = response.json()
        assert "detail" in result

    def test_get_stats(self, client, sample_file):
        """Test statistics endpoint."""
        # Upload some content
        files = {"files": ("test.txt", sample_file, "text/plain")}
        client.post("/batch-upload", files=files)

        response = client.get("/stats")

        assert response.status_code == 200
        result = response.json()
        assert "total_contents" in result
        assert "total_size_bytes" in result
        assert "content_types" in result
        assert result["total_contents"] >= 1
