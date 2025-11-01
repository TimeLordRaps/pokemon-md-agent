"""Tests for path sanitization utilities."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from src.agent.utils import sanitize_hf_home, get_hf_cache_dir


class TestPathSanitization:
    """Test path sanitization functions."""

    def test_sanitize_hf_home_with_double_quotes(self):
        """Test HF_HOME sanitization strips double quotes."""
        test_cases = [
            ('"E:\\transformer_models"', 'E:\\transformer_models'),
            ("'E:\\transformer_models'", 'E:\\transformer_models'),
            ('E:\\transformer_models', 'E:\\transformer_models'),
            ('"E:\\transformer_models\\"', 'E:\\transformer_models'),  # Escaped quote
            ('  "E:\\transformer_models"  ', 'E:\\transformer_models'),  # With whitespace
        ]

        for raw, expected in test_cases:
            with patch.dict(os.environ, {'HF_HOME': raw}):
                result = sanitize_hf_home()
                assert os.path.normpath(result) == os.path.normpath(expected), \
                    f"Failed for input: {raw}"

    def test_sanitize_hf_home_none_when_not_set(self):
        """Test sanitize_hf_home returns None when HF_HOME not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = sanitize_hf_home()
            assert result is None

    def test_sanitize_hf_home_expanduser(self):
        """Test sanitize_hf_home expands user path."""
        with patch.dict(os.environ, {'HF_HOME': '~/test_path'}):
            with patch('os.path.expanduser') as mock_expand:
                mock_expand.return_value = '/home/user/test_path'
                result = sanitize_hf_home()
                mock_expand.assert_called_once_with('~/test_path')
                # On Windows, normpath converts to backslashes
                expected = os.path.normpath('/home/user/test_path')
                assert result == expected

    def test_sanitize_hf_home_normpath(self):
        """Test sanitize_hf_home normalizes path separators."""
        with patch.dict(os.environ, {'HF_HOME': 'E:\\\\transformer_models\\\\hub'}):
            result = sanitize_hf_home()
            # normpath should normalize separators
            expected = os.path.normpath('E:\\transformer_models\\hub')
            assert result == expected

    def test_get_hf_cache_dir_appends_hub(self):
        """Test get_hf_cache_dir appends hub subdirectory."""
        with patch.dict(os.environ, {'HF_HOME': 'E:\\transformer_models'}):
            result = get_hf_cache_dir()
            expected = os.path.join('E:\\transformer_models', 'hub')
            assert os.path.normpath(result) == os.path.normpath(expected)

    def test_get_hf_cache_dir_with_quotes(self):
        """Test get_hf_cache_dir handles quoted HF_HOME."""
        with patch.dict(os.environ, {'HF_HOME': '"E:\\transformer_models"'}):
            result = get_hf_cache_dir()
            expected = os.path.join('E:\\transformer_models', 'hub')
            assert os.path.normpath(result) == os.path.normpath(expected)

    def test_get_hf_cache_dir_none_when_not_set(self):
        """Test get_hf_cache_dir returns None when HF_HOME not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_hf_cache_dir()
            assert result is None

    @pytest.mark.parametrize("input_path,expected_hub", [
        ('E:\\transformer_models', 'E:\\transformer_models\\hub'),
        ('/home/user/.cache', '/home/user/.cache/hub'),
        ('C:\\Users\\test\\huggingface', 'C:\\Users\\test\\huggingface\\hub'),
    ])
    def test_get_hf_cache_dir_parametrized(self, input_path, expected_hub):
        """Test get_hf_cache_dir with various paths."""
        with patch.dict(os.environ, {'HF_HOME': input_path}):
            result = get_hf_cache_dir()
            assert os.path.normpath(result) == os.path.normpath(expected_hub)