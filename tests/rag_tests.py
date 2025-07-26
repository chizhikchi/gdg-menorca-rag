"""
Test Suite for GDG Menorca RAG Manager
=====================================

Comprehensive test suite covering:
- RAG manager functionality
- Corpus operations
- Error handling
- Configuration management
- CLI interface
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
from rag_manager import HotelRAGManager, CorpusStatus, CorpusMetadata
from unittest.mock import AsyncMock


class TestCorpusMetadata:
    """Test CorpusMetadata dataclass"""
    
    def test_metadata_creation(self):
        metadata = CorpusMetadata(
            name="test-corpus",
            display_name="Test Corpus",
            created_at="2024-01-01T00:00:00",
            document_count=10,
            status=CorpusStatus.COMPLETE,
            last_updated="2024-01-01T00:00:00",
            generation_config={"model": "gemini-2.5-flash"}
        )
        
        assert metadata.name == "test-corpus"
        assert metadata.document_count == 10
        assert metadata.status == CorpusStatus.COMPLETE
    
    def test_metadata_serialization(self):
        metadata = CorpusMetadata(
            name="test-corpus",
            display_name="Test Corpus", 
            created_at="2024-01-01T00:00:00",
            document_count=5,
            status=CorpusStatus.PARTIAL,
            last_updated="2024-01-01T00:00:00",
            generation_config={}
        )
        
        # Test to_dict
        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == "test-corpus"
        
        # Test from_dict
        reconstructed = CorpusMetadata.from_dict(data)
        assert reconstructed.name == metadata.name
        assert reconstructed.status == metadata.status


class TestHotelRAGManager:
    """Test HotelRAGManager class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_documents(self, temp_dir):
        """Create sample documents JSON file"""
        documents = [
            {
                "title": "Hotel Overview",
                "prompt": "Generate a hotel overview"
            },
            {
                "title": "Room Types",
                "prompt": "Describe room types"
            }
        ]
        
        docs_file = temp_dir / "test_documents.json"
        with open(docs_file, 'w') as f:
            json.dump(documents, f)
        
        return docs_file
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration"""
        return {
            "api_key_env": "TEST_GEMINI_KEY",
            "project_env": "TEST_PROJECT",
            "corpus_name_env": "TEST_CORPUS",
            "location_env": "TEST_LOCATION",
            "model": "gemini-2.5-flash",
            "additional_instructions": "Test instructions"
        }
    
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus',
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init')
    def test_manager_initialization(self, mock_vertexai_init, mock_genai_client, temp_dir, sample_documents):
        """Test RAG manager initialization"""
        
        # Create config file
        config_file = temp_dir / "test_config.json"
        config = {
            "api_key_env": "TEST_GEMINI_KEY",
            "project_env": "TEST_PROJECT",
            "corpus_name_env": "TEST_CORPUS",
            "location_env": "TEST_LOCATION"
        }
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        manager = HotelRAGManager(
            config_file=str(config_file),
            input_json=str(sample_documents),
            output_dir=str(temp_dir / "output"),
            backup_dir=str(temp_dir / "backup")
        )
        
        assert manager.api_key == "test_key"
        assert manager.project == "test_project"
        assert manager.corpus_display_name == "test_corpus"
        assert manager.location == "us-central1"
        
        # Verify client initialization
        mock_genai_client.assert_called_once_with(api_key="test_key")
        mock_vertexai_init.assert_called_once_with(project="test_project", location="us-central1")
    
    def test_sanitize_filename(self, temp_dir, sample_documents):
        """Test filename sanitization"""
        with patch.dict('os.environ', {
            'TEST_GEMINI_KEY': 'test_key',
            'TEST_PROJECT': 'test_project', 
            'TEST_CORPUS': 'test_corpus',
            'TEST_LOCATION': 'us-central1'
        }), patch('rag_manager.genai.Client'), patch('rag_manager.vertexai.init'):
            
            manager = HotelRAGManager(
                input_json=str(sample_documents),
                output_dir=str(temp_dir / "output")
            )
            
            # Test various problematic filenames
            test_cases = [
                ("Hotel & Spa Overview", "Hotel___Spa_Overview"),
                ("Room/Suite Types", "Room_Suite_Types"),
                ("Dining @ Resort", "Dining___Resort"),
                ("Kids' Activities", "Kids__Activities"),
                ("Wi-Fi & Internet", "Wi-Fi___Internet")
            ]
            
            for input_name, expected in test_cases:
                result = manager.sanitize_filename(input_name)
                assert result == expected
    
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus', 
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init')
    @pytest.mark.asyncio
    async def test_generate_documents_success(self, mock_vertexai_init, mock_genai_client, temp_dir, sample_documents):
        """Test successful document generation"""
        
        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Generated hotel content"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_client
        
        manager = HotelRAGManager(
            input_json=str(sample_documents),
            output_dir=str(temp_dir / "output"),
            backup_dir=str(temp_dir / "backup")
        )
        
        # Test document generation
        success = await manager.generate_documents(interactive=False)
        
        assert success is True
        assert manager.metadata.document_count == 2
        assert manager.metadata.status == CorpusStatus.COMPLETE
        
        # Verify files were created
        output_dir = temp_dir / "output"
        assert (output_dir / "Hotel_Overview.txt").exists()
        assert (output_dir / "Room_Types.txt").exists()
        
        # Verify content
        with open(output_dir / "Hotel_Overview.txt", 'r') as f:
            content = f.read()
            assert content == "Generated hotel content"
    
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus',
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init') 
    @pytest.mark.asyncio
    async def test_generate_documents_partial_failure(self, mock_vertexai_init, mock_genai_client, temp_dir, sample_documents):
        """Test document generation with some failures"""
        
        # Mock client with mixed success/failure
        mock_client = Mock()
        mock_genai_client.return_value = mock_client
        
        def mock_generate_content(model, contents):
            if "Hotel Overview" in contents:
                mock_response = Mock()
                mock_response.text = "Generated content"
                return mock_response
            else:
                raise Exception("Generation failed")
        
        mock_client.models.generate_content.side_effect = mock_generate_content
        
        manager = HotelRAGManager(
            input_json=str(sample_documents),
            output_dir=str(temp_dir / "output"),
            backup_dir=str(temp_dir / "backup")
        )
        
        success = await manager.generate_documents(interactive=False)
        
        assert success is False  # Should be False due to partial failure
        assert manager.metadata.document_count == 1  # Only one succeeded
        assert manager.metadata.status == CorpusStatus.PARTIAL
    
    @patch('rag_manager.rag.list_corpora')
    def test_get_corpus_status_not_found(self, mock_list_corpora, temp_dir, sample_documents):
        """Test corpus status when corpus doesn't exist"""
        
        mock_list_corpora.return_value = []  # No corpora found
        
        with patch.dict('os.environ', {
            'TEST_GEMINI_KEY': 'test_key',
            'TEST_PROJECT': 'test_project',
            'TEST_CORPUS': 'test_corpus',
            'TEST_LOCATION': 'us-central1'
        }), patch('rag_manager.genai.Client'), patch('rag_manager.vertexai.init'):
            
            manager = HotelRAGManager(
                input_json=str(sample_documents),
                output_dir=str(temp_dir / "output")
            )
            
            corpus, status = manager.get_corpus_status()
            
            assert corpus is None
            assert status == CorpusStatus.NOT_FOUND
    
    @patch('rag_manager.rag.list_corpora')
    @patch('rag_manager.rag.list_files')
    def test_get_corpus_status_complete(self, mock_list_files, mock_list_corpora, temp_dir, sample_documents):
        """Test corpus status when complete"""
        
        # Mock corpus
        mock_corpus = Mock()
        mock_corpus.display_name = "test_corpus"
        mock_corpus.name = "corpus_name"
        mock_list_corpora.return_value = [mock_corpus]
        
        # Mock files (matching document count)
        mock_files = [Mock() for _ in range(2)]
        mock_list_files.return_value = mock_files
        
        with patch.dict('os.environ', {
            'TEST_GEMINI_KEY': 'test_key',
            'TEST_PROJECT': 'test_project',
            'TEST_CORPUS': 'test_corpus',
            'TEST_LOCATION': 'us-central1'
        }), patch('rag_manager.genai.Client'), patch('rag_manager.vertexai.init'):
            
            manager = HotelRAGManager(
                input_json=str(sample_documents),
                output_dir=str(temp_dir / "output")
            )
            manager.metadata.document_count = 2  # Set expected count
            
            corpus, status = manager.get_corpus_status()
            
            assert corpus == mock_corpus
            assert status == CorpusStatus.COMPLETE
    
    @patch('rag_manager.rag.create_corpus')
    def test_create_corpus_success(self, mock_create_corpus, temp_dir, sample_documents):
        """Test successful corpus creation"""
        
        # Mock successful corpus creation
        mock_corpus = Mock()
        mock_corpus.name = "created_corpus_name"
        mock_corpus.display_name = "test_corpus"
        mock_create_corpus.return_value = mock_corpus
        
        with patch.dict('os.environ', {
            'TEST_GEMINI_KEY': 'test_key',
            'TEST_PROJECT': 'test_project',
            'TEST_CORPUS': 'test_corpus',
            'TEST_LOCATION': 'us-central1'
        }), patch('rag_manager.genai.Client'), patch('rag_manager.vertexai.init'):
            
            manager = HotelRAGManager(
                input_json=str(sample_documents),
                output_dir=str(temp_dir / "output")
            )
            
            # Mock get_corpus_status to return NOT_FOUND
            with patch.object(manager, 'get_corpus_status', return_value=(None, CorpusStatus.NOT_FOUND)):
                corpus = manager.create_corpus()
            
            assert corpus == mock_corpus
            assert manager.metadata.name == "created_corpus_name"
            assert manager.metadata.status == CorpusStatus.EMPTY
            mock_create_corpus.assert_called_once()
    
    @patch('rag_manager.rag.upload_file')
    def test_upload_documents_success(self, mock_upload_file, temp_dir, sample_documents):
        """Test successful document upload"""
        
        with patch.dict('os.environ', {
            'TEST_GEMINI_KEY': 'test_key',
            'TEST_PROJECT': 'test_project',
            'TEST_CORPUS': 'test_corpus',
            'TEST_LOCATION': 'us-central1'
        }), patch('rag_manager.genai.Client'), patch('rag_manager.vertexai.init'):
            
            manager = HotelRAGManager(
                input_json=str(sample_documents),
                output_dir=str(temp_dir / "output")
            )
            
            # Create test files
            output_dir = temp_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            test_files = ["doc1.txt", "doc2.txt"]
            for filename in test_files:
                (output_dir / filename).write_text("Test content")
            
            # Mock corpus
            mock_corpus = Mock()
            mock_corpus.name = "test_corpus_name"
            
            success = manager.upload_documents(mock_corpus)
            
            assert success is True
            assert manager.metadata.document_count == 2
            assert manager.metadata.status == CorpusStatus.COMPLETE
            assert mock_upload_file.call_count == 2


class TestCLIInterface:
    """Test CLI interface functionality"""
    
    @patch('rag_manager.HotelRAGManager')
    def test_cli_status_command(self, mock_manager_class):
        """Test CLI status command"""
        from click.testing import CliRunner
        from rag_manager import cli
        
        # Mock manager instance
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        mock_manager.show_status.assert_called_once()
    
    @patch('rag_manager.HotelRAGManager')
    @patch('rag_manager.asyncio.run')
    def test_cli_generate_command(self, mock_asyncio_run, mock_manager_class):
        """Test CLI generate command"""
        from click.testing import CliRunner
        from rag_manager import cli
        
        # Mock manager and methods
        mock_manager = Mock()
        mock_corpus = Mock()
        mock_manager.create_corpus.return_value = mock_corpus
        mock_manager.upload_documents.return_value = True
        mock_manager_class.return_value = mock_manager
        mock_asyncio_run.return_value = True  # Successful generation
        
        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '--no-interactive'])
        
        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()
        mock_manager.create_corpus.assert_called_once()
        mock_manager.upload_documents.assert_called_once_with(mock_corpus)


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch.dict('os.environ', {}, clear=True)
    def test_missing_environment_variables(self, temp_dir):
        """Test handling of missing environment variables"""
        
        with pytest.raises(ValueError, match="Environment variable.*not found"):
            HotelRAGManager(
                input_json=str(temp_dir / "docs.json"),
                output_dir=str(temp_dir / "output")
            )
    
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus',
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init')
    def test_missing_input_file(self, mock_vertexai_init, mock_genai_client, temp_dir):
        """Test handling of missing input file"""
        
        manager = HotelRAGManager(
            input_json=str(temp_dir / "nonexistent.json"),
            output_dir=str(temp_dir / "output")
        )
        
        # This should handle the missing file gracefully in generate_documents
        result = asyncio.run(manager.generate_documents(interactive=False))
        assert result is False


class TestPerformanceAndScaling:
    """Test performance and scaling scenarios"""
    
    @pytest.fixture
    def large_document_set(self, temp_dir):
        """Create large set of documents for testing"""
        documents = []
        for i in range(100):
            documents.append({
                "title": f"Document {i:03d}",
                "prompt": f"Generate content for document {i}"
            })
        
        docs_file = temp_dir / "large_documents.json"
        with open(docs_file, 'w') as f:
            json.dump(documents, f)
        
        return docs_file
    
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus',
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init')
    @pytest.mark.asyncio
    async def test_large_document_generation(self, mock_vertexai_init, mock_genai_client, temp_dir, large_document_set):
        """Test generation with large document set"""
        
        # Mock successful responses
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Generated content"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_client
        
        manager = HotelRAGManager(
            input_json=str(large_document_set),
            output_dir=str(temp_dir / "output"),
            backup_dir=str(temp_dir / "backup")
        )
        
        success = await manager.generate_documents(interactive=False)
        
        assert success is True
        assert manager.metadata.document_count == 100
        
        # Verify all files were created
        output_files = list((temp_dir / "output").glob("*.txt"))
        assert len(output_files) == 100


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.integration
    @patch.dict('os.environ', {
        'TEST_GEMINI_KEY': 'test_key',
        'TEST_PROJECT': 'test_project',
        'TEST_CORPUS': 'test_corpus',
        'TEST_LOCATION': 'us-central1'
    })
    @patch('rag_manager.genai.Client')
    @patch('rag_manager.vertexai.init')
    @patch('rag_manager.rag.create_corpus')
    @patch('rag_manager.rag.upload_file')
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_upload_file, mock_create_corpus, 
                                   mock_vertexai_init, mock_genai_client, temp_dir, sample_documents):
        """Test complete workflow from document generation to corpus upload"""
        
        # Mock all external dependencies
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Generated hotel content"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_client
        
        mock_corpus = Mock()
        mock_corpus.name = "created_corpus"
        mock_create_corpus.return_value = mock_corpus
        
        manager = HotelRAGManager(
            input_json=str(sample_documents),
            output_dir=str(temp_dir / "output"),
            backup_dir=str(temp_dir / "backup")
        )
        
        # Mock corpus status to return NOT_FOUND initially
        with patch.object(manager, 'get_corpus_status', return_value=(None, CorpusStatus.NOT_FOUND)):
            # Step 1: Generate documents
            gen_success = await manager.generate_documents(interactive=False)
            assert gen_success is True
            
            # Step 2: Create corpus
            corpus = manager.create_corpus()
            assert corpus == mock_corpus
            
            # Step 3: Upload documents
            upload_success = manager.upload_documents(corpus)
            assert upload_success is True
        
        # Verify final state
        assert manager.metadata.status == CorpusStatus.COMPLETE
        assert manager.metadata.document_count == 2
        mock_create_corpus.assert_called_once()
        assert mock_upload_file.call_count == 2


# Fixtures for common test data
@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "model": "gemini-2.5-flash",
        "api_key_env": "TEST_GEMINI_KEY",
        "project_env": "TEST_PROJECT",
        "corpus_name_env": "TEST_CORPUS",
        "location_env": "TEST_LOCATION",
        "additional_instructions": "Test instructions"
    }


# Run tests with: pytest tests/ -v
# Run with coverage: pytest tests/ --cov=rag_manager --cov-report=html
# Run integration tests: pytest tests/ -v -m integration