# 🏨 GDG Menorca Resort - Advanced RAG Management System

> **A sophisticated, production-ready RAG (Retrieval-Augmented Generation) system for hotel management with beautiful CLI interface, comprehensive monitoring, and enterprise-grade features.**

> **Note:** Docker support is temporarily removed for this GitHub upload. Docker files and deployment instructions will be available in a future release.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Powered by Vertex AI](https://img.shields.io/badge/Powered%20by-Vertex%20AI-4285f4.svg)](https://cloud.google.com/vertex-ai)

## ✨ Features at a Glance

🎨 **Beautiful CLI Interface** • 📊 **Progress Tracking** • 🔍 **Health Monitoring** • 🛡️ **Error Recovery** • 📝 **Comprehensive Logging** • 🚀 **Production Ready**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Project with Vertex AI enabled
- Gemini API key

### 1. Installation
```bash
git clone <your-repository>
cd gdg-menorca-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Your `.env` should contain:
```env
GEMINI_KEY=your_gemini_api_key_here
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
CORPUS_DISPLAY_NAME=gdg-menorca-hotel-docs
```

### 3. Generate Your First Corpus
```bash
# Interactive generation (recommended for first time)
python rag_manager.py generate --interactive

# Check status
python rag_manager.py status

# Start the chat interface
python app.py
```

---

## 🎯 Core Components

### 🧠 RAG Manager (`rag_manager.py`)
The heart of the system - handles document generation, corpus management, and Vertex AI integration.

**Key Features:**
- 🎨 Rich CLI with progress bars and colored output
- 🔄 Async document generation with retry logic
- 📊 Real-time status monitoring
- 🛡️ Comprehensive error handling
- 📝 Detailed logging and metrics

### 💬 Chat Interface (`app.py`)
Enhanced Gradio interface with admin panel and health monitoring.

**Features:**
- 🎨 Modern, responsive UI
- 🔧 Built-in admin panel for corpus management
- 📊 Real-time system status display
- 🏥 Health check endpoints
- 📱 Mobile-friendly design

### 🧪 Test Suite (`tests/`)
Comprehensive test coverage for all components.

**Coverage:**
- Unit tests for all core functions
- Integration tests for complete workflows
- Performance tests with large datasets
- Error handling and edge cases

---

## 🎨 Beautiful CLI Experience

### Status Dashboard
```bash
python rag_manager.py status
```
![Status Output](docs/images/status-example.png)

### Interactive Generation
```bash
python rag_manager.py generate --interactive
```

Visual progress tracking with:
- 🎯 Real-time document generation progress
- 📊 Upload progress with file names
- ✅ Success/failure counts
- 🎨 Color-coded status indicators

### Logs Monitoring
```bash
python rag_manager.py logs
```

---

## 🛠️ Advanced Usage

### Command Line Interface

#### Generate Corpus
```bash
# Interactive mode (recommended)
python rag_manager.py generate --interactive

# Automated mode
python rag_manager.py generate --no-interactive --upload

# Generate without uploading
python rag_manager.py generate --no-upload
```

#### Status Monitoring
```bash
# Detailed status information
python rag_manager.py status

# Health check for monitoring systems
curl http://localhost:8080/health
```

#### Maintenance Operations
```bash
# Safe cleanup (dry run first)
python rag_manager.py cleanup --dry-run
python rag_manager.py cleanup --no-dry-run

# View recent logs
python rag_manager.py logs
```

### Programmatic Usage

```python
from rag_manager import HotelRAGManager, CorpusStatus

# Initialize manager
manager = HotelRAGManager()

# Check current status
corpus, status = manager.get_corpus_status()
print(f"Corpus status: {status.value}")

# Generate documents
success = await manager.generate_documents(interactive=False)

# Create and upload corpus
if success:
    corpus = manager.create_corpus()
    if corpus:
        manager.upload_documents(corpus)

# Monitor health
health_status = manager.get_health_status()
```

---

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Gradio WebUI   │    │  Health Checks  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴───────────────┐
                    │     RAG Manager Core        │
                    │  • Document Generation      │
                    │  • Corpus Management        │
                    │  • Status Monitoring        │
                    │  • Error Handling           │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼──────────┐  ┌─────────▼──────────┐  ┌─────────▼──────────┐
│   Gemini API       │  │   Vertex AI RAG    │  │   Local Storage    │
│  • Content Gen     │  │  • Corpus Storage  │  │  • Metadata        │
│  • Model Access    │  │  • File Upload     │  │  • Logs            │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

### Data Flow
1. **Document Templates** → JSON configuration files
2. **Generation** → Gemini API processes templates
3. **Storage** → Local files with metadata tracking
4. **Upload** → Vertex AI RAG corpus creation
5. **Retrieval** → Chat interface queries corpus

---

## 📊 Monitoring & Observability

### Built-in Metrics
- 📈 Generation success rates
- ⏱️ Processing times per document
- 📊 Corpus health status
- 🔍 Error tracking and categorization

### Health Endpoints
```bash
# System health
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/api/status
```

### Logging
Structured logging with multiple levels:
- 📝 **INFO**: Normal operations
- ⚠️ **WARNING**: Recoverable issues
- ❌ **ERROR**: Failed operations
- 🔍 **DEBUG**: Detailed diagnostics

---

## 🚢 Deployment

### Docker Deployment (Coming Soon)
Docker support is currently being prepared and will be available in a future release.

```bash
# Docker commands will be available soon:
# make docker-build
# make docker-run
# docker-compose up -d
```

### Cloud Run Deployment
```bash
# Deploy to Google Cloud Run
gcloud run deploy gdg-menorca-rag \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Environment Variables for Production
```env
# Required
GEMINI_KEY=your_production_key
GOOGLE_CLOUD_PROJECT=your-prod-project
GOOGLE_CLOUD_LOCATION=us-central1
CORPUS_DISPLAY_NAME=prod-hotel-docs

# Optional
GRADIO_SERVER_PORT=8080
GRADIO_SERVER_NAME=0.0.0.0
LOG_LEVEL=INFO
MAX_CONCURRENT_GENERATIONS=5
```

---

## 🔧 Configuration

### `rag_config.json`
```json
{
  "model": "gemini-2.5-flash",
  "temperature": 0.7,
  "max_tokens": 8192,
  "batch_size": 10,
  "retry_attempts": 3,
  "timeout_seconds": 30,
  "generation_settings": {
    "include_metadata": true,
    "add_timestamps": true,
    "validate_content": true
  },
  "backup_settings": {
    "auto_backup": true,
    "backup_frequency": "daily",
    "max_backups": 7
  }
}
```

### Document Templates (`hotel_chatbot_documents.json`)
```json
[
  {
    "title": "Hotel Overview",
    "prompt": "Create a comprehensive overview of GDG Menorca Resort including location, history, and unique features."
  },
  {
    "title": "Room Types and Amenities",
    "prompt": "Detail all room types available at GDG Menorca Resort with amenities and capacity."
  }
]
```

---

## 🎯 Best Practices

### Development Workflow
1. **Always test locally first**: Use `--dry-run` flags
2. **Monitor logs**: Check `rag_corpus.log` regularly
3. **Backup before major changes**: Use built-in backup system
4. **Validate status**: Run `status` command before operations

### Production Guidelines
1. **Resource Limits**: Configure appropriate timeouts and batch sizes
2. **Error Handling**: Monitor error rates and adjust retry settings
3. **Backup Strategy**: Enable automatic backups
4. **Health Monitoring**: Set up alerts on health endpoints

### Security Considerations
1. **API Keys**: Never commit keys to version control
2. **Access Control**: Use IAM roles for Vertex AI access
3. **Network Security**: Configure appropriate firewall rules
4. **Audit Logging**: Enable Cloud Audit Logs

---

## 🎨 Customization

### Adding New Document Types
1. Edit `hotel_chatbot_documents.json`
2. Add new document template
3. Run generation: `python rag_manager.py generate`

### Custom Prompts
```json
{
  "title": "Custom Content",
  "prompt": "Your custom prompt here. Remember to include the hotel name: GDG Menorca Resort"
}
```

### Extending the CLI
```python
@cli.command()
@click.option('--custom-param', help='Custom parameter')
def custom_command(custom_param):
    """Custom command description"""
    manager = HotelRAGManager()
    # Your custom logic here
```

---

## 📈 Performance Optimization

### Large Document Sets
- **Batch Processing**: Adjust `batch_size` in config
- **Concurrent Generation**: Set `MAX_CONCURRENT_GENERATIONS`
- **Memory Management**: Monitor memory usage during generation

### API Rate Limits
- **Retry Logic**: Built-in exponential backoff
- **Request Throttling**: Configurable delays between requests
- **Error Recovery**: Automatic retry on transient failures

---

## 🛟 Troubleshooting

### Common Issues

#### "Corpus not found" Error
```bash
# Check if corpus exists
python rag_manager.py status

# Create new corpus
python rag_manager.py generate --interactive
```

#### Generation Failures
```bash
# Check logs for details
python rag_manager.py logs

# Retry with debug logging
LOG_LEVEL=DEBUG python rag_manager.py generate
```

#### Upload Errors
```bash
# Verify permissions
gcloud auth list

# Check quota limits
gcloud quota list --project=your-project
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python rag_manager.py generate --interactive
```

---

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
make dev-install

# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### Code Style
- **Black** formatting
- **Flake8** linting
- **Type hints** for all functions
- **Comprehensive docstrings**

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Vertex AI** for RAG capabilities
- **Gradio** for the web
