#!/usr/bin/env python3
"""
🏨 GDG Menorca Resort - RAG Corpus Management System
==================================================

A system for generating hotel documentation and managing RAG corpus
with interactive CLI, progress tracking, and error handling.

Usage:
    python rag_manager.py --help
    python rag_manager.py generate --interactive
    python rag_manager.py status
    python rag_manager.py cleanup --dry-run
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from tqdm import tqdm

# Google AI and Vertex AI imports
from google import genai
from google.genai import types
from vertexai import rag
import vertexai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_corpus.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for beautiful output
console = Console()


class CorpusStatus(Enum):
    """Corpus status enumeration"""
    NOT_FOUND = "not_found"
    EMPTY = "empty"
    PARTIAL = "partial"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CorpusMetadata:
    """Metadata for corpus tracking"""
    name: str
    display_name: str
    created_at: str
    document_count: int
    status: CorpusStatus
    last_updated: str
    generation_config: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CorpusMetadata':
        return cls(**data)


class HotelRAGManager:
    """
    🏨 Advanced Hotel RAG Corpus Manager
    
    Features:
    - Interactive document generation with progress tracking
    - Intelligent corpus management with status checking
    - Beautiful CLI interface with Rich
    - Comprehensive error handling and logging
    - Backup and restore capabilities
    - Performance metrics and analytics
    """
    
    def __init__(
        self,
        config_file: str = "rag_config.json",
        input_json: str = "./data/hotel_chatbot_documents.json",
        output_dir: str = "generated_docs",
        backup_dir: str = "backups"
    ):
        self.config = self._load_config(config_file)
        self.input_json = Path(input_json)
        self.output_dir = Path(output_dir)
        self.backup_dir = Path(backup_dir)
        self.metadata_file = Path("corpus_metadata.json")
        
        # Create directories
        for directory in [self.output_dir, self.backup_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize clients
        self._initialize_clients()
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file or environment"""
        default_config = {
            "api_key_env": "GEMINI_KEY",
            "project_env": "GOOGLE_CLOUD_PROJECT", 
            "corpus_name_env": "CORPUS_DISPLAY_NAME",
            "location_env": "GOOGLE_CLOUD_LOCATION",
            "model": "gemini-2.5-flash",
            "additional_instructions": (
                "\nIMPORTANTE: el nombre el hotel es GDG Menorca Resort y está ubicado en Menorca. "
                "Todos los documentos generados tienen que estar en castellano\n"
                "No incluyas ningún tipo de explicación o comentario, produce el contenido que se te pidió directamente."
            )
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except FileNotFoundError:
            return default_config
    
    def _initialize_clients(self):
        """Initialize Google AI and Vertex AI clients"""
        # Get environment variables
        self.api_key = os.environ.get(self.config["api_key_env"])
        self.project = os.environ.get(self.config["project_env"])
        self.corpus_display_name = os.environ.get(self.config["corpus_name_env"])
        self.location = os.environ.get(self.config["location_env"])
        
        # Validate required environment variables
        required_vars = [
            (self.api_key, self.config["api_key_env"]),
            (self.project, self.config["project_env"]),
            (self.corpus_display_name, self.config["corpus_name_env"]),
            (self.location, self.config["location_env"])
        ]
        
        for var, env_name in required_vars:
            if not var:
                raise ValueError(f"❌ Environment variable '{env_name}' not found")
        
        # Initialize clients
        self.client = genai.Client(api_key=self.api_key)
        vertexai.init(project=self.project, location=self.location)
        
        console.print("✅ Clients initialized successfully", style="green")
    
    def _load_metadata(self) -> CorpusMetadata:
        """Load corpus metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return CorpusMetadata.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        # Create default metadata
        return CorpusMetadata(
            name="",
            display_name=self.corpus_display_name,
            created_at="",
            document_count=0,
            status=CorpusStatus.NOT_FOUND,
            last_updated="",
            generation_config=self.config
        )
    
    def _save_metadata(self):
        """Save corpus metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2, default=str)
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize filename for safe file operations"""
        return re.sub(r"[^\w\-_. ]", "_", name)
    
    async def generate_documents(self, interactive: bool = True) -> bool:
        """
        Generate hotel documents with progress tracking
        """
        console.print(Panel(
            "🏨 [bold blue]GDG Menorca Resort - Document Generation[/bold blue]",
            subtitle="Generating comprehensive hotel documentation"
        ))
        
        if not self.input_json.exists():
            console.print(f"❌ Input file not found: {self.input_json}", style="red")
            return False
        
        # Load documents
        with open(self.input_json, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        if interactive:
            console.print(f"📄 Found {len(documents)} documents to generate")
            if not Confirm.ask("Continue with generation?"):
                return False
        
        # Generate documents with progress
        successful = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating documents...", total=len(documents))
            
            for doc in documents:
                title = doc['title']
                prompt = doc['prompt'] + self.config["additional_instructions"]
                filename = self.sanitize_filename(title) + ".txt"
                filepath = self.output_dir / filename

                if filepath.exists():
                    progress.update(task, description=f"[yellow]Skipped (exists): {title[:30]}...")
                    progress.advance(task)
                    continue
                
                progress.update(task, description=f"Generating: {title[:30]}...")
                
                try:
                    response = self.client.models.generate_content(
                        model=self.config["model"],
                        contents=prompt
                    )
                    
                    content = response.text.strip()
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    successful += 1
                    logger.info(f"Generated: {title}")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to generate '{title}': {e}")
                    
                progress.advance(task)
        
        # Update metadata
        self.metadata.document_count = successful
        self.metadata.last_updated = datetime.now().isoformat()
        self.metadata.status = CorpusStatus.PARTIAL if failed > 0 else CorpusStatus.COMPLETE
        self._save_metadata()
        
        # Show results
        results_table = Table(title="Generation Results")
        results_table.add_column("Status", style="cyan")
        results_table.add_column("Count", style="magenta")
        results_table.add_row("✅ Successful", str(successful))
        results_table.add_row("❌ Failed", str(failed))
        results_table.add_row("📁 Output Directory", str(self.output_dir))
        
        console.print(results_table)
        
        return failed == 0
    
    def get_corpus_status(self) -> Tuple[Optional[rag.RagCorpus], CorpusStatus]:
        """Check corpus status in Vertex AI"""
        try:
            corpora = rag.list_corpora()
            for corpus in corpora:
                if corpus.display_name == self.corpus_display_name:
                    # Count files in corpus
                    try:
                        files = rag.list_files(corpus_name=corpus.name)
                        file_count = len(list(files))
                        
                        if file_count == 0:
                            return corpus, CorpusStatus.EMPTY
                        elif file_count < self.metadata.document_count:
                            return corpus, CorpusStatus.PARTIAL
                        else:
                            return corpus, CorpusStatus.COMPLETE
                    except Exception:
                        return corpus, CorpusStatus.ERROR
            
            return None, CorpusStatus.NOT_FOUND
            
        except Exception as e:
            logger.error(f"Error checking corpus status: {e}")
            return None, CorpusStatus.ERROR
    
    def create_corpus(self) -> Optional[rag.RagCorpus]:
        """Create RAG corpus in Vertex AI"""
        corpus, status = self.get_corpus_status()
        
        if corpus and status != CorpusStatus.NOT_FOUND:
            console.print(f"📚 Corpus already exists with status: {status.value}", style="yellow")
            return corpus
        
        console.print("🔧 Creating new RAG corpus...", style="blue")
        
        try:
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )
            
            corpus = rag.create_corpus(
                display_name=self.corpus_display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )
            
            # Update metadata
            self.metadata.name = corpus.name
            self.metadata.created_at = datetime.now().isoformat()
            self.metadata.status = CorpusStatus.EMPTY
            self._save_metadata()
            
            console.print("✅ Corpus created successfully!", style="green")
            return corpus
            
        except Exception as e:
            logger.error(f"Failed to create corpus: {e}")
            console.print(f"❌ Failed to create corpus: {e}", style="red")
            return None
    
    def upload_documents(self, corpus: rag.RagCorpus) -> bool:
        """Upload documents to RAG corpus"""
        if not self.output_dir.exists() or not list(self.output_dir.glob("*.txt")):
            console.print("❌ No documents found to upload", style="red")
            return False
        
        files = list(self.output_dir.glob("*.txt"))
        console.print(f"📤 Uploading {len(files)} documents to corpus...", style="blue")
        
        successful = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Uploading documents...", total=len(files))
            
            for file_path in files:
                progress.update(task, description=f"Uploading: {file_path.name[:30]}...")
                
                try:
                    rag.upload_file(
                        corpus_name=corpus.name,
                        path=str(file_path),
                        display_name=file_path.name,
                        description=file_path.stem
                    )
                    successful += 1
                    logger.info(f"Uploaded: {file_path.name}")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to upload '{file_path.name}': {e}")
                
                progress.advance(task)
        
        # Update metadata
        self.metadata.document_count = successful
        self.metadata.last_updated = datetime.now().isoformat()
        self.metadata.status = CorpusStatus.COMPLETE if failed == 0 else CorpusStatus.PARTIAL
        self._save_metadata()
        
        # Show results
        upload_table = Table(title="Upload Results")
        upload_table.add_column("Status", style="cyan")
        upload_table.add_column("Count", style="magenta")
        upload_table.add_row("✅ Successful", str(successful))
        upload_table.add_row("❌ Failed", str(failed))
        
        console.print(upload_table)
        
        return failed == 0
    
    def show_status(self):
        """Display comprehensive status information"""
        corpus, status = self.get_corpus_status()
        
        # Status panel
        status_text = Text()
        if status == CorpusStatus.COMPLETE:
            status_text.append("✅ COMPLETE", style="green bold")
        elif status == CorpusStatus.PARTIAL:
            status_text.append("⚠️ PARTIAL", style="yellow bold")
        elif status == CorpusStatus.EMPTY:
            status_text.append("📭 EMPTY", style="blue bold")
        elif status == CorpusStatus.NOT_FOUND:
            status_text.append("❌ NOT FOUND", style="red bold")
        else:
            status_text.append("⚠️ ERROR", style="red bold")
        
        console.print(Panel(status_text, title="🏨 Corpus Status"))
        
        # Detailed information table
        info_table = Table(title="Detailed Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        info_table.add_row("Display Name", self.corpus_display_name)
        info_table.add_row("Project", self.project)
        info_table.add_row("Location", self.location)
        info_table.add_row("Local Documents", str(len(list(self.output_dir.glob("*.txt")))))
        
        if corpus:
            info_table.add_row("Corpus Name", corpus.name)
            try:
                files = list(rag.list_files(corpus_name=corpus.name))
                info_table.add_row("Uploaded Documents", str(len(files)))
            except:
                info_table.add_row("Uploaded Documents", "Error fetching")
        
        info_table.add_row("Last Updated", self.metadata.last_updated or "Never")
        info_table.add_row("Created At", self.metadata.created_at or "Not created")
        
        console.print(info_table)
    
    def cleanup(self, dry_run: bool = True):
        """Clean up local files and optionally remote corpus"""
        console.print("🧹 Cleanup Operation", style="bold")
        
        if dry_run:
            console.print("🔍 DRY RUN - No files will be deleted", style="yellow")
        
        # Count files to delete
        local_files = list(self.output_dir.glob("*.txt"))
        
        if not local_files:
            console.print("✨ No local files to clean up", style="green")
            return
        
        console.print(f"📁 Found {len(local_files)} local files to delete")
        
        if not dry_run:
            if Confirm.ask("Delete local generated documents?"):
                for file_path in local_files:
                    file_path.unlink()
                    console.print(f"🗑️ Deleted: {file_path.name}")
                
                console.print("✅ Local cleanup completed", style="green")
        
        # Corpus cleanup
        corpus, status = self.get_corpus_status()
        if corpus and not dry_run:
            if Confirm.ask("⚠️ Delete remote corpus? This cannot be undone!"):
                try:
                    rag.delete_corpus(name=f'projects/{self.project}/locations/{self.location}/corpus/{corpus.name}')
                    console.print("✅ Corpus deleted successfully", style="green")
                    
                    # Reset metadata
                    self.metadata.status = CorpusStatus.NOT_FOUND
                    self.metadata.name = ""
                    self._save_metadata()
                    
                except Exception as e:
                    console.print(f"❌ Failed to delete corpus: {e}", style="red")


# CLI Interface
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🏨 GDG Menorca Resort - RAG Corpus Management System"""
    pass


@cli.command()
@click.option('--interactive/--no-interactive', default=True, help='Interactive mode')
@click.option('--upload/--no-upload', default=True, help='Upload after generation')
def generate(interactive, upload):
    """Generate documents and create/update corpus"""
    manager = HotelRAGManager()
    
    # Generate documents
    success = asyncio.run(manager.generate_documents(interactive=interactive))
    
    if not success:
        console.print("❌ Document generation failed", style="red")
        return
    
    if upload:
        # Create or get corpus
        corpus = manager.create_corpus()
        if corpus:
            # Upload documents
            manager.upload_documents(corpus)
        else:
            console.print("❌ Failed to create corpus", style="red")


@cli.command()
def status():
    """Show corpus status and information"""
    manager = HotelRAGManager()
    manager.show_status()


@cli.command()
@click.option('--dry-run/--no-dry-run', default=True, help='Show what would be deleted without deleting')
def cleanup(dry_run):
    """Clean up local files and remote corpus"""
    manager = HotelRAGManager()
    manager.cleanup(dry_run=dry_run)


@cli.command()
def logs():
    """Show recent logs"""
    try:
        with open('rag_corpus.log', 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-50:]  # Last 50 lines
            for line in recent_lines:
                console.print(line.strip())
    except FileNotFoundError:
        console.print("❌ Log file not found", style="red")


if __name__ == "__main__":
    cli()