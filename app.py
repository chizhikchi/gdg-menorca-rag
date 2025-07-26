"""Enhanced app.py with RAG Manager Integration

This enhanced version includes:
- Corpus status monitoring
- Health checks
- Better error handling
- Status dashboard
- Integration with the RAG manager
"""

import base64
from google import genai
from google.genai import types
import gradio as gr
import utils
from datetime import datetime
import logging
from pathlib import Path

# Import Rich console for beautiful output
from rich.console import Console
console = Console()

# Import our enhanced RAG manager
try:
    from rag_manager import HotelRAGManager, CorpusStatus
    RAG_MANAGER_AVAILABLE = True
except ImportError:
    RAG_MANAGER_AVAILABLE = False
    logging.warning("RAG Manager not available. Some features will be disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG manager if available
rag_manager = None
if RAG_MANAGER_AVAILABLE:
    try:
        rag_manager = HotelRAGManager()
    except Exception as e:
        logger.error(f"Failed to initialize RAG manager: {e}")
        RAG_MANAGER_AVAILABLE = False


def get_corpus_status_info():
    """Get corpus status information for display"""
    if not RAG_MANAGER_AVAILABLE or not rag_manager:
        return "⚠️ RAG Manager not available", "warning"
    
    try:
        corpus, status = rag_manager.get_corpus_status()
        
        status_messages = {
            CorpusStatus.COMPLETE: ("✅ Corpus is ready and complete", "success"),
            CorpusStatus.PARTIAL: ("⚠️ Corpus is partially loaded", "warning"),
            CorpusStatus.EMPTY: ("📭 Corpus exists but is empty", "warning"),
            CorpusStatus.NOT_FOUND: ("❌ Corpus not found - run setup first", "error"),
            CorpusStatus.ERROR: ("⚠️ Error accessing corpus", "error")
        }
        
        return status_messages.get(status, ("❓ Unknown status", "warning"))
        
    except Exception as e:
        logger.error(f"Error checking corpus status: {e}")
        return "⚠️ Error checking corpus status", "error"


def create_admin_interface():
    """Create admin interface for corpus management"""
    if not RAG_MANAGER_AVAILABLE:
        return gr.HTML("❌ RAG Manager not available")
    
    with gr.Column() as admin_interface:
        gr.Markdown("## 🔧 Corpus Management")
        
        # Status display
        status_display = gr.HTML()
        
        # Action buttons
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
            generate_btn = gr.Button("🏗️ Generate Corpus", variant="primary")
            cleanup_btn = gr.Button("🧹 Cleanup", variant="stop")
        
        # Logs display
        logs_display = gr.Textbox(
            label="Recent Logs",
            lines=10,
            max_lines=20,
            interactive=False
        )
        
        def refresh_status():
            message, status_type = get_corpus_status_info()
            color = {
                "success": "green",
                "warning": "orange", 
                "error": "red"
            }.get(status_type, "gray")
            
            html = f"""
            <div style="padding: 15px; border-radius: 8px; background-color: {color}15; border-left: 4px solid {color};">
                <strong>Corpus Status:</strong> {message}<br>
                <small>Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
            </div>
            """
            return html
        
        def generate_corpus():
            try:
                import asyncio
                
                # Generate documents
                success = asyncio.run(rag_manager.generate_documents(interactive=False))
                
                if success:
                    # Create corpus and upload
                    corpus = rag_manager.create_corpus()
                    if corpus:
                        upload_success = rag_manager.upload_documents(corpus)
                        if upload_success:
                            return "✅ Corpus generated and uploaded successfully!"
                        else:
                            return "⚠️ Documents generated but upload failed"
                    else:
                        return "⚠️ Documents generated but corpus creation failed"
                else:
                    return "❌ Document generation failed"
                    
            except Exception as e:
                logger.error(f"Error in corpus generation: {e}")
                return f"❌ Error: {str(e)}"
        
        def cleanup_corpus():
            try:
                rag_manager.cleanup(dry_run=False)
                return "✅ Cleanup completed"
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                return f"❌ Cleanup error: {str(e)}"
        
        def get_recent_logs():
            try:
                log_file = Path("rag_corpus.log")
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        return ''.join(lines[-50:])  # Last 50 lines
                else:
                    return "No logs available"
            except Exception as e:
                return f"Error reading logs: {e}"
        
        # Event handlers
        refresh_btn.click(refresh_status, outputs=status_display)
        generate_btn.click(generate_corpus, outputs=gr.Textbox(label="Generation Result"))
        cleanup_btn.click(cleanup_corpus, outputs=gr.Textbox(label="Cleanup Result"))
        
        # Initial status load
        status_display.value = refresh_status()
        logs_display.value = get_recent_logs()
    
    return admin_interface


def generate(
    message,
    history: list[gr.ChatMessage],
    request: gr.Request
):
    """Enhanced generate function with corpus validation"""
    
    # Skip key validation for local development
    # Only validate key if we're running in production (Cloud Run)
    import os
    if os.getenv('CLOUD_RUN_SERVICE') or os.getenv('K_SERVICE'):
        # We're running on Cloud Run, validate key
        validate_key_result = utils.validate_key(request)
        if validate_key_result is not None:
            yield validate_key_result
            return
    # For local development, skip key validation
    
    # Check corpus status if RAG manager is available
    if RAG_MANAGER_AVAILABLE and rag_manager:
        try:
            corpus, status = rag_manager.get_corpus_status()
            if status not in [CorpusStatus.COMPLETE, CorpusStatus.PARTIAL]:
                status_msg, _ = get_corpus_status_info()
                yield gr.Error(f"Corpus not ready: {status_msg}. Please use the admin panel to generate the corpus first.")
                return
        except Exception as e:
            logger.warning(f"Could not check corpus status: {e}")
            # Continue anyway - corpus might be working despite check failure
    
    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project="model-fastness-466612-t7",
        location="global",
    )
    
    # Your existing message setup
    msg2_text1 = types.Part.from_text(text=f"""Sí, el GDG Menorca Resort podría ser un buen lugar para usted y sus 3 hijos.

Varias de sus habitaciones pueden alojar a familias:
*   **Suite Ejecutiva:** Máximo 2 adultos + 2 niños o 3 adultos.
*   **Suite Familiar:** Máximo 2 adultos + 3 niños o 4 adultos.

La Suite Familiar tiene entre 70-90 m², lo que la hace una opción espaciosa.

El hotel también ofrece:
*   Estacionamiento gratuito para huéspedes [3].
*   Piscinas adaptadas para todas las edades [4].
*   Actividades dedicadas para familias [4].

Además, el hotel cuenta con habitaciones de hasta 160 m² como el Penthouse [2].

Por favor, tenga en cuenta que para llevar un coche al hotel, el GDG Menorca Resort dispone de estacionamiento para sus huéspedes [7].""")

    model = "gemini-2.5-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Vengo en coche con 3 hijos y quiero una habitación grande, es este hotel un buen sitio para mí?""")
            ]
        ),
        types.Content(
            role="model",
            parts=[msg2_text1]
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""Hay programas de entretenimiento para adultos?""")
            ]
        ),
    ]

    # Add conversation history
    for prev_msg in history:
        role = "user" if prev_msg["role"] == "user" else "model"
        parts = utils.get_parts_from_message(prev_msg["content"])
        if parts:
            contents.append(types.Content(role=role, parts=parts))

    if message:
        contents.append(
            types.Content(role="user", parts=utils.get_parts_from_message(message))
        )

    # RAG tools configuration - make it optional
    tools = []
    
    # Only add RAG if corpus exists (you can configure this)
    try:
        # Try to use RAG corpus if available
        rag_corpus_name = rag_manager.metadata.name
        if rag_corpus_name is None:
          rag_corpus_name = os.getenv('RAG_CORPUS_ID')
        if rag_corpus_name:
            tools = [
                types.Tool(
                    retrieval=types.Retrieval(
                        vertex_rag_store=types.VertexRagStore(
                            rag_resources=[
                                types.VertexRagStoreRagResource(
                                    rag_corpus=rag_corpus_name
                                )
                            ],
                        )
                    )
                )
            ]
            logger.info("Using RAG corpus for enhanced responses")
        else:
            logger.info("No RAG corpus configured, using base model only")
            
    except Exception as e:
        logger.warning(f"RAG corpus not available: {e}")
        tools = []  # Fall back to no RAG
    
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        tools=tools,
    )

    # Generate response with error handling
    try:
        results = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0] and chunk.candidates[0].content:
                results.extend(
                    utils.convert_content_to_gr_type(chunk.candidates[0].content)
                )
                if results:
                    yield results
                    
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        # Return a proper error message instead of gr.Error for ChatInterface
        error_message = f"❌ Error: The RAG corpus doesn't exist yet. Please create it first using the RAG manager.\n\nDetailed error: {str(e)}"
        yield error_message


# Health check endpoint
def health_check():
    """Health check for monitoring"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "rag_manager": RAG_MANAGER_AVAILABLE,
        }
    }
    
    if RAG_MANAGER_AVAILABLE and rag_manager:
        try:
            corpus, corpus_status = rag_manager.get_corpus_status()
            status["components"]["corpus"] = corpus_status.value
            if corpus_status != CorpusStatus.COMPLETE:
                status["status"] = "degraded"
        except Exception as e:
            status["components"]["corpus"] = "error"
            status["status"] = "degraded"
            status["error"] = str(e)
    
    return status


# Enhanced Gradio interface
with gr.Blocks(title="🏨 GDG Menorca Resort Assistant") as demo:
    # Header
    with gr.Row():
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏨 GDG Menorca Resort - AI Assistant</h1>
            <p>Your intelligent hotel concierge powered by advanced RAG technology</p>
        </div>
        """)
    
    # Status bar
    if RAG_MANAGER_AVAILABLE:
        status_message, status_type = get_corpus_status_info()
        status_color = {
            "success": "#d4edda",
            "warning": "#fff3cd", 
            "error": "#f8d7da"
        }.get(status_type, "#e2e3e5")
        
        with gr.Row():
            gr.HTML(f"""
            <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>System Status:</strong> {status_message}
            </div>
            """)
    
    # Public access warning
    with gr.Row():
        gr.HTML("""
        <div style="background-color: #fffacd; border: 1px solid #eedc82; padding: 20px; margin: 20px; border-radius: 5px; color: #8b4513; font-weight: bold; text-align: center;">
          <span style="margin-right: 10px;">⚠️</span>
          Warning: This app allows unauthenticated access by default. Avoid using it for sensitive data. Access control is coming soon.
        </div>
        """)
    
    # Main content
    with gr.Row():
        # Left column - Information and Admin
        with gr.Column(scale=1):
            # Welcome section
            with gr.Row():
                gr.HTML("<h2>Welcome to Vertex AI GenAI App!</h2>")
            with gr.Row():
                gr.HTML("""This prototype was built using your Vertex AI Studio prompt.
                    Follow the steps and recommendations below to begin.""")
            with gr.Row():
                gr.HTML("""
                <span>Next steps:</span>
                <ul style="list-style-position: outside; margin-left: 1em;">
                  <li>Go to Cloud Run source code editor to customize application code.</li>
                  <li>Go to your Vertex AI Studio prompt to update and redeploy it.</li>
                  <li>Go to Cloud Run Security settings to turn off unauthenticated access when it's not needed.</li>
                </ul>
                """)
            
            # Admin interface (if RAG manager available)
            if RAG_MANAGER_AVAILABLE:
                with gr.Accordion("🔧 Admin Panel", open=False):
                    create_admin_interface()

        # Right column - Chat Interface
        with gr.Column(scale=2, variant="panel"):
            gr.ChatInterface(
                fn=generate,
                title="Hotel Recommendation Chatbot",
                description="Ask me anything about GDG Menorca Resort!",
                type="messages",
                multimodal=True,
                examples=[
                    "¿Qué servicios ofrece el hotel?",
                    "¿Hay actividades para niños?",
                    "¿Cuáles son los tipos de habitaciones disponibles?",
                    "¿El hotel tiene piscina?",
                    "¿Hay restaurantes en el hotel?"
                ]
            )

# Add custom CSS for better styling
demo.css = """
.gradio-container {
    max-width: 1200px !important;
}

.admin-panel {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-healthy {
    background-color: #28a745;
}

.status-warning {
    background-color: #ffc107;
}

.status-error {
    background-color: #dc3545;
}
"""

def create_health_check_app():
    """Create FastAPI app with health check endpoint"""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="GDG Menorca RAG Health Check")
        
        @app.get("/health")
        def get_health():
            return JSONResponse(content=health_check())
        
        @app.get("/api/status")
        def get_status():
            """Detailed status endpoint"""
            status_info = health_check()
            if RAG_MANAGER_AVAILABLE and rag_manager:
                try:
                    corpus, corpus_status = rag_manager.get_corpus_status()
                    status_info["corpus_details"] = {
                        "status": corpus_status.value,
                        "name": corpus.name if corpus else None,
                        "document_count": rag_manager.metadata.document_count
                    }
                except Exception as e:
                    status_info["corpus_details"] = {"error": str(e)}
            
            return JSONResponse(content=status_info)
        
        return app
        
    except ImportError:
        logger.warning("FastAPI not available. Health endpoints will not be created.")
        return None


if __name__ == "__main__":
    # Create health check app (optional)
    health_app = create_health_check_app()
    
    if health_app:
        # Run with FastAPI for health endpoints
        import uvicorn
        from threading import Thread
        
        def run_health_server():
            uvicorn.run(health_app, host="0.0.0.0", port=8081, log_level="warning")
        
        # Start health server in background
        health_thread = Thread(target=run_health_server, daemon=True)
        health_thread.start()
        
        print("🏥 Health check server started on http://localhost:8081/health")
    
    # Launch the main Gradio demo
    print("🚀 Starting GDG Menorca Resort Assistant...")
    
    demo.launch(
        show_error=True,
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        show_api=False,
        quiet=False,
        favicon_path=None,
        auth=None  # Add authentication here if needed
    )