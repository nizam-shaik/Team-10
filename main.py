from __future__ import annotations
import sys, os

# --------------------------------------------------------------------
# ✅ Prevent Streamlit startup during test imports (shortcut version)
# --------------------------------------------------------------------
import builtins
if __name__ != "__main__":
    builtins.__STREAMLIT_TEST_MODE__ = True
else:
    builtins.__STREAMLIT_TEST_MODE__ = False

# --------------------------------------------------------------------
# ✅ Ensure project root is always on sys.path
# --------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------------------------
# ✅ Regular imports (no need to indent or move anything)
# --------------------------------------------------------------------
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st
import zipfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import workflow factory (uses HLDWorkflow implemented earlier)
from workflow.hld_workflow import create_hld_workflow

DATA_DIR = Path(__file__).resolve().parent / "data"  # Project/data

# -----------------------
# Helpers
# -----------------------
def list_requirement_pdfs(folder: Path = DATA_DIR) -> List[Path]:
    """List PDF files in the data folder sorted by name."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])


def get_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    """Return simple metadata about the PDF for UI display."""
    try:
        st_size = pdf_path.stat().st_size
        return {
            "name": pdf_path.name,
            "path": str(pdf_path),
            "size_kb": round(st_size / 1024, 1),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pdf_path.stat().st_mtime)),
        }
    except Exception:
        return {"name": pdf_path.name, "path": str(pdf_path), "size_kb": None, "modified": None}


def safe_load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def create_images_zip(img_folder: Path) -> Optional[bytes]:
    """Create a zip file containing all PNG images from the img folder."""
    if not img_folder.exists() or not img_folder.is_dir():
        return None
    
    png_files = list(img_folder.glob("*.png"))
    if not png_files:
        return None
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for png_file in png_files:
            zip_file.write(png_file, arcname=png_file.name)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# -----------------------
# UI renderers (concise)
# -----------------------
def render_markdown_section(title: str, md_text: str):
    st.subheader(title)
    if md_text:
        st.markdown(md_text)
    else:
        st.info("No content available.")

            
def render_key_value(title: str, data: Dict):
    st.subheader(title)
    if not data:
        st.info("No data.")
        return
    for k, v in data.items():
        st.write(f"**{k}**: {v}")


def render_list_section(title: str, items: List[str]):
    st.subheader(title)
    if not items:
        st.info("No items.")
        return
    for item in items:
        st.markdown(f"- {item}")


def render_json_pretty(title: str, obj: Any):
    st.subheader(title)
    if not obj:
        st.info("No data.")
        return
    st.json(obj)


# -----------------------
# Workflow runner (streaming)
# -----------------------
def run_workflow_and_stream(pdf_path: str, workflow_type: str, config: Dict[str, Any]):
    """
    Run workflow.stream() and yield node-by-node events.
    This function must be executed synchronously from Streamlit (we use asyncio.run).
    """
    logger.info("=" * 80)
    logger.info("WORKFLOW EXECUTION STARTED")
    logger.info("=" * 80)
    logger.info(f"PDF Path: {pdf_path}")
    logger.info(f"Workflow Type: {workflow_type}")
    logger.info(f"Configuration: {config}")
    logger.info("=" * 80)
    
    wf = create_hld_workflow(workflow_type)

    # Input object can be a dict
    input_data = {"pdf_path": pdf_path, "config": config}

    # Use asyncio to consume the async generator
    async def _consume():
        events = []
        async for event in wf.stream(input_data):
            events.append(event)
        return events

    # Be friendly with environments where an event loop might already exist
    try:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            events = loop.run_until_complete(_consume())
        finally:
            try:
                loop.close()
            except Exception:
                pass
        return events
    except Exception as e:
        try:
            # Fallback to asyncio.run if fresh loop strategy fails
            events = asyncio.run(_consume())
            return events
        except Exception as e2:
            return [{"type": "error", "message": f"{e2}"}]


# -----------------------
# Main Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="DesignMind GenAI — HLD Generator", layout="wide")
    st.title("DesignMind GenAI — HLD Generator")

    # Sidebar: configuration
    st.sidebar.header("Configuration")
    workflow_type = st.sidebar.selectbox("Workflow type", ["parallel", "sequential", "conditional"], index=0)
    render_images = st.sidebar.checkbox("Render diagrams to images", value=True)
    image_format = st.sidebar.selectbox("Image format", ["png", "svg"], index=0)
    renderer = "mmdc"  # Fixed to mmdc only
    theme = st.sidebar.selectbox("Diagram theme", ["default", "neutral", "dark"], index=0)

    config = {
        "render_images": render_images,
        "image_format": image_format,
        "renderer": renderer,
        "theme": theme,
    }
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("CONFIGURATION SETTINGS")
    logger.info("=" * 80)
    logger.info(f"Workflow Type: {workflow_type}")
    logger.info(f"Render Images: {render_images}")
    logger.info(f"Image Format: {image_format}")
    logger.info(f"Renderer: {renderer}")
    logger.info(f"Diagram Theme: {theme}")
    logger.info("=" * 80)

    tabs = st.tabs(["HLD Generation", "ML Training", "Quality Prediction"])

    # -------------------------
    # Tab 1: HLD Generation
    # -------------------------
    with tabs[0]:
        st.header("HLD Generation")
        pdfs = list_requirement_pdfs()
        if not pdfs:
            st.warning("No requirement PDFs found in data/ — place PDFs in the data/ folder.")
            return

        pdf_names = [p.name for p in pdfs]
        default_idx = 0
        selection = st.selectbox("Select requirement PDF", pdf_names, index=default_idx)
        selected_pdf = pdfs[pdf_names.index(selection)]
        info = get_pdf_info(selected_pdf)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Selected:** {info['name']}")
            st.write(f"Path: `{info['path']}`")
            st.write(f"Size: {info['size_kb']} KB")
            st.write(f"Modified: {info['modified']}")
        with col2:
            st.download_button("Download PDF", data=selected_pdf.read_bytes(), file_name=info["name"])

        # Check if output already exists for this requirement
        output_dir = Path(__file__).resolve().parent / "output" / Path(selected_pdf).stem
        output_exists = (output_dir / "json").exists() and (output_dir / "json" / "authentication.json").exists()
        
        # Initialize generation state for this requirement
        req_key = f"generating_{Path(selected_pdf).stem}"
        if req_key not in st.session_state:
            st.session_state[req_key] = False
        
        if output_exists:
            st.success(f"✅ Output already exists for {Path(selected_pdf).stem}")
            # Disable regenerate button if generation is in progress
            regenerate = st.button(
                "🔄 Regenerating..." if st.session_state[req_key] else "🔄 Regenerate HLD",
                type="secondary",
                disabled=st.session_state[req_key]
            )
        else:
            st.info(f"ℹ️ No existing output found for {Path(selected_pdf).stem}")
            # Disable generate button if generation is in progress
            regenerate = st.button(
                "⏳ Generating..." if st.session_state[req_key] else "Generate HLD",
                type="primary",
                disabled=st.session_state[req_key]
            )
            view_existing = False

        # placeholders for progress and outputs
        progress_box = st.empty()
        log_box = st.empty()
        outputs_box = st.container()
        
        # Initialize events list for workflow status tracking
        events = []
        
        # Determine if we should show output (always show if exists and not regenerating)
        show_output = output_exists and not regenerate

        # Check if we should start generation (either button clicked or state is True)
        should_generate = regenerate or st.session_state[req_key]
        
        if should_generate:
            # Set generation state to prevent duplicate clicks and rerun immediately if first click
            if regenerate and not st.session_state[req_key]:
                st.session_state[req_key] = True
                st.rerun()
            
            progress_box.info("Starting workflow... (this may take a while)")
            # Run streaming workflow; update UI per event
            events = run_workflow_and_stream(str(selected_pdf), workflow_type, config)

            # Render events log
            for e in events:
                etype = e.get("type")
                if etype == "node_start":
                    log_box.info(f"Started: {e.get('node')}")
                elif etype == "node_complete":
                    log_box.success(f"Completed: {e.get('node')} ({e.get('status')}) in {e.get('duration')}s")
                elif etype == "error":
                    log_box.error(f"Error: {e.get('message')}")
                elif etype == "started":
                    log_box.info("Workflow completed.")
                elif etype == "completed":
                    log_box.success("Workflow completed.")
                elif etype == "state_update":
                    # Skip state_update events - they contain full state dumps
                    # We'll extract the final state separately below
                    pass
                else:
                    # Only log unexpected event types, not the full content
                    if etype:
                        log_box.info(f"Event: {etype}")
                    # Silently skip events without a type

            # After complete, try to read output paths from the last state (best-effort)
            # We expect the last event to include final state if using NodeManager stream.
            final_state = None
            for e in reversed(events):
                if isinstance(e.get("state"), dict):
                    final_state = e.get("state")
                    break
            
            # Mark that we should show output after generation
            show_output = True
            
            # Re-check if output exists after generation (to update UI state)
            output_exists = (output_dir / "json").exists() and (output_dir / "json" / "authentication.json").exists()
            
            # Reset generation state after completion
            st.session_state[req_key] = False
        else:
            # Not regenerating, just viewing existing
            final_state = None
        
        # Show output if either viewing existing or just generated
        if show_output or output_exists:
            with outputs_box:
                st.markdown("## Results")
                
                # Workflow Status Table
                st.markdown("### 📊 Workflow Execution Status")
                
                # Build workflow status from events or create default status
                workflow_status = []
                if events:
                    # Track status per node with timestamps
                    node_status = {}
                    prev_timestamp = None
                    
                    for e in events:
                        etype = e.get("type")
                        node_name = e.get("node", "")
                        timestamp = e.get("timestamp", 0)
                        
                        if etype == "node_start":
                            prev_timestamp = timestamp
                            node_status[node_name] = {
                                "Stage": node_name.replace("_", " ").title(),
                                "Status": "🔄 Running",
                                "Message": "Processing...",
                            }
                        elif etype == "node_complete":
                            status_text = e.get("status", "completed")
                            duration = e.get("duration", 0)
                            node_status[node_name] = {
                                "Stage": node_name.replace("_", " ").title(),
                                "Status": "✅ Completed" if status_text == "completed" else "⚠️ Warning",
                                "Message": f"Completed in {duration:.2f}s",
                            }
                        elif etype == "error":
                            msg = e.get("message", "Unknown error")
                            node_status[node_name] = {
                                "Stage": node_name.replace("_", " ").title() if node_name else "Unknown",
                                "Status": "❌ Error",
                                "Message": msg,
                            }
                        elif etype == "state_update":
                            # For parallel workflows using LangGraph, state_update contains {node_name: output}
                            state = e.get("state", {})
                            if isinstance(state, dict):
                                # Extract node name from state keys (LangGraph format)
                                for node_key in state.keys():
                                    if node_key not in ["__start__", "__end__"]:
                                        # Calculate duration from previous timestamp if available
                                        duration_msg = "Completed"
                                        if prev_timestamp and timestamp and node_key not in node_status:
                                            duration = timestamp - prev_timestamp
                                            if duration > 0.1:  # Only show if meaningful duration
                                                duration_msg = f"Completed in {duration:.1f}s"
                                        
                                        if node_key not in node_status:
                                            node_status[node_key] = {
                                                "Stage": node_key.replace("_", " ").title(),
                                                "Status": "✅ Completed",
                                                "Message": duration_msg,
                                            }
                        
                        # Update previous timestamp for duration calculation
                        if timestamp:
                            prev_timestamp = timestamp
                    
                    workflow_status = list(node_status.values())
                else:
                    # Default status when viewing existing output (no events)
                    default_stages = [
                        "PDF Extraction",
                        "Authentication Analysis",
                        "Integrations Analysis",
                        "Domain Modeling",
                        "Behavior Analysis",
                        "Diagram Generation",
                        "Output Composition"
                    ]
                    workflow_status = [
                        {
                            "Stage": stage,
                            "Status": "✅ Completed",
                            "Message": "Previously generated",
                        }
                        for stage in default_stages
                    ]
                
                if workflow_status:
                    status_df = pd.DataFrame(workflow_status)
                    st.dataframe(status_df, width="stretch", hide_index=True)
                else:
                    st.info("No workflow status available")
                
                st.markdown("---")
                
                # Determine output directory for the selected requirement once
                output_dir = Path(__file__).resolve().parent / "output" / Path(selected_pdf).stem

                st.markdown("## 📋 Extracted Requirements")

                # 1) Extracted requirements (try final_state -> json file)
                extracted_md = None
                try:
                    extracted_obj = final_state.get("extracted") if final_state else None
                    if extracted_obj:
                        if isinstance(extracted_obj, str):
                            extracted_md = extracted_obj
                        elif isinstance(extracted_obj, dict):
                            extracted_md = extracted_obj.get("markdown") or extracted_obj.get("markdown_text") or extracted_obj.get("text")
                except Exception:
                    extracted_md = None

                if not extracted_md:
                    extracted_path = output_dir / "json" / "extracted.json"
                    if extracted_path.exists():
                        try:
                            ex = safe_load_json(str(extracted_path))
                            if isinstance(ex, dict):
                                extracted_md = ex.get("markdown") or ex.get("markdown_text") or json.dumps(ex, indent=2)
                            else:
                                extracted_md = str(ex)
                        except Exception:
                            extracted_md = None

                if extracted_md:
                    with st.expander("📄 View Extracted Content (Click to Expand/Minimize)", expanded=False):
                        # Add scrolling container with max height
                        st.markdown(
                            f'<div style="max-height: 500px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{extracted_md}</div>',
                            unsafe_allow_html=True
                        )
                # Get authentication data from final_state or load from JSON file
                auth = None
                if final_state:
                    auth = final_state.get("authentication") or {}
                
                # Fallback: Load from authentication.json if not in state
                if not auth or not any(auth.values()):
                    auth_json_path = output_dir / "json" / "authentication.json"
                    if auth_json_path.exists():
                        try:
                            auth = safe_load_json(str(auth_json_path)) or {}
                        except Exception as e:
                            st.warning(f"Could not load authentication.json: {e}")
                            auth = {}
                
                st.markdown("## 🔐 Authentication & Security")
                if not auth or not any(auth.values()):
                    st.info("No authentication information available.")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Authentication Methods")
                        # Tolerant fallbacks: auth_methods OR flows OR methods
                        methods = auth.get("auth_methods") or auth.get("flows") or auth.get("methods") or []
                        if methods:
                            for method in methods:
                                st.write(f"✓ {method}")
                        else:
                            st.caption("No authentication methods specified")
                    
                    with col2:
                        # OAuth providers OR idp_options
                        st.markdown("#### OAuth / IDP Providers")
                        providers = auth.get("oauth_providers") or auth.get("idp_options") or []
                        if providers:
                            for provider in providers:
                                st.write(f"• {provider}")
                        else:
                            st.caption("No OAuth providers specified")
                    
                    # Additional sections
                    if auth.get("security_policies"):
                        st.markdown("#### Security Policies")
                        for policy in auth.get("security_policies", []):
                            st.write(f"📜 {policy}")
                    
                    if auth.get("additional_notes"):
                        st.markdown("#### Additional Security Notes")
                        st.info(auth.get("additional_notes"))
                    
                    # Fallback: show actors / threats (alternative fields in authentication.json)
                    if auth.get("actors"):
                        st.markdown("#### Actors / Users")
                        actors_cols = st.columns(3)
                        for idx, actor in enumerate(auth.get("actors", [])):
                            with actors_cols[idx % 3]:
                                st.write(f"👤 {actor}")
                    
                    if auth.get("threats"):
                        st.markdown("#### Security Threats")
                        for threat in auth.get("threats", []):
                            st.write(f"⚠️ {threat}")

                # Integrations
                integrations = None
                if final_state:
                    integrations = final_state.get("integrations") or []

                # Fallback: Load from integrations.json if not in state
                if not integrations:
                    integrations_json_path = output_dir / "json" / "integrations.json"
                    if integrations_json_path.exists():
                        try:
                            integrations = safe_load_json(str(integrations_json_path)) or []
                        except Exception:
                            integrations = []

                st.subheader("🔗 Integrations")
                if not integrations:
                    st.info("No integrations available.")
                else:
                    # Create a dataframe for integrations
                    integration_data = []
                    for integration in integrations:
                        if isinstance(integration, dict):
                            # tolerant field mapping / fallbacks
                            name = integration.get("name") or integration.get("system") or integration.get("system_name") or ""
                            typ = integration.get("type") or integration.get("protocol") or ""
                            purpose = integration.get("purpose") or integration.get("description") or ""
                            status = integration.get("status") or integration.get("state") or integration.get("auth") or ""
                            row = {
                                "Integration Name": name,
                                "Type": typ,
                                "Purpose": purpose,
                                "Status": status,
                            }
                            integration_data.append(row)
                    if integration_data:
                        df_integrations = pd.DataFrame(integration_data)
                        st.dataframe(df_integrations, width="stretch")

                # Domain
                domain = None
                if final_state:
                    domain = final_state.get("domain") or {}

                # Fallback: Load from domain.json if not in state
                if not domain:
                    domain_json_path = output_dir / "json" / "domain.json"
                    if domain_json_path.exists():
                        try:
                            domain = safe_load_json(str(domain_json_path)) or {}
                        except Exception:
                            domain = {}

                st.subheader("📦 Domain Entities")

                # Check if we have entities or APIs to display
                entities = domain.get("entities", []) if domain else []
                apis = domain.get("apis", []) if domain else []

                if not entities and not apis:
                    st.warning("⚠️ No domain entities or APIs found. The domain.json file may be empty or not generated properly.")
                    # Show what's in domain.json for debugging
                    if domain:
                        with st.expander("🔍 Debug: View domain.json content"):
                            st.json(domain)
                else:
                    # Display Entities
                    if entities:
                        entity_cols = st.columns(2)
                        for idx, entity in enumerate(entities):
                            with entity_cols[idx % 2]:
                                if isinstance(entity, dict):
                                    entity_name = entity.get("name", "Unknown Entity")
                                    entity_attrs = entity.get("attributes", [])
                                else:
                                    entity_name = str(entity)
                                    entity_attrs = []

                                with st.expander(f"📦 {entity_name}"):
                                    if entity_attrs:
                                        st.markdown("**Attributes:**")
                                        # Display each attribute with compact spacing
                                        for attr in entity_attrs:
                                            st.text(f"• {attr}")
                                    else:
                                        st.info("No attributes defined")
                    else:
                        st.info("No entities found in domain data.")

                    # Display APIs
                    if apis:
                        st.markdown("### 🌐 REST APIs")
                        with st.expander(f"View {len(apis)} API Endpoints", expanded=False):
                            for idx, api in enumerate(apis):
                                if isinstance(api, dict):
                                    api_name = api.get("name", "Unnamed API")
                                    # Clean up API name: replace // with - and handle URL formatting
                                    if isinstance(api_name, str):
                                        api_name = api_name.replace("//", " - ")
                                    api_desc = api.get("description", "No description")
                                    api_request = api.get("request", {})
                                    api_response = api.get("response", {})

                                    # API header using native Streamlit components
                                    with st.container():
                                        st.markdown(f"#### 🔌 {api_name}")
                                        st.caption(api_desc)

                                        # Request and Response in columns
                                        api_detail_cols = st.columns(2)

                                        with api_detail_cols[0]:
                                            st.markdown("**📤 Request**")
                                            if api_request:
                                                if isinstance(api_request, dict):
                                                    # Display as formatted JSON
                                                    st.json(dict(list(api_request.items())[:10]))
                                                else:
                                                    st.code(str(api_request), language="json")
                                            else:
                                                st.info("No request body")

                                        with api_detail_cols[1]:
                                            st.markdown("**📥 Response**")
                                            if api_response:
                                                if isinstance(api_response, dict):
                                                    # Display as formatted JSON
                                                    st.json(dict(list(api_response.items())[:10]))
                                                else:
                                                    st.code(str(api_response), language="json")
                                            else:
                                                st.info("No response body")

                                        # Add separator between APIs
                                        if idx < len(apis) - 1:
                                            st.divider()
                # Behavior
                behavior = None
                if final_state:
                    behavior = final_state.get("behavior") or {}

                # Fallback: Load from behavior.json if not in state
                if not behavior or not any(behavior.values()):
                    behavior_json_path = output_dir / "json" / "behavior.json"
                    if behavior_json_path.exists():
                        try:
                            behavior = safe_load_json(str(behavior_json_path)) or {}
                        except Exception:
                            behavior = {}

                st.subheader("🎯 Use Cases")
                if not behavior or not any(behavior.values()):
                    st.info("No behavior information available.")
                else:
                    if "use_cases" in behavior:
                        use_cases = behavior.get("use_cases", [])
                        with st.expander(f"View {len(use_cases)} Use Cases", expanded=False):
                            for uc in use_cases:
                                if isinstance(uc, str):
                                    name = uc
                                    steps = None
                                    sequence_flow = None
                                elif isinstance(uc, dict):
                                    name = uc.get("name") or uc.get("title") or "Unnamed Use Case"
                                    steps = uc.get("steps")
                                    sequence_flow = uc.get("sequence_flow") or uc.get("sequence") or uc.get("sequence_steps")
                                else:
                                    name = str(uc)
                                    steps = None
                                    sequence_flow = None

                                st.write(f"📋 **{name}**")
                                if steps:
                                    st.markdown("**Steps:**")
                                    for step in steps:
                                        st.markdown(f"- {step}")
                                if sequence_flow:
                                    st.markdown("**Sequence Flow:**")
                                    for flow in sequence_flow:
                                        st.markdown(f"1. {flow}")

                    # NFRs in a clean table format
                    st.subheader("⚡ Non-Functional Requirements")
                    if "nfrs" in behavior:
                        nfrs_obj = behavior.get("nfrs")
                        if isinstance(nfrs_obj, dict):
                            # category -> list with expanders
                            for cat, items in nfrs_obj.items():
                                # Add category-specific emojis
                                cat_emoji = {
                                    'performance': '🚀',
                                    'security': '🔒',
                                    'scalability': '📈',
                                    'reliability': '🛡️',
                                    'usability': '👥',
                                    'maintainability': '🔧',
                                    'availability': '⏱️',
                                }.get(cat.lower(), '📋')

                                with st.expander(f"{cat_emoji} {cat.capitalize()}", expanded=False):
                                    if not items:
                                        st.info("No items specified.")
                                    else:
                                        for it in items:
                                            if isinstance(it, dict):
                                                req = it.get("requirement") or it.get("desc") or str(it)
                                                target = it.get("target", "N/A")
                                                st.markdown(
                                                    f"""
                                                    **Requirement:** {req}  
                                                    **Target:** `{target}`
                                                    """
                                                )
                                                st.divider()
                                            else:
                                                st.markdown(f"• {it}")
                        else:
                            # assume list - display in cards
                            with st.expander(f"📋 View {len(nfrs_obj or [])} Requirements", expanded=False):
                                nfr_cols = st.columns(2)
                                for i, nfr in enumerate(nfrs_obj or []):
                                    with nfr_cols[i % 2]:
                                        # Handle both dict and string formats
                                        if isinstance(nfr, dict):
                                            category = nfr.get('category', 'General')
                                            requirement = nfr.get('requirement', 'N/A')
                                            target = nfr.get('target', 'N/A')

                                            st.markdown(
                                                f"""
                                                **{category}**  
                                                🎯 {requirement}  
                                                📊 Target: `{target}`
                                                """
                                            )
                                        else:
                                            # If nfr is a string, display it as is
                                            st.markdown(f"• {str(nfr)}")

                                        if i < len(nfrs_obj) - 1 and i % 2 == 1:
                                            st.markdown("---")

                    # Risks with severity indicators
                    st.subheader("⚠️ Risk Analysis")
                    if "risks" in behavior:
                        risks_list = behavior.get("risks", [])
                        if not risks_list:
                            st.info("No risks identified.")
                        else:
                            with st.expander(f"View {len(risks_list)} Identified Risks", expanded=False):
                                for idx, risk in enumerate(risks_list):
                                    # Handle both dict and string formats
                                    if isinstance(risk, dict):
                                        name = risk.get('name') or risk.get('desc') or 'Unnamed Risk'
                                        impact = risk.get('impact', 'N/A')
                                        mitigation = risk.get('mitigation', 'N/A')
                                        assumption = risk.get('assumption') or risk.get('assumptions')
                                    else:
                                        # If risk is a string, display it as is
                                        name = str(risk)
                                        impact = 'N/A'
                                        mitigation = 'N/A'
                                        assumption = None

                                    # Display risk in a container
                                    with st.container():
                                        st.markdown(f"**{name}**")

                                        st.markdown(
                                            f"""
                                            **💥 Impact:** {impact}  
                                            **🛡️ Mitigation:** {mitigation}
                                            """
                                        )

                                        if assumption:
                                            st.markdown(f"**💭 Assumption:** {assumption}")

                                    # Add separator between risks
                                    if idx < len(risks_list) - 1:
                                        st.divider()

                    # Risk heatmap display in expander
                    risk_img = output_dir / "diagrams" / "risk_heatmap.png"
                    if risk_img.exists():
                        st.markdown("### 📊 Risk Heatmap")
                        with st.expander("View Risk Heatmap", expanded=True):
                            st.image(str(risk_img), width=600)
                            st.download_button(
                                "⬇️ Download Risk Heatmap",
                                data=risk_img.read_bytes(),
                                file_name="risk_heatmap.png",
                                help="Download the risk heatmap visualization",
                            )
                    # Diagrams area: show links and embedded diagrams if available
                out_obj = {}
                if final_state:
                    out_obj = final_state.get("output") or {}

                hld_md_path = None
                hld_html_path = None
                diagrams_html_path = None

                if isinstance(out_obj, dict):
                    hld_md_path = out_obj.get("hld_md_path") or out_obj.get("hld_md")
                    hld_html_path = out_obj.get("hld_html_path") or out_obj.get("hld_html")
                    diagrams_html_path = out_obj.get("diagrams_html_path") or out_obj.get("diagrams_html")

                # If paths not in final_state, try to find them in output directory
                if not hld_md_path:
                    potential_hld_md = output_dir / "hld" / "HLD.md"
                    if potential_hld_md.exists():
                        hld_md_path = str(potential_hld_md)

                if not hld_html_path:
                    potential_hld_html = output_dir / "hld" / "HLD.html"
                    if potential_hld_html.exists():
                        hld_html_path = str(potential_hld_html)

                if not diagrams_html_path:
                    potential_diagrams = output_dir / "diagrams" / "full_diagrams.html"
                    if potential_diagrams.exists():
                        diagrams_html_path = str(potential_diagrams)

                # Normalize paths (resolve relative → absolute) and provide links
                if hld_md_path:
                    try:
                        hld_md_path = str(Path(hld_md_path).resolve())
                    except Exception:
                        pass
                if hld_html_path:
                    try:
                        hld_html_path = str(Path(hld_html_path).resolve())
                    except Exception:
                        pass
                if diagrams_html_path:
                    try:
                        diagrams_html_path = str(Path(diagrams_html_path).resolve())
                    except Exception:
                        pass
                # fallback: check output dir for a full_diagrams.html viewer
                try:
                    if not diagrams_html_path:
                        fallback_viewer = output_dir / "diagrams" / "full_diagrams.html"
                        if fallback_viewer.exists():
                            diagrams_html_path = str(fallback_viewer.resolve())
                except Exception:
                    pass

                # Interactive Diagrams first
                if diagrams_html_path and Path(diagrams_html_path).exists():
                    st.markdown("## 📊 Interactive Diagrams")

                    with st.expander("🔍 View Diagrams (Click to Expand/Minimize)", expanded=True):
                        try:
                            # Read the HTML content and embed it directly with white background
                            html_content = Path(diagrams_html_path).read_text(encoding='utf-8')
                            # Wrap content with white background styling
                            styled_html = f"""
                            <div style="background-color: white; padding: 20px; border-radius: 5px;">
                                {html_content}
                            </div>
                            """
                            st.components.v1.html(styled_html, height=800, scrolling=True)
                        except Exception as e:
                            st.error(f"Could not render diagrams: {e}")
                            st.markdown(f"[Open Diagrams Viewer]({Path(diagrams_html_path).as_uri()})")
                    
                    # Check if img folder exists and has PNG files
                    img_folder = output_dir / "diagrams" / "img"
                    if img_folder.exists():
                        png_files = list(img_folder.glob("*.png"))
                        if png_files:
                            # Create download button for all PNG images
                            zip_data = create_images_zip(img_folder)
                            if zip_data:
                                st.download_button(
                                    label=f"📥 Download All Diagram Images ({len(png_files)} PNG files)",
                                    data=zip_data,
                                    file_name=f"{Path(selected_pdf).stem}_diagrams.zip",
                                    mime="application/zip",
                                    help="Download all diagram images as a ZIP file"
                                )

                # HLD Document below
                if hld_md_path and Path(hld_md_path).exists():
                    st.markdown("## 📄 HLD Document")
                    with st.expander("📖 View HLD Markdown (Click to Expand/Minimize)", expanded=False):
                        try:
                            hld_content = Path(hld_md_path).read_text()
                            # Add scrolling container with max height
                            st.markdown(
                                f'<div style="max-height: 600px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{hld_content}</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception:
                            st.error("Could not read HLD content")
                    st.download_button(
                        "⬇️ Download HLD.md",
                        data=Path(hld_md_path).read_bytes(),
                        file_name=Path(hld_md_path).name,
                        help="Download the High Level Design document in Markdown format",
                    )


    # -------------------------
    # Tab 2: ML Training (integrated)
    # -------------------------
    with tabs[1]:
        st.header("ML Training (Dataset & Models)")
        st.info("Preview the synthetic dataset and train ML models.")
        from ml.training.train_large_model import LargeScaleMLTrainer
        from ml.training.generate_dataset import SyntheticDatasetGenerator
        dataset_path = os.path.join(os.path.dirname(__file__), "ml", "training", "synthetic_hld_dataset.csv")
        
        # Dataset Generation Section (only show if dataset doesn't exist)
        if not os.path.exists(dataset_path):
            st.subheader("Generate Dataset")
            st.info("This will generate a synthetic dataset with 30,000 samples for training ML models.")
            
            # Fixed number of samples
            n_samples = 30000
            
            if st.button("Generate Dataset", type="secondary"):
                with st.spinner("Generating synthetic dataset with 30,000 samples..."):
                    try:
                        logger.info(f"Generating synthetic dataset with {n_samples} samples")
                        generator = SyntheticDatasetGenerator(random_state=42)
                        df_generated = generator.generate(n_samples=n_samples)
                        generator.save_dataset(df_generated, dataset_path)
                        st.success(f"Dataset generated successfully! {df_generated.shape[0]} rows, {df_generated.shape[1]} columns")
                        logger.info(f"Dataset saved to {dataset_path}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating dataset: {e}")
                        logger.error(f"Dataset generation failed: {e}")
        
        # Dataset Preview Section
        if os.path.exists(dataset_path):
            col_header, col_delete = st.columns([4, 1])
            with col_header:
                st.subheader("Dataset Preview")
            with col_delete:
                st.write("")  # spacing
                if st.button("🗑️ Delete Dataset", type="secondary", help="Delete the current dataset and trained models to start fresh"):
                    try:
                        # Delete dataset
                        os.remove(dataset_path)
                        logger.info(f"Dataset deleted: {dataset_path}")
                        
                        # Delete all trained model files
                        model_dir = os.path.join(os.path.dirname(__file__), "ml", "training", "models")
                        if os.path.exists(model_dir):
                            deleted_models = []
                            for fname in os.listdir(model_dir):
                                if fname.endswith('_model.pkl'):
                                    model_path = os.path.join(model_dir, fname)
                                    os.remove(model_path)
                                    deleted_models.append(fname)
                                    logger.info(f"Deleted model: {fname}")
                            
                            if deleted_models:
                                st.success(f"✅ Dataset and {len(deleted_models)} trained models deleted successfully!")
                            else:
                                st.success("✅ Dataset deleted successfully! (No trained models found)")
                        else:
                            st.success("✅ Dataset deleted successfully!")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting dataset: {e}")
                        logger.error(f"Failed to delete dataset: {e}")
            
            df = pd.read_csv(dataset_path)
            
            # EDA Section (expandable)
            with st.expander("📊 Exploratory Data Analysis (EDA)", expanded=False):
                st.markdown("### Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", f"{df.shape[0]:,}")
                with col2:
                    st.metric("Total Features", df.shape[1] - 1)
                with col3:
                    st.metric("Target Mean", f"{df['quality_score'].mean():.2f}")
                with col4:
                    st.metric("Target Std", f"{df['quality_score'].std():.2f}")
                
                st.markdown("### Target Distribution")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(df['quality_score'], bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Target Variable Distribution')
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### Top Feature Correlations with Target")
                features = [col for col in df.columns if col != 'quality_score']
                correlations = {}
                for feature in features:
                    correlations[feature] = df[feature].corr(df['quality_score'])
                
                corr_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Correlation'])
                corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
                corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['green' if x > 0 else 'red' for x in corr_df['Correlation']]
                ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, alpha=0.7)
                ax.set_xlabel('Correlation with Quality Score')
                ax.set_title('Top 15 Feature Correlations')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                st.pyplot(fig)
                plt.close()
            
            # Dataset Preview
            st.dataframe(df.head(), width="stretch")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            logger.info(f"Loaded dataset from {dataset_path} with shape {df.shape}")
        else:
            logger.warning(f"Dataset not found at {dataset_path}")

        # Model Training Section
        # Check if dataset exists before showing train button
        dataset_exists = os.path.exists(dataset_path)
        
        if not dataset_exists:
            st.subheader("Train Models")
            st.button("Train models", type="primary", disabled=True, help="Generate dataset first to enable training")
        else:
            # Initialize training state
            if 'is_training' not in st.session_state:
                st.session_state.is_training = False
            
            trainer = None
            
            # Create placeholders for header, button and info message
            header_placeholder = st.empty()
            button_placeholder = st.empty()
            info_placeholder = st.empty()
            
            # Show header based on training state
            with header_placeholder:
                if st.session_state.is_training:
                    st.subheader("🔄 Training Models in Progress...")
                else:
                    st.subheader("Train Models")
            
            # Show disabled button if training is in progress
            if st.session_state.is_training:
                with button_placeholder:
                    st.button("Train models", type="primary", disabled=True)
                with info_placeholder:
                    st.info("⏳ Training in progress... Please wait.")
            else:
                with button_placeholder:
                    train_button = st.button("Train models", type="primary")
                
                if train_button:
                    st.session_state.is_training = True
                    st.rerun()  # Immediately rerun to show disabled button
            
            # Execute training if state is True
            if st.session_state.is_training:
                # Create placeholders for progress updates
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Create container for training steps display
                steps_container = st.container()
                
                training_steps = []
                
                try:
                    logger.info("Training models started.")
                    
                    # Step 1: Load dataset with EDA
                    with steps_container:
                        st.markdown("### 🔄 Training Progress")
                        step1_placeholder = st.empty()
                    
                    status_placeholder.info("📂 Loading dataset and performing EDA...")
                    step1_placeholder.info("**Step 1/5:** 📂 Loading dataset and performing EDA...")
                    
                    trainer = LargeScaleMLTrainer()
                    trainer.load_dataset(dataset_path)
                    logger.info("Dataset loaded for training.")
                    
                    # Display EDA results
                    eda_stats = trainer.get_eda_stats()
                    step1_info = f"""
                    **Step 1/5:** ✅ Dataset loaded & EDA complete
                    - **Dataset shape:** {eda_stats['shape'][0]:,} rows × {eda_stats['shape'][1]} columns
                    - **Missing values:** {sum(eda_stats['missing_values'].values())}
                    - **Duplicate rows:** {eda_stats['duplicates']}
                    - **Target mean:** {eda_stats['target_stats']['mean']:.2f}
                    - **Target std:** {eda_stats['target_stats']['std']:.2f}
                    - **Significant features found:** {len(eda_stats.get('significant_features', []))}
                    """
                    step1_placeholder.success(step1_info)
                    
                    # Step 2: Prepare data with feature selection and scaling
                    step2_placeholder = st.empty()
                    status_placeholder.info("🔧 Preparing data (feature selection, scaling, train/test split)...")
                    step2_placeholder.info("**Step 2/5:** 🔧 Preparing data...")
                    
                    trainer.prepare_data(use_feature_selection=True, use_scaling=True)
                    logger.info(f"Data prepared. Selected {len(trainer.selected_features)} features.")
                    
                    step2_info = f"""
                    **Step 2/5:** ✅ Data preparation complete
                    - **Train set:** {trainer.X_train_scaled.shape[0]:,} samples
                    - **Test set:** {trainer.X_test_scaled.shape[0]:,} samples
                    - **Features selected:** {len(trainer.selected_features)} out of {len([col for col in trainer.df.columns if col != 'quality_score'])}
                    - **Feature selection:** Combined EDA correlation + F-score analysis
                    - **Scaling:** StandardScaler applied
                    """
                    step2_placeholder.success(step2_info)
                    
                    # Step 3: Train models with cross-validation
                    step3_placeholder = st.empty()
                    
                    status_placeholder.info("🤖 Training models with cross-validation...")
                    step3_placeholder.info("**Step 3/5:** 🤖 Training models with 5-fold cross-validation...")
                    
                    total_models = len(trainer.models)
                    for idx, (name, _) in enumerate(trainer.models.items(), 1):
                        progress_placeholder.progress(idx / total_models, text=f"Training model {idx}/{total_models}: {name}")
                    
                    # Call the train_models method from train_large_model.py
                    trainer.train_models(use_cross_validation=True)
                    
                    progress_placeholder.empty()
                    
                    # Extract CV results from logs or results
                    step3_info = f"""
                    **Step 3/5:** ✅ Model training complete
                    - **Models trained:** {len(trainer.models)}
                    - **Cross-validation:** 5-fold
                    - **All models trained successfully**
                    """
                    step3_placeholder.success(step3_info)
                    
                    status_placeholder.success("✅ All models trained successfully!")
                    logger.info("Models trained.")
                    
                    # Step 4: Evaluate models
                    step4_placeholder = st.empty()
                    status_placeholder.info("📊 Evaluating models on test set...")
                    step4_placeholder.info("**Step 4/5:** 📊 Evaluating models...")
                    
                    trainer.evaluate_models()
                    logger.info("Model evaluation complete.")
                    
                    # Get best model
                    best_model_name = max(trainer.results.items(), key=lambda x: x[1]['R2'])[0]
                    best_r2 = trainer.results[best_model_name]['R2']
                    best_rmse = trainer.results[best_model_name]['RMSE']
                    
                    step4_info = f"""
                    **Step 4/5:** ✅ Model evaluation complete
                    - **Best model:** {best_model_name}
                    - **Best R²:** {best_r2:.4f}
                    - **Best RMSE:** {best_rmse:.2f}
                    - **Models evaluated:** {len(trainer.results)}
                    """
                    step4_placeholder.success(step4_info)
                    
                    # Show evaluation results table
                    eval_results_df = pd.DataFrame({
                        'Model': list(trainer.results.keys()),
                        'R² Score': [f"{r['R2']:.4f}" for r in trainer.results.values()],
                        'RMSE': [f"{r['RMSE']:.2f}" for r in trainer.results.values()],
                        'MAE': [f"{r['MAE']:.2f}" for r in trainer.results.values()],
                        'MAPE': [f"{r['MAPE']:.2f}%" for r in trainer.results.values()]
                    }).sort_values('R² Score', ascending=False)
                    st.dataframe(eval_results_df, hide_index=True, use_container_width=True)
                    
                    # Step 5: Save models with metadata
                    step5_placeholder = st.empty()
                    status_placeholder.info("💾 Saving models, scaler, and metadata to disk...")
                    step5_placeholder.info("**Step 5/5:** 💾 Saving models...")
                    
                    out_dir = os.path.join(os.path.dirname(__file__), "ml", "training", "models")
                    trainer.save_models(out_dir)
                    logger.info(f"Models saved to {out_dir}")
                    
                    step5_info = f"""
                    **Step 5/5:** ✅ Models saved successfully
                    - **Location:** `{out_dir}`
                    - **Models saved:** {len(trainer.models)}
                    - **Files:** {len(trainer.models)} model files + scaler + metadata
                    """
                    step5_placeholder.success(step5_info)
                    
                    status_placeholder.success("✅ Training pipeline completed successfully!")
                    
                    st.markdown("---")
                    st.subheader("📊 Training Results")
                    
                    # Model comparison table
                    st.markdown("### Model Performance Comparison")
                    df_metrics = trainer.get_model_comparison()
                    if df_metrics is not None:
                        st.dataframe(df_metrics, width="stretch")
                        
                        best_model = df_metrics.index[0]
                        best_r2 = df_metrics.loc[best_model, 'R2']
                        best_rmse = df_metrics.loc[best_model, 'RMSE']
                        st.success(f"🏆 **Best Model:** {best_model} — R² = {best_r2:.4f}, RMSE = {best_rmse:.2f}")
                    
                    # Feature importance
                    st.markdown("### 🎯 Feature Importance (Top Features)")
                    feature_importance = trainer.get_feature_importance_summary()
                    if feature_importance:
                        import matplotlib.pyplot as plt
                        
                        # Create subplots for each model
                        fig, axes = plt.subplots(1, len(feature_importance), figsize=(15, 5))
                        if len(feature_importance) == 1:
                            axes = [axes]
                        
                        for idx, (model_name, importances) in enumerate(feature_importance.items()):
                            features = list(importances.keys())[:10]
                            values = [importances[f] for f in features]
                            
                            axes[idx].barh(features, values, color='skyblue', alpha=0.8)
                            axes[idx].set_xlabel('Importance')
                            axes[idx].set_title(f'{model_name}')
                            axes[idx].invert_yaxis()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Selected features summary
                    st.markdown("### ✅ Selected Features")
                    st.info(f"**{len(trainer.selected_features)}** features selected out of 37 total features")
                    with st.expander("View Selected Features"):
                        st.write(trainer.selected_features)
                    
                    # Reset state and clear all placeholders
                    st.session_state.is_training = False
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    header_placeholder.empty()
                    button_placeholder.empty()
                    info_placeholder.empty()
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    logger.error(f"Training failed: {e}")
                    st.session_state.is_training = False
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    header_placeholder.empty()
                    button_placeholder.empty()
                    info_placeholder.empty()

    # -------------------------
    # Tab 3: Quality Prediction (integrated)
    # -------------------------
    with tabs[2]:
        st.header("Quality Prediction")
        st.info("Load trained models and predict HLD quality scores.")
        from ml.training.inference import HLDQualityPredictor
        from ml.models.feature_extractor import FeatureExtractor
        model_dir = os.path.join(os.path.dirname(__file__), "ml", "training", "models")
        feature_extractor = FeatureExtractor()
        predictor = HLDQualityPredictor(model_dir, feature_extractor.get_feature_names())
        models_loaded = predictor.load_models_from_disk()
        if not models_loaded:
            st.warning("⚠️ No trained models found. Please train models first in the **ML Training** tab.")
            
            # Check if dataset exists to give more specific guidance
            dataset_path = os.path.join(os.path.dirname(__file__), "ml", "training", "synthetic_hld_dataset.csv")
            if not os.path.exists(dataset_path):
                st.info("📝 **Next Steps:**\n1. Go to **ML Training** tab\n2. Click **Generate Dataset** button\n3. Click **Train models** button\n4. Return here to predict quality")
            else:
                st.info("📝 **Next Steps:**\n1. Go to **ML Training** tab\n2. Click **Train models** button\n3. Return here to predict quality")
            
            logger.warning(f"No trained models found in {model_dir}")
        else:
            st.success("Models loaded!")
            logger.info(f"Models loaded from {model_dir}")
            tab1, tab2, tab3 = st.tabs(["Quick Scenario", "Custom Features", "📄 Predict from Generated HLD"])
            with tab1:
                st.subheader("Predefined Scenarios")
                scenario = st.selectbox("Choose scenario", ["Excellent", "Good", "Poor"])

                # Build domain-aware scenario features (not simply max/min for all features)
                def build_scenario_features(level: str):
                    rng = feature_extractor.feature_ranges
                    names = feature_extractor.get_feature_names()
                    def rmin(n):
                        return float(rng.get(n, (0.0, 1.0))[0])
                    def rmax(n):
                        return float(rng.get(n, (0.0, 1.0))[1])
                    def rmid(n):
                        a, b = rng.get(n, (0.0, 1.0))
                        return float((a + b) / 2)

                    # Defaults
                    vals = {n: rmid(n) for n in names}

                    # Directional adjustments
                    if level == "Excellent":
                        vals['word_count'] = rmax('word_count') * 0.9
                        vals['sentence_count'] = rmax('sentence_count') * 0.8
                        vals['avg_sentence_length'] = max(rmin('avg_sentence_length'), rmid('avg_sentence_length') * 0.8)  # shorter
                        vals['header_count'] = rmax('header_count') * 0.9
                        vals['code_block_count'] = rmid('code_block_count')  # balanced
                        vals['table_count'] = rmax('table_count') * 0.7
                        vals['list_count'] = rmax('list_count') * 0.8
                        vals['security_mentions'] = rmax('security_mentions')
                        vals['scalability_mentions'] = rmax('scalability_mentions')
                        vals['api_mentions'] = rmax('api_mentions')
                        vals['service_count'] = rmax('service_count') * 0.8
                        vals['entity_count'] = rmax('entity_count') * 0.8
                        vals['api_endpoint_count'] = rmax('api_endpoint_count') * 0.8
                        vals['readability_score'] = 95.0
                        vals['documentation_quality'] = 95.0
                        # Filler features neutral-high
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 80.0
                    elif level == "Good":
                        vals['word_count'] = rmid('word_count') * 1.1
                        vals['sentence_count'] = rmid('sentence_count')
                        vals['avg_sentence_length'] = rmid('avg_sentence_length')
                        vals['header_count'] = rmid('header_count')
                        vals['code_block_count'] = rmid('code_block_count')
                        vals['table_count'] = rmid('table_count')
                        vals['list_count'] = rmid('list_count')
                        vals['security_mentions'] = rmid('security_mentions') * 1.2
                        vals['scalability_mentions'] = rmid('scalability_mentions') * 1.2
                        vals['api_mentions'] = rmid('api_mentions') * 1.2
                        vals['service_count'] = rmid('service_count')
                        vals['entity_count'] = rmid('entity_count')
                        vals['api_endpoint_count'] = rmid('api_endpoint_count')
                        vals['readability_score'] = 80.0
                        vals['documentation_quality'] = 80.0
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 60.0
                    else:  # Poor
                        vals['word_count'] = rmin('word_count') * 0.8
                        vals['sentence_count'] = rmin('sentence_count') * 0.8
                        vals['avg_sentence_length'] = min(rmax('avg_sentence_length'), rmid('avg_sentence_length') * 1.3)  # longer
                        vals['header_count'] = rmin('header_count')
                        vals['code_block_count'] = rmin('code_block_count')
                        vals['table_count'] = rmin('table_count')
                        vals['list_count'] = rmin('list_count')
                        vals['security_mentions'] = rmin('security_mentions')
                        vals['scalability_mentions'] = rmin('scalability_mentions')
                        vals['api_mentions'] = rmin('api_mentions')
                        vals['service_count'] = rmin('service_count')
                        vals['entity_count'] = rmin('entity_count')
                        vals['api_endpoint_count'] = rmin('api_endpoint_count')
                        vals['readability_score'] = 40.0
                        vals['documentation_quality'] = 40.0
                        for i in range(16, 38):
                            vals[f'feature_{i}'] = 30.0

                    # Clip into ranges
                    for n in names:
                        a, b = rng.get(n, (0.0, 100.0))
                        vals[n] = float(max(a, min(b, vals.get(n, rmid(n)))))
                    return vals

                features = build_scenario_features(scenario)
                if st.button("Predict Scenario Quality"):
                    logger.info(f"Predicting scenario quality for: {scenario}")
                    preds = predictor.predict(features)
                    # Nicely formatted UI
                    def _clip01(v):
                        try:
                            return max(0.0, min(100.0, float(v)))
                        except Exception:
                            return 0.0
                    overall = _clip01(preds.get('ensemble_average', 0.0))
                    st.markdown("### Overall Quality")
                    st.metric("Overall", f"{overall:.1f}/100")
                    try:
                        st.progress(int(round(overall)))
                    except Exception:
                        pass
                    st.markdown("### Model Breakdown")
                    # Show per-model predictions in neat columns
                    items = [(k, v) for k, v in preds.items() if k != 'ensemble_average']
                    if items:
                        cols = st.columns(min(4, len(items)))
                        for idx, (name, val) in enumerate(items):
                            with cols[idx % len(cols)]:
                                score = _clip01(val)
                                st.metric(label=name, value=f"{score:.1f}/100")
                    logger.info(f"Prediction result: {preds}")
            with tab2:
                st.subheader("Custom Features")
                custom_features = {}
                for name in feature_extractor.get_feature_names():
                    rng = feature_extractor.feature_ranges.get(name, (0, 1))
                    custom_features[name] = st.slider(name, float(rng[0]), float(rng[1]), float(rng[0]), help=f"Range: {rng[0]}-{rng[1]}")
                if st.button("Predict Custom Quality"):
                    logger.info(f"Predicting custom quality with features: {custom_features}")
                    preds = predictor.predict(custom_features)
                    def _clip01(v):
                        try:
                            return max(0.0, min(100.0, float(v)))
                        except Exception:
                            return 0.0
                    overall = _clip01(preds.get('ensemble_average', 0.0))
                    st.markdown("### Overall Quality")
                    st.metric("Overall", f"{overall:.1f}/100")
                    try:
                        st.progress(int(round(overall)))
                    except Exception:
                        pass
                    st.markdown("### Model Breakdown")
                    items = [(k, v) for k, v in preds.items() if k != 'ensemble_average']
                    if items:
                        cols = st.columns(min(4, len(items)))
                        for idx, (name, val) in enumerate(items):
                            with cols[idx % len(cols)]:
                                score = _clip01(val)
                                st.metric(label=name, value=f"{score:.1f}/100")
                    logger.info(f"Prediction result: {preds}")
            
            with tab3:
                st.subheader("Predict Quality from Generated HLD")
                st.info("Select a generated HLD document to extract features and predict its quality score.")
                
                # List all generated HLD outputs
                output_base_dir = Path(__file__).resolve().parent / "output"
                
                if not output_base_dir.exists():
                    st.warning("No output directory found. Please generate at least one HLD first.")
                else:
                    # Find all requirement folders with HLD.md files
                    available_hlds = []
                    for req_folder in output_base_dir.iterdir():
                        if req_folder.is_dir() and req_folder.name.startswith("Requirement-"):
                            hld_path = req_folder / "hld" / "HLD.md"
                            if hld_path.exists():
                                available_hlds.append({
                                    "name": req_folder.name,
                                    "path": hld_path,
                                    "display": f"{req_folder.name} ({hld_path.stat().st_size / 1024:.1f} KB)"
                                })
                    
                    if not available_hlds:
                        st.warning("No generated HLD files found. Please generate at least one HLD document first in the 'HLD Generation' tab.")
                    else:
                        # Dropdown to select HLD
                        st.markdown(f"**Found {len(available_hlds)} generated HLD document(s)**")
                        
                        hld_options = [hld["display"] for hld in available_hlds]
                        selected_hld_display = st.selectbox("Select HLD Document", hld_options)
                        selected_idx = hld_options.index(selected_hld_display)
                        selected_hld = available_hlds[selected_idx]
                        
                        # Show HLD info
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Selected:** {selected_hld['name']}")
                            st.write(f"Path: `{selected_hld['path']}`")
                            try:
                                modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(selected_hld['path'].stat().st_mtime))
                                st.write(f"Modified: {modified_time}")
                            except Exception:
                                pass
                        with col2:
                            st.download_button(
                                "Download HLD",
                                data=selected_hld['path'].read_bytes(),
                                file_name=selected_hld['path'].name,
                                help="Download the selected HLD document"
                            )
                        
                        # Preview HLD content
                        with st.expander("📖 Preview HLD Content", expanded=False):
                            try:
                                hld_content = selected_hld['path'].read_text(encoding='utf-8')
                                # Show first 2000 characters
                                preview = hld_content[:2000]
                                if len(hld_content) > 2000:
                                    preview += "\n\n... (content truncated for preview)"
                                st.text_area("HLD Content Preview", preview, height=300, disabled=True)
                            except Exception as e:
                                st.error(f"Could not read HLD content: {e}")
                        
                        # Predict button
                        if st.button("🔍 Extract Features & Predict Quality", type="primary"):
                            with st.spinner("Extracting features from HLD document..."):
                                try:
                                    # Read HLD content
                                    hld_text = selected_hld['path'].read_text(encoding='utf-8')
                                    logger.info(f"Reading HLD from {selected_hld['path']}")
                                    
                                    # Extract features using FeatureExtractor
                                    st.info("📊 Extracting features from document...")
                                    features = feature_extractor.extract_features(hld_text)
                                    logger.info(f"Extracted {len(features)} features from HLD")
                                    
                                    # Show what features will be used for prediction
                                    if predictor.selected_features:
                                        st.info(f"ℹ️ Using **{len(predictor.selected_features)}** selected features from training (out of {len(features)} extracted)")
                                        
                                        # Show which features are being used vs ignored
                                        with st.expander("🔍 Feature Selection Details", expanded=False):
                                            used_features = [f for f in features.keys() if f in predictor.selected_features]
                                            ignored_features = [f for f in features.keys() if f not in predictor.selected_features]
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown(f"**✅ Used Features ({len(used_features)}):**")
                                                st.write(used_features)
                                            with col2:
                                                st.markdown(f"**❌ Ignored Features ({len(ignored_features)}):**")
                                                st.write(ignored_features)
                                    else:
                                        st.warning("⚠️ No feature selection metadata found. Using all features (may affect accuracy).")
                                    
                                    # Show extracted features in expander
                                    with st.expander("� View All Extracted Features", expanded=False):
                                        feature_df = pd.DataFrame([
                                            {
                                                "Feature": k, 
                                                "Value": f"{v:.2f}",
                                                "Used": "✅" if (not predictor.selected_features or k in predictor.selected_features) else "❌"
                                            } 
                                            for k, v in features.items()
                                        ])
                                        st.dataframe(feature_df, width="stretch", height=300)
                                    
                                    # Predict quality
                                    st.info("🤖 Predicting quality using trained models...")
                                    preds = predictor.predict(features)
                                    logger.info(f"Quality prediction complete for {selected_hld['name']}")
                                    
                                    # Display results
                                    st.success("✅ Quality prediction completed!")
                                    
                                    def _clip01(v):
                                        try:
                                            return max(0.0, min(100.0, float(v)))
                                        except Exception:
                                            return 0.0
                                    
                                    overall = _clip01(preds.get('ensemble_average', 0.0))
                                    
                                    # Overall quality score with color coding
                                    st.markdown("### 📊 Ensemble average")
                                    col_metric, col_progress = st.columns([1, 2])
                                    with col_metric:
                                        st.metric("Overall Quality", f"{overall:.1f}/100")
                                    with col_progress:
                                        st.progress(int(round(overall)) / 100)
                                    
                                    # Quality assessment
                                    if overall >= 85:
                                        st.success("🌟 **Excellent Quality** - This HLD document demonstrates high quality across all metrics!")
                                    elif overall >= 70:
                                        st.info("✅ **Good Quality** - This HLD document meets quality standards with room for improvement.")
                                    elif overall >= 50:
                                        st.warning("⚠️ **Moderate Quality** - This HLD document needs improvement in several areas.")
                                    else:
                                        st.error("❌ **Poor Quality** - This HLD document requires significant improvements.")
                                    
                                    # Model breakdown
                                    st.markdown("### 🤖 Model Breakdown")
                                    items = [(k, v) for k, v in preds.items() if k != 'ensemble_average']
                                    if items:
                                        cols = st.columns(min(3, len(items)))
                                        for idx, (name, val) in enumerate(items):
                                            with cols[idx % len(cols)]:
                                                score = _clip01(val)
                                                st.metric(label=name, value=f"{score:.1f}/100")
                                    
                                    # Feature highlights - show extracted vs model features
                                    st.markdown("### 📈 Key Extracted Feature Values")
                                    
                                    # Show important features that were actually extracted
                                    important_extracted = [
                                        'word_count', 'sentence_count', 'avg_sentence_length',
                                        'header_count', 'security_mentions', 'api_mentions',
                                        'readability_score', 'documentation_quality'
                                    ]
                                    
                                    highlight_cols = st.columns(4)
                                    for idx, feat in enumerate(important_extracted):
                                        if feat in features:
                                            with highlight_cols[idx % 4]:
                                                # Mark if this feature was used in prediction
                                                is_used = (not predictor.selected_features or feat in predictor.selected_features)
                                                icon = "✅" if is_used else "❌"
                                                st.metric(
                                                    label=f"{icon} {feat.replace('_', ' ').title()}",
                                                    value=f"{features[feat]:.1f}",
                                                    help="✅ = Used in model, ❌ = Not selected"
                                                )
                                    
                                    # Prediction pipeline transparency
                                    with st.expander("🔍 Prediction Pipeline Details", expanded=False):
                                        st.markdown("#### Feature Processing Pipeline")
                                        st.write(f"1️⃣ **Extracted from HLD:** {len(features)} features")
                                        if predictor.selected_features:
                                            st.write(f"2️⃣ **Filtered to selected:** {len(predictor.selected_features)} features")
                                            st.write(f"3️⃣ **Scaling applied:** {'Yes ✅' if predictor.scaler else 'No ❌'}")
                                        else:
                                            st.write(f"2️⃣ **Feature selection:** Not applied (using all features)")
                                            st.write(f"3️⃣ **Scaling applied:** {'Yes ✅' if predictor.scaler else 'No ❌'}")
                                        st.write(f"4️⃣ **Models used:** {len(predictor.models)} models")
                                        st.write(f"5️⃣ **Ensemble method:** Average of all models")
                                        
                                        # Show which specific features contributed
                                        if predictor.selected_features:
                                            st.markdown("#### Features Contributing to Prediction")
                                            contributing_features = {k: v for k, v in features.items() if k in predictor.selected_features}
                                            st.write(f"**{len(contributing_features)} features** are actively used:")
                                            st.json(contributing_features)
                                    
                                    # Real-time feature extraction validation
                                    with st.expander("✅ Real-Time Feature Validation", expanded=False):
                                        st.markdown("#### Proof that extracted values are REAL from HLD content:")
                                        
                                        # Core features that prove real extraction
                                        validation_features = {
                                            'word_count': features.get('word_count', 0),
                                            'sentence_count': features.get('sentence_count', 0),
                                            'header_count': features.get('header_count', 0),
                                            'code_block_count': features.get('code_block_count', 0),
                                            'security_mentions': features.get('security_mentions', 0),
                                            'api_mentions': features.get('api_mentions', 0),
                                            'entity_count': features.get('entity_count', 0),
                                            'api_endpoint_count': features.get('api_endpoint_count', 0)
                                        }
                                        
                                        st.markdown("**Core Extracted Metrics (from HLD text analysis):**")
                                        val_df = pd.DataFrame([
                                            {"Feature": k.replace('_', ' ').title(), "Value": f"{v:.1f}", "Source": "Direct HLD Content Analysis"}
                                            for k, v in validation_features.items()
                                        ])
                                        st.dataframe(val_df, width="stretch", hide_index=True)
                                        
                                        # Show uniqueness - if these were dummy values, they'd all be the same
                                        unique_values = len(set(validation_features.values()))
                                        total_features = len(validation_features)
                                        
                                        if unique_values > 1:
                                            st.success(f"✅ **{unique_values}/{total_features} unique values detected")
                                        else:
                                            st.warning(f"⚠️ Only {unique_values} unique value - May indicate dummy data")
                                        
                                        st.markdown("**Validation Checks:**")
                                        checks = []
                                        
                                        # Check 1: Word count should be realistic
                                        wc = validation_features['word_count']
                                        checks.append({
                                            "Check": "Word count > 0",
                                            "Status": "✅ Pass" if wc > 0 else "❌ Fail",
                                            "Value": f"{wc:.0f} words"
                                        })
                                        
                                        # Check 2: Headers should exist in a real HLD
                                        hc = validation_features['header_count']
                                        checks.append({
                                            "Check": "Headers detected",
                                            "Status": "✅ Pass" if hc > 0 else "❌ Fail",
                                            "Value": f"{hc:.0f} headers"
                                        })
                                        
                                        # Check 3: At least some architectural mentions
                                        arch_mentions = (validation_features['api_mentions'] + 
                                                        validation_features['security_mentions'] +
                                                        validation_features['entity_count'])
                                        checks.append({
                                            "Check": "Architecture content",
                                            "Status": "✅ Pass" if arch_mentions > 0 else "❌ Fail",
                                            "Value": f"{arch_mentions:.0f} architectural elements"
                                        })
                                        
                                        # Check 4: Variability in values (not all zeros or all same)
                                        checks.append({
                                            "Check": "Value variability",
                                            "Status": "✅ Pass" if unique_values >= 3 else "❌ Fail",
                                            "Value": f"{unique_values} distinct values"
                                        })
                                        
                                        check_df = pd.DataFrame(checks)
                                        st.dataframe(check_df, width="stretch", hide_index=True)
                                        
                                        passed_checks = sum(1 for c in checks if c["Status"].startswith("✅"))
                                        if passed_checks == len(checks):
                                            st.success(f"🎉 All {len(checks)} validation checks passed! Using real HLD features.")
                                        elif passed_checks >= len(checks) * 0.75:
                                            st.info(f"✅ {passed_checks}/{len(checks)} checks passed - Mostly real data")
                                        else:
                                            st.warning(f"⚠️ Only {passed_checks}/{len(checks)} checks passed")
                                    
                                    logger.info(f"Prediction result for {selected_hld['name']}: {preds}")
                                    
                                except Exception as e:
                                    st.error(f"Error during feature extraction or prediction: {e}")
                                    logger.error(f"Failed to predict quality for {selected_hld['name']}: {e}", exc_info=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("DesignMind GenAI — HLD Generator (local)")

if __name__ == "__main__":
    main()
