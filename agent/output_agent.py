from __future__ import annotations
"""
Output Composition Agent
"""

# TODO: Implement OutputAgent class extending BaseAgent
# TODO: Aggregate all analysis results from workflow state
# TODO: Call hld_to_markdown() utility to generate comprehensive HLD
# TODO: Convert Markdown HLD to HTML with styling and formatting
# TODO: Create Diagrams.html interactive viewer with embedded diagrams
# TODO: Generate risk heatmap visualization (impact x likelihood matrix)
# TODO: Create output directory structure (json/, diagrams/, hld/)
# TODO: Save all artifacts to filesystem with proper file naming
# TODO: Generate table of contents for HLD documentation
# TODO: Create index pages linking all generated documents
# TODO: Embed diagram images in HTML with proper paths and captions
# TODO: Add CSS styling for professional HLD presentation
# TODO: Include executive summary in final output
# TODO: Create metadata file with generation timestamp and configuration
# TODO: Implement file organization: separate json responses, diagrams, and HLD
# TODO: Handle file write errors and missing output directories gracefully
"""
Output Composition Agent
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base_agent import BaseAgent
from state.models import HLDState, OutputData, ProcessingStatus
from utils.compose_output import hld_to_markdown, save_markdown
from utils.risk_heatmap import generate_risk_heatmap
from diagram_publisher import publish_diagrams

logger = logging.getLogger(__name__)


class OutputAgent(BaseAgent):
    """
    Compose final HLD outputs:
      - aggregate analysis results
      - generate HLD.md (and HLD.html)
      - save JSON raw responses under json/
      - save diagram sources/images under diagrams/
      - create risk heatmap image
      - write metadata & index files
    """

    @property
    def system_prompt(self) -> str:
        # OutputAgent doesn't call LLM for main composition; keep a short descriptor
        return "Output composition agent: aggregate results and write HLD artifacts."

    def process(self, state: HLDState) -> Dict[str, Any]:
        """
        Compose and write all output artifacts to disk and update state.output.
        Returns a dict with paths to generated content or an error entry.
        """
        state.set_status("output_composition", "processing", "Composing final HLD outputs")
        try:
            # Determine output base directory
            base_out = Path("output") / (state.requirement_name or Path(state.pdf_path or "unknown").stem or "requirement")
            json_dir = base_out / "json"
            diagrams_dir = base_out / "diagrams"
            hld_dir = base_out / "hld"

            # Create directories
            for d in (json_dir, diagrams_dir, hld_dir):
                d.mkdir(parents=True, exist_ok=True)

            # 1) Save raw JSON responses (from state.meta / state fields) for traceability
            raw_json_map: Dict[str, str] = {}
            try:
                # Save major components: extracted, authentication, domain, behavior, diagrams, meta
                components = {
                    "extracted": getattr(state.extracted, "dict", lambda: {})() if state.extracted else None,
                    "authentication": state.authentication.dict() if state.authentication else None,
                    "integrations": [i.dict() for i in (state.integrations or [])],
                    "domain": state.domain.dict() if state.domain else None,
                    "behavior": state.behavior.dict() if state.behavior else None,
                    "diagrams": state.diagrams.dict() if state.diagrams else None,
                    # Use extracted.meta as the canonical metadata source
                    "meta": (getattr(state.extracted, "meta", {}) or {})
                }
                for name, obj in components.items():
                    if obj is None:
                        continue
                    path = json_dir / f"{name}.json"
                    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                    raw_json_map[name] = str(path.resolve())
            except Exception as e:
                logger.exception("Failed saving JSON components")
                state.add_warning(f"Failed to save some JSON components: {e}")

            # 2) Generate HLD markdown via utility
            try:
                # Compose HLD using fields expected by hld_to_markdown
                prd_markdown = getattr(state.extracted, "markdown", "") if state.extracted else ""
                class_text = (state.diagrams.class_text if state.diagrams else None) if hasattr(state, "diagrams") else None
                seq_texts = (state.diagrams.sequence_texts if state.diagrams else []) if hasattr(state, "diagrams") else []

                hld_md_text = hld_to_markdown(
                    requirement_name=state.requirement_name or Path(state.pdf_path or "requirement").stem,
                    prd_markdown=prd_markdown,
                    authentication=(state.authentication.dict() if state.authentication else {}),
                    integrations=[i.dict() for i in (state.integrations or [])],
                    entities=[e.dict() for e in (state.domain.entities if state.domain and getattr(state.domain, "entities", None) else [])],
                    apis=[a.dict() for a in (state.domain.apis if state.domain and getattr(state.domain, "apis", None) else [])],
                    use_cases=(state.behavior.use_cases if state.behavior else []),
                    nfrs=(state.behavior.nfrs if state.behavior else {}),
                    risks=[r.dict() for r in (state.behavior.risks if state.behavior else [])],
                    class_mermaid_text=class_text or "",
                    sequence_mermaid_texts=seq_texts or [],
                    class_img=(state.diagrams.class_img_path if state.diagrams and getattr(state.diagrams, "class_img_path", None) else None),
                    seq_imgs=(state.diagrams.seq_img_paths if state.diagrams and getattr(state.diagrams, "seq_img_paths", None) else []),
                    hld_base_dir=str(hld_dir),
                )
                hld_md_path = save_markdown(hld_md_text, hld_dir / "HLD.md")
            except Exception as e:
                logger.exception("Failed composing HLD markdown")
                state.add_error(f"Failed to compose HLD markdown: {e}")
                state.set_status("output_composition", "failed", "Failed to compose HLD markdown")
                return {"error": str(e)}

            # 3) Convert Markdown -> HTML (simple conversion using markdown lib)
            hld_html_path = None
            try:
                import markdown as _md
                md_html = _md.markdown(hld_md_text, extensions=["fenced_code", "tables", "toc"])
                # simple styling wrapper (inline CSS)
                html_wrap = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>HLD - {state.requirement_name}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu;padding:24px;max-width:1000px;margin:auto;}}
h1{{font-size:1.8rem}}
pre{{background:#f6f6f6;padding:12px;overflow:auto;border-radius:8px}}
code{{font-family:monospace}}
.table-of-contents{{margin-bottom:20px}}
.diagram{{max-width:100%}}
</style>
</head>
<body>
<h1>High-Level Design — {state.requirement_name}</h1>
<p><em>Generated: {datetime.utcnow().isoformat()}</em></p>
{md_html}
</body>
</html>"""
                hld_html_path = hld_dir / "HLD.html"
                hld_html_path.write_text(html_wrap, encoding="utf-8")
            except Exception as e:
                logger.exception("Failed to convert HLD markdown to HTML")
                state.add_warning(f"HLD HTML generation failed: {e}")
                hld_html_path = None

            # 4) Generate risk heatmap if risks exist
            risk_heatmap_path = None
            try:
                risks_list = [r.dict() for r in (state.behavior.risks if state.behavior else [])]
                if risks_list:
                    heatmap_out = diagrams_dir / "risk_heatmap.png"
                    generate_risk_heatmap(risks_list, str(heatmap_out), title=f"Risk Heatmap - {state.requirement_name}")
                    risk_heatmap_path = str(heatmap_out.resolve())
            except Exception as e:
                logger.exception("Risk heatmap generation failed")
                state.add_warning(f"Risk heatmap generation failed: {e}")
                risk_heatmap_path = None

            # 5) Copy or reference diagram viewer (if publish_diagrams produced something earlier)
            diagrams_viewer = None
            try:
                # If state.diagrams has Mermaid texts, publish a viewer
                mermaid_map = {}
                sequence_titles = {}
                if state.diagrams:
                    if getattr(state.diagrams, "class_text", None):
                        mermaid_map["class_diagram"] = state.diagrams.class_text
                    for idx, seq in enumerate(getattr(state.diagrams, "sequence_texts", []) or []):
                        mermaid_map[f"sequence_{idx+1}"] = seq
                    # Get sequence titles from state
                    sequence_titles = getattr(state.diagrams, "sequence_titles", {}) or {}
                    logger.info(f"[OutputAgent DEBUG] Retrieved sequence_titles from state: {sequence_titles}")

                publish_res = {"full_html": None, "hld_html": None}
                if mermaid_map:
                    publish_res = publish_diagrams(
                        mermaid_map=mermaid_map,
                        out_dir=str(diagrams_dir),
                        title=f"HLD Diagrams - {state.requirement_name or 'HLD'}",
                        theme=(state.config.get("theme") if isinstance(state.config, dict) else "default"),
                        preview=False,
                        save_fullpage_html=True,
                        hld_markdown=None,
                        hld_html_out_path=str(hld_html_path) if hld_html_path else None,
                        sequence_titles=sequence_titles,
                    )
                    diagrams_viewer = publish_res.get("full_html")
            except Exception as e:
                logger.exception("Failed to publish diagrams viewer")
                state.add_warning(f"Diagrams viewer publish failed: {e}")
                diagrams_viewer = None

            # 6) Write metadata and index
            try:
                metadata = {
                    "requirement_name": state.requirement_name,
                    "pdf_path": state.pdf_path,
                    "generated_at": datetime.utcnow().isoformat(),
                    "config": state.config,
                    "outputs": {
                        "hld_md": str(hld_md_path.resolve()) if hld_md_path else None,
                        "hld_html": str(hld_html_path.resolve()) if hld_html_path else None,
                        "diagrams_html": diagrams_viewer,
                        "risk_heatmap": risk_heatmap_path,
                        "json": raw_json_map,
                    },
                    # Persist extracted metadata if available
                    "meta": (getattr(state.extracted, "meta", {}) or {}),
                }
                meta_path = base_out / "metadata.json"
                meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

                # create a simple index.html linking artifacts
                index_html = "<!doctype html><html><head><meta charset='utf-8'><title>HLD Outputs</title></head><body>"
                index_html += f"<h1>HLD Outputs — {state.requirement_name}</h1><ul>"
                if hld_md_path:
                    index_html += f"<li><a href='hld/HLD.md'>HLD.md</a></li>"
                if hld_html_path:
                    index_html += f"<li><a href='hld/HLD.html'>HLD.html</a></li>"
                if diagrams_viewer:
                    index_html += f"<li><a href='diagrams/{Path(diagrams_viewer).name}'>Diagrams Viewer</a></li>"
                if risk_heatmap_path:
                    index_html += f"<li><a href='diagrams/{Path(risk_heatmap_path).name}'>Risk Heatmap</a></li>"
                index_html += "<li>Raw JSON outputs:</li><ul>"
                for k, p in raw_json_map.items():
                    index_html += f"<li><a href='json/{Path(p).name}'>{k}.json</a></li>"
                index_html += "</ul></ul></body></html>"
                (base_out / "index.html").write_text(index_html, encoding="utf-8")
            except Exception as e:
                logger.exception("Failed to write metadata/index")
                state.add_warning(f"Failed to write metadata/index: {e}")

            # 7) Update state.output
            outdata = OutputData(
                output_dir=str(base_out.resolve()),
                hld_md_path=str(hld_md_path.resolve()) if hld_md_path else "",
                hld_html_path=str(hld_html_path.resolve()) if hld_html_path else "",
                diagrams_html_path=str(diagrams_viewer) if diagrams_viewer else "",
                risk_heatmap_path=str(risk_heatmap_path) if risk_heatmap_path else None,
            )
            state.output = outdata

            state.set_status("output_composition", "completed", "Output composition completed")
            logger.info("[OutputAgent] Output composition completed successfully.")
            return {
                "output_dir": outdata.output_dir,
                "hld_md": outdata.hld_md_path,
                "hld_html": outdata.hld_html_path,
                "diagrams_html": outdata.diagrams_html_path,
                "risk_heatmap": outdata.risk_heatmap_path,
                "metadata": str((base_out / "metadata.json").resolve()),
            }

        except Exception as exc:
            logger.exception("OutputAgent failed")
            state.add_error(str(exc))
            state.set_status("output_composition", "failed", str(exc))
            return {"error": str(exc)}
