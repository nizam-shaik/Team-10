from __future__ import annotations
"""
Output Composition Node - Generates final HLD documentation
"""

# TODO: Implement OutputCompositionNode extending BaseNode
# TODO: Import OutputAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create OutputAgent instance
#       - Call agent.process(state)
#       - Validate output files were created
#       - Update state.output with OutputData
#       - Update stage status to completed
#       - Return updated state
# TODO: Implement HLD composition
#       - Call hld_to_markdown() with all state data
#       - Generate comprehensive Markdown documentation
#       - Create table of contents
# TODO: Generate HTML output
#       - Convert Markdown to HTML
#       - Add CSS styling and formatting
#       - Embed diagram images
#       - Create navigation and links
# TODO: Create interactive viewers
#       - Build Diagrams.html interactive viewer
#       - Include diagram filters and search
#       - Add diagram metadata and descriptions
# TODO: Generate visualizations
#       - Create risk heatmap matrix
#       - Generate architecture diagram
#       - Create data flow diagrams
# TODO: Implement file management
#       - Create output directory structure
#       - Save all artifacts with proper naming
#       - Create index files and README
#       - Organize files by type (json, diagrams, hld)
# TODO: Error handling
#       - Handle file write failures
#       - Validate all output paths
#       - Create backup of important files
# TODO: Log output metrics
#       - Output file sizes and counts
#       - Generation time for each artifact
#       - Quality assessment scores
"""
Output Composition Node - Generates final HLD documentation
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any

from .base_node import BaseNode
from agent.output_agent import OutputAgent
from state.models import HLDState, OutputData

logger = logging.getLogger(__name__)


class OutputCompositionNode(BaseNode):
    """
    Node responsible for aggregating workflow results and generating
    final HLD markdown, HTML, and supporting visualization artifacts.
    """

    def __init__(self):
        super().__init__(name="output_composition", agent=OutputAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Executes OutputAgent to generate final HLD artifacts:
        Markdown, HTML, diagrams viewer, and risk visualizations.
        """
        logger.info("[OutputCompositionNode] Starting final HLD composition.")
        state.set_status("output_composition", "processing", "Composing final HLD output")

        try:
            # Run OutputAgent
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("OutputAgent did not return a valid result dict")

            # Extract generated paths (align with OutputAgent return keys)
            hld_md_path = result.get("hld_md") or result.get("hld_md_path")
            hld_html_path = result.get("hld_html") or result.get("hld_html_path")
            diagrams_html_path = result.get("diagrams_html") or result.get("diagrams_html_path")
            risk_heatmap_path = result.get("risk_heatmap") or result.get("risk_heatmap_path")

            # --- Validate Outputs ---
            missing = []
            for path_label, path_val in {
                "HLD Markdown": hld_md_path,
                "HLD HTML": hld_html_path,
                "Diagrams HTML": diagrams_html_path,
            }.items():
                if not path_val or not os.path.exists(path_val):
                    missing.append(path_label)

            if missing:
                msg = f"Missing output files: {', '.join(missing)}"
                logger.warning(f"[OutputCompositionNode] {msg}")
                state.add_warning(msg)

            # --- Directory Structure ---
            # Use Project/output directory by default
            project_dir = Path(__file__).resolve().parent.parent
            base_dir = Path(state.output.output_dir if state.output else project_dir / "output")
            (base_dir / "json").mkdir(parents=True, exist_ok=True)
            (base_dir / "diagrams").mkdir(parents=True, exist_ok=True)
            (base_dir / "hld").mkdir(parents=True, exist_ok=True)

            # --- File Validation ---
            all_paths = [hld_md_path, hld_html_path, diagrams_html_path, risk_heatmap_path]
            for path in all_paths:
                if path and os.path.exists(path):
                    size_kb = round(os.path.getsize(path) / 1024, 1)
                    logger.info(f"[OutputCompositionNode] Output file: {path} ({size_kb} KB)")

            # --- Update State ---
            output_data = OutputData(
                output_dir=str(base_dir),
                hld_md_path=hld_md_path or "",
                hld_html_path=hld_html_path or "",
                diagrams_html_path=diagrams_html_path or "",
                risk_heatmap_path=risk_heatmap_path or "",
            )

            state.output = output_data
            state.set_status("output_composition", "completed", "HLD documentation successfully generated")

            logger.info("[OutputCompositionNode] Final HLD documentation generated successfully.")
            return state

        except Exception as e:
            logger.exception("[OutputCompositionNode] Output composition failed.")
            state.add_error(str(e))
            state.set_status("output_composition", "failed", str(e))
            return state
