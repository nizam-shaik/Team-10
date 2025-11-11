from __future__ import annotations
"""
Diagram Generation Node - Converts design plans to visual diagrams
"""

# TODO: Implement DiagramGenerationNode extending BaseNode
# TODO: Import DiagramAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create DiagramAgent instance
#       - Call agent.process(state)
#       - Validate diagram data (Mermaid syntax)
#       - Update state.diagrams with DiagramData
#       - Update stage status to completed
#       - Return updated state
# TODO: Add Mermaid syntax validation
#       - Validate class diagram syntax
#       - Validate sequence diagram syntax
#       - Check for common Mermaid errors
# TODO: Implement diagram rendering
#       - Convert Mermaid to SVG or PNG based on config
#       - Use configured renderer (kroki or mmdc)
#       - Handle rendering failures gracefully
# TODO: Output directory management
#       - Create diagrams/ subdirectory
#       - Save Mermaid source files
#       - Save rendered image files
# TODO: Error handling
#       - Catch Mermaid syntax errors
#       - Handle renderer failures
#       - Provide fallback Mermaid text if rendering fails
# TODO: Image processing
#       - Optimize image sizes
#       - Generate thumbnails if needed
#       - Embed images in output
# TODO: Log diagram metrics
#       - Diagram file sizes
#       - Mermaid syntax validation results
#       - Rendering success/failure rates
"""
Diagram Generation Node - Converts design plans to visual diagrams
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from .base_node import BaseNode
from agent.diagram_agent import DiagramAgent
from state.models import HLDState, DiagramData

logger = logging.getLogger(__name__)


class DiagramGenerationNode(BaseNode):
    """
    Node responsible for converting design plans (domain + behavior)
    into visual diagrams using Mermaid + configured renderer.
    """

    def __init__(self):
        super().__init__(name="diagram_generation", agent=DiagramAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Run the DiagramAgent to generate Mermaid diagrams and rendered images.
        Validates syntax, manages output directories, and updates workflow state.
        """
        logger.info("[DiagramGenerationNode] Starting diagram generation process.")
        state.set_status("diagram_generation", "processing", "Generating visual diagrams")

        try:
            # Run DiagramAgent
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("DiagramAgent returned an invalid result object.")

            diagram_data = result.get("diagram")
            renderer_results = result.get("renderer_results", {})
            publish_results = result.get("publish_results", {})

            if not diagram_data:
                raise ValueError("No diagram data returned by DiagramAgent.")

            # --- Mermaid Syntax Validation ---
            self._validate_mermaid(diagram_data.get("class_text"), "class diagram")
            for seq_text in diagram_data.get("sequence_texts", []):
                self._validate_mermaid(seq_text, "sequence diagram")

            # --- Directory Management ---
            # Use Project/output directory
            project_dir = Path(__file__).resolve().parent.parent
            output_dir = Path(state.output.output_dir if state.output else project_dir / "output" / (state.requirement_name or "unknown")) / "diagrams"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save Mermaid sources
            if diagram_data.get("class_text"):
                class_path = output_dir / "class_diagram.mmd"
                class_path.write_text(diagram_data["class_text"], encoding="utf-8")
                logger.info(f"[DiagramGenerationNode] Saved class diagram source → {class_path}")

            for i, seq_text in enumerate(diagram_data.get("sequence_texts", []), start=1):
                seq_path = output_dir / f"sequence_{i}.mmd"
                seq_path.write_text(seq_text, encoding="utf-8")
                logger.info(f"[DiagramGenerationNode] Saved sequence diagram source → {seq_path}")

            # --- Image Rendering ---
            class_img = diagram_data.get("class_img_path")
            seq_imgs = diagram_data.get("seq_img_paths", [])

            # Fallback if rendering failed
            if not class_img or not os.path.exists(class_img):
                logger.warning("[DiagramGenerationNode] Class diagram image missing; fallback to Mermaid text.")
            if not seq_imgs:
                logger.warning("[DiagramGenerationNode] Sequence diagram images missing; fallback to Mermaid text.")

            # --- Image Metrics ---
            if class_img and os.path.exists(class_img):
                class_size = round(os.path.getsize(class_img) / 1024, 1)
                logger.info(f"[DiagramGenerationNode] Class diagram image size: {class_size} KB")

            for img_path in seq_imgs:
                if img_path and os.path.exists(img_path):
                    size_kb = round(os.path.getsize(img_path) / 1024, 1)
                    logger.info(f"[DiagramGenerationNode] Sequence diagram image size: {size_kb} KB")

            # --- Update State ---
            diagrams_obj = DiagramData(
                class_text=diagram_data.get("class_text", ""),
                sequence_texts=diagram_data.get("sequence_texts", []),
                sequence_titles=diagram_data.get("sequence_titles", {}),
                class_img_path=class_img,
                seq_img_paths=seq_imgs,
            )
            state.diagrams = diagrams_obj
            state.set_status("diagram_generation", "completed", "Diagram generation completed successfully")

            logger.info("[DiagramGenerationNode] Diagram generation completed successfully.")
            return state

        except Exception as e:
            logger.exception("[DiagramGenerationNode] Diagram generation failed.")
            state.add_error(str(e))
            state.set_status("diagram_generation", "failed", str(e))
            return state

    # --------------------------
    # Internal Helper Methods
    # --------------------------

    def _validate_mermaid(self, text: str, label: str) -> None:
        """
        Lightweight Mermaid syntax validator.
        Ensures valid diagram headers and checks for common formatting issues.
        """
        if not text or not isinstance(text, str):
            logger.warning(f"[DiagramGenerationNode] Empty {label} Mermaid text.")
            return

        header_pattern = re.compile(r"^\s*(classDiagram|sequenceDiagram)", re.IGNORECASE)
        if not header_pattern.search(text):
            logger.warning(f"[DiagramGenerationNode] {label} missing proper Mermaid header.")

        # Check for unclosed braces, arrows, or malformed blocks
        if text.count("{") != text.count("}"):
            logger.warning(f"[DiagramGenerationNode] {label} has mismatched braces.")
        if "->" not in text and "--" not in text:
            logger.warning(f"[DiagramGenerationNode] {label} may lack relationships or message flows.")
