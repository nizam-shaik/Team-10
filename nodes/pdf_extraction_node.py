from __future__ import annotations
"""
PDF Extraction Node - Orchestrates PDF extraction stage
"""

# TODO: Implement PDFExtractionNode extending BaseNode
# TODO: Import PDFExtractionAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create PDFExtractionAgent instance
#       - Call agent.process(state)
#       - Validate extracted content
#       - Update state.extracted with ExtractedContent
#       - Update stage status to completed
#       - Return updated state
# TODO: Add content validation
#       - Check markdown is not empty
#       - Validate metadata fields (title, date, version)
#       - Ensure schema_version is set
# TODO: Implement error handling
#       - Catch file not found errors
#       - Handle PDF parsing failures
#       - Add meaningful error messages to state
# TODO: Consider content preprocessing
#       - Trim excessive whitespace
#       - Normalize markdown formatting
#       - Remove binary artifacts if any
# TODO: Log extraction metrics
#       - PDF file size
#       - Extracted markdown length
#       - Metadata details
"""
PDF Extraction Node - Orchestrates PDF extraction stage
"""

import logging
import os
from pathlib import Path
from typing import Any

from .base_node import BaseNode
from agent.pdf_agent import PDFExtractionAgent
from state.models import HLDState, ExtractedContent, ProcessingStatus

logger = logging.getLogger(__name__)


class PDFExtractionNode(BaseNode):
    """
    Node that manages PDF loading, extraction, and state updates.
    """

    def __init__(self):
        super().__init__(name="pdf_extraction", agent=PDFExtractionAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Run the PDF extraction process:
          - Loads PDF
          - Calls the PDFExtractionAgent
          - Validates and normalizes output
          - Updates HLDState
        """
        logger.info("[PDFExtractionNode] Starting PDF extraction workflow.")
        state.set_status("pdf_extraction", "processing", "Extracting PDF content")

        try:
            # Validate PDF file path
            pdf_path = Path(state.pdf_path or "")
            if not pdf_path.exists() or not pdf_path.is_file():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"Invalid file type for PDF extraction: {pdf_path.suffix}")

            # Log file size
            pdf_size_kb = os.path.getsize(pdf_path) / 1024
            logger.info(f"[PDFExtractionNode] Reading PDF: {pdf_path.name} ({pdf_size_kb:.1f} KB)")

            # Call agent for extraction
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("Invalid extraction result structure from agent")

            extracted_data = result.get("extracted") or result
            markdown = (extracted_data.get("markdown") or "").strip()
            meta = extracted_data.get("meta") or {}
            schema_version = extracted_data.get("schema_version", "1.0")
            generated_at = extracted_data.get("generated_at")
            source = extracted_data.get("source") or {"path": str(pdf_path)}

            # --- Content Validation ---
            if not markdown:
                raise ValueError("Extraction returned empty markdown content.")
            if not meta or not isinstance(meta, dict):
                meta = {"title": pdf_path.stem, "version": "1.0"}
            if "title" not in meta:
                meta["title"] = pdf_path.stem
            if "version" not in meta:
                meta["version"] = "1.0"
            if "date" not in meta:
                from datetime import datetime
                meta["date"] = datetime.now().strftime("%Y-%m-%d")

            # Normalize markdown (trim excessive whitespace)
            markdown = "\n".join(line.rstrip() for line in markdown.splitlines()).strip()

            # Build ExtractedContent
            extracted_obj = ExtractedContent(
                markdown=markdown,
                meta=meta,
                schema_version=schema_version,
                source=source
            )

            # Update state
            state.extracted = extracted_obj
            state.set_status("pdf_extraction", "completed", "PDF extraction completed")

            # Log metrics
            logger.info(f"[PDFExtractionNode] Extraction completed successfully.")
            logger.info(f"  Markdown length: {len(markdown):,} chars")
            logger.info(f"  Metadata: {meta}")

            return state

        except FileNotFoundError as e:
            logger.exception("PDF file not found")
            state.add_error(str(e))
            state.set_status("pdf_extraction", "failed", str(e))
            return state

        except Exception as e:
            logger.exception("PDF extraction failed")
            state.add_error(str(e))
            state.set_status("pdf_extraction", "failed", str(e))
            return state
