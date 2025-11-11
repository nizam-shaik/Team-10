from __future__ import annotations
"""
PDF Extraction Agent for converting PDFs to structured markdown
"""

# TODO: Implement PDFExtractionAgent class extending BaseAgent
# TODO: Load PDF files from filesystem and extract bytes
# TODO: Send PDF content to Gemini API with system prompt
# TODO: Parse JSON response containing markdown and metadata
# TODO: Handle fallback cases when JSON parsing fails (wrap as markdown)
# TODO: Implement title extraction from markdown using regex patterns
# TODO: Normalize extracted data with required fields (markdown, meta, title, date)
# TODO: Update HLDState with extracted content and processing status
# TODO: Handle PDF file validation (exists, is PDF, readable)
# TODO: Implement error handling for file read failures
# TODO: Add metadata extraction (title, version, date from PDF or LLM response)
# TODO: Create ExtractedContent objects with schema_version tracking
# TODO: Support both direct PDF bytes and base64 encoding if needed
# TODO: Implement content length validation and truncation if needed
"""
PDF Extraction Agent for converting PDFs to structured markdown
"""

import os
import re
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import fitz  # PyMuPDF
from pydantic import ValidationError

from .base_agent import BaseAgent
from state.models import HLDState, ExtractedContent, ProcessingStatus


# ==========================================================
# PDF Extraction Agent
# ==========================================================
class PDFExtractionAgent(BaseAgent):
    """
    Extracts structured Markdown content and metadata from input PDFs using Gemini.
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are a document analysis model specialized in converting PDF-based requirement documents "
            "into structured Markdown format suitable for downstream analysis. "
            "Return your output as valid JSON with the following structure:\n"
            "{\n"
            '  "markdown": "<full text in markdown>",\n'
            '  "meta": {"title": "<title>", "version": "<version>", "date": "<date>"}\n'
            "}\n"
            "Ensure all sections (Objectives, Scope, Features, etc.) are properly formatted with Markdown headings."
        )

    # ------------------------------------------------------
    # Core Processing
    # ------------------------------------------------------
    def process(self, state: HLDState) -> Dict[str, Any]:
        """
        Extract text content and metadata from the input PDF file, update state.
        """
        pdf_path = Path(state.pdf_path)
        state.set_status("pdf_extraction", "processing", "Extracting content from PDF")

        try:
            if not pdf_path.exists() or not pdf_path.is_file():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(f"Invalid file type: {pdf_path.suffix}. Expected .pdf")

            logging.info(f"[PDFExtractionAgent] Reading PDF: {pdf_path}")
            text = self._extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("PDF appears empty or unreadable")

            # Send text to LLM for markdown conversion
            # Use more text for better context (up to 100k chars)
            prompt = f"Extracted Text:\n{text[:100000]}"
            
            # ✅ Use LangChain's call_llm with automatic retry and JSON parsing
            llm_response = self.call_llm(prompt, parse_json=True)
            self.log_cost(llm_response)

            # Parse structured result
            parsed = llm_response.parsed_json or {}
            markdown = parsed.get("markdown") or text
            meta = parsed.get("meta") or self._extract_metadata_fallback(markdown)

            # Truncate extremely long markdown
            if len(markdown) > 120_000:
                markdown = markdown[:120_000] + "\n\n[Truncated for length]"

            extracted = ExtractedContent(
                markdown=markdown.strip(),
                meta=meta,
                schema_version="1.0",
                source={"path": str(pdf_path)}
            )

            state.extracted = extracted
            state.set_status("pdf_extraction", "completed", "PDF extraction completed successfully")
            logging.info("[PDFExtractionAgent] Extraction complete.")
            return {"extracted": extracted.dict()}

        except Exception as e:
            logging.exception("[PDFExtractionAgent] PDF extraction failed.")
            state.add_error(str(e))
            state.set_status("pdf_extraction", "failed", str(e))
            return {"error": str(e)}

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF using PyMuPDF (fitz).
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF: {e}")

    def _extract_metadata_fallback(self, markdown: str) -> Dict[str, Any]:
        """
        Attempt to extract title, version, and date heuristically from Markdown text.
        """
        title_match = re.search(r"(?im)^(?:#|##)\s*(.+)", markdown)
        version_match = re.search(r"(?i)version[:\s]*([\d.]+)", markdown)
        date_match = re.search(r"(?i)\b(?:date|updated)[:\s]*(\w+\s*\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", markdown)

        title = title_match.group(1).strip() if title_match else "Untitled Document"
        version = version_match.group(1).strip() if version_match else "1.0"
        date = date_match.group(1).strip() if date_match else "Unknown"

        return {
            "title": title,
            "version": version,
            "date": date
        }

    def _encode_pdf_base64(self, pdf_path: Path) -> str:
        """
        Encode PDF content in base64 (optional if model supports binary input).
        """
        with open(pdf_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
