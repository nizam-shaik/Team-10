
from __future__ import annotations
"""
Input/Output schemas and validation for workflow
Configuration and workflow input/output definitions
"""

# TODO: Import Pydantic BaseModel and Field
# TODO: Import HLDState from models
# TODO: Define ConfigSchema model
#       - render_images: bool = True (whether to render diagrams to images)
#       - image_format: str = "png" (svg or png)
#       - renderer: str = "kroki" (kroki or mmdc)
#       - theme: str = "default" (default, neutral, dark)
# TODO: Define WorkflowInput model
#       - pdf_path: str (path to input PDF)
#       - config: ConfigSchema (workflow configuration)
# TODO: Define WorkflowOutput model
#       - success: bool (workflow success flag)
#       - state: HLDState (final workflow state)
#       - output_paths: Dict[str, str] (paths to output files)
#       - processing_time: float (execution time in seconds)
#       - errors: List[str] (error messages)
#       - warnings: List[str] (warning messages)
# TODO: Implement create_initial_state(pdf_path: str, config: ConfigSchema) -> HLDState
#       - Validate PDF path exists
#       - Extract requirement name from PDF filename
#       - Initialize status dict with all stages as "pending"
#       - Create initial HLDState with empty collections
#       - Return initialized state ready for workflow
# TODO: Add validation methods to ConfigSchema
#       - Validate renderer is in allowed values
#       - Validate image_format is svg or png
#       - Validate theme is in allowed values
# TODO: Add validation methods to WorkflowInput
#       - Validate pdf_path is not empty
#       - Validate PDF file exists
#       - Validate config object
# TODO: Implement configuration defaults
# TODO: Add error handling for invalid configurations
# TODO: Consider configuration serialization for logging
"""
Input/Output schemas and validation for workflow
Configuration and workflow input/output definitions
"""

from typing import Dict, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ValidationError
from .models import HLDState, ProcessingStatus


# ==========================================================
# Config Schema
# ==========================================================
class ConfigSchema(BaseModel):
    """Configuration options controlling diagram rendering and workflow behavior."""

    render_images: bool = Field(default=True, description="Whether to render Mermaid diagrams into image files")
    image_format: str = Field(default="png", description="Image format for rendered diagrams: svg or png")
    renderer: str = Field(default="kroki", description="Diagram rendering engine: kroki or mmdc")
    theme: str = Field(default="default", description="Diagram theme: default, neutral, or dark")

    # --------------------------------------------
    # Validation
    # --------------------------------------------
    @field_validator("renderer")
    def validate_renderer(cls, v):
        allowed = {"kroki", "mmdc"}
        if v not in allowed:
            raise ValueError(f"Invalid renderer '{v}'. Must be one of {allowed}.")
        return v

    @field_validator("image_format")
    def validate_image_format(cls, v):
        allowed = {"svg", "png"}
        if v not in allowed:
            raise ValueError(f"Invalid image_format '{v}'. Must be one of {allowed}.")
        return v

    @field_validator("theme")
    def validate_theme(cls, v):
        allowed = {"default", "neutral", "dark"}
        if v not in allowed:
            raise ValueError(f"Invalid theme '{v}'. Must be one of {allowed}.")
        return v

    class Config:
        validate_assignment = True
        schema_extra = {
            "example": {
                "render_images": True,
                "image_format": "png",
                "renderer": "kroki",
                "theme": "dark",
            }
        }


# ==========================================================
# Workflow Input Schema
# ==========================================================
class WorkflowInput(BaseModel):
    """Defines the input required to start a workflow."""
    pdf_path: str = Field(..., description="Absolute or relative path to the input PDF requirements document")
    config: ConfigSchema = Field(default_factory=ConfigSchema, description="Workflow configuration options")

    # --------------------------------------------
    # Validation
    # --------------------------------------------
    @field_validator("pdf_path")
    def validate_pdf_path(cls, v):
        if not v:
            raise ValueError("pdf_path cannot be empty.")
        path = Path(v)
        if not path.exists() or not path.is_file():
            raise ValueError(f"PDF file not found: {v}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("Input file must be a PDF (.pdf).")
        return str(path)

    @field_validator("config")
    def validate_config(cls, v):
        if not isinstance(v, ConfigSchema):
            raise ValueError("config must be a valid ConfigSchema object.")
        return v

    class Config:
        validate_assignment = True


# ==========================================================
# Workflow Output Schema
# ==========================================================
class WorkflowOutput(BaseModel):
    """Encapsulates the results of a workflow execution."""
    success: bool = Field(default=False, description="Whether the workflow completed successfully")
    state: HLDState = Field(..., description="Final state of the HLD workflow")
    output_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to all generated output artifacts")
    processing_time: float = Field(default=0.0, description="Workflow execution time in seconds")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    warnings: List[str] = Field(default_factory=list, description="List of warning messages")

    class Config:
        validate_assignment = True


# ==========================================================
# Helper: create_initial_state
# ==========================================================
def create_initial_state(pdf_path: str, config: ConfigSchema) -> HLDState:
    """
    Creates a fully initialized HLDState ready for workflow execution.
    - Validates file path
    - Initializes stage statuses
    - Extracts requirement name
    """
    try:
        path = Path(pdf_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

        requirement_name = path.stem  # filename without extension
        stages = [
            "pdf_extraction",
            "auth_integrations",
            "domain_api_design",
            "behavior_quality",
            "diagram_generation",
            "output_composition",
        ]

        status_dict = {stage: ProcessingStatus(status="pending") for stage in stages}

        state = HLDState(
            pdf_path=str(path),
            requirement_name=requirement_name,
            config=config.dict(),
            status=status_dict,
            errors=[],
            warnings=[]
        )
        return state

    except (ValidationError, FileNotFoundError, Exception) as e:
        raise ValueError(f"Failed to initialize state: {e}")