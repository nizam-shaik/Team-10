from __future__ import annotations
"""
Pydantic data models for HLD workflow state management
Type-safe state structures for all stages of HLD generation
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ==========================================================
# Processing Status
# ==========================================================
class ProcessingStatus(BaseModel):
    """Represents the current processing status of a workflow stage."""
    status: str = Field(default="pending", description="Current status: pending, processing, completed, or failed")
    message: Optional[str] = Field(default=None, description="Optional status message for this stage")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this status was last updated")


# ==========================================================
# Extracted Content
# ==========================================================
class ExtractedContent(BaseModel):
    """Represents structured content extracted from a PDF."""
    markdown: str = Field(default="", description="Extracted PDF content as markdown text")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata (title, version, author, etc.)")
    schema_version: str = Field(default="1.0", description="Schema version for compatibility tracking")
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp of extraction")
    source: Dict[str, Any] = Field(default_factory=dict, description="Source metadata including file path and origin")


# ==========================================================
# Authentication Data
# ==========================================================
class AuthenticationData(BaseModel):
    """Authentication and security configuration extracted from requirements."""
    actors: List[str] = Field(default_factory=list, description="Actors or user roles involved in authentication")
    flows: List[str] = Field(default_factory=list, description="Authentication flows (OAuth2, JWT, etc.)")
    idp_options: List[str] = Field(default_factory=list, description="Identity Provider options (Okta, Auth0, etc.)")
    threats: List[str] = Field(default_factory=list, description="Known security threats identified in the system")


# ==========================================================
# Integration Data
# ==========================================================
class IntegrationData(BaseModel):
    """Represents external system integrations and protocols."""
    system: str = Field(..., description="External system name")
    purpose: str = Field(default="", description="Purpose of integration")
    protocol: str = Field(default="", description="Communication protocol (REST, gRPC, WebSocket, etc.)")
    auth: str = Field(default="", description="Authentication method (API Key, OAuth, etc.)")
    endpoints: List[str] = Field(default_factory=list, description="API endpoints used in integration")
    data_contract: Dict[str, List[str]] = Field(default_factory=dict, description="Input/output data contracts")


# ==========================================================
# Entity Data
# ==========================================================
class EntityData(BaseModel):
    """Represents an entity in the domain model."""
    name: str = Field(..., description="Entity name")
    attributes: List[str] = Field(default_factory=list, description="Entity properties or fields")


# ==========================================================
# API Data
# ==========================================================
class APIData(BaseModel):
    """Represents API specification details."""
    name: str = Field(..., description="API name or endpoint")
    description: Optional[str] = Field(default=None, description="Brief description of API purpose")
    request: Dict[str, str] = Field(default_factory=dict, description="Request schema fields and types")
    response: Dict[str, str] = Field(default_factory=dict, description="Response schema fields and types")


# ==========================================================
# Domain Data
# ==========================================================
class DomainData(BaseModel):
    """Represents the overall domain model and API design."""
    entities: List[EntityData] = Field(default_factory=list, description="List of domain entities")
    apis: List[APIData] = Field(default_factory=list, description="List of API specifications")
    diagram_plan: Dict[str, Any] = Field(default_factory=dict, description="Plan for generating class diagrams")


# ==========================================================
# Risk Data
# ==========================================================
class RiskData(BaseModel):
    """Represents a risk item with mitigation plan."""
    id: str = Field(default="", description="Risk identifier")
    desc: str = Field(default="", description="Risk description")
    assumption: str = Field(default="", description="Assumption related to this risk")
    mitigation: str = Field(default="", description="Mitigation strategy")
    impact: int = Field(default=3, ge=1, le=5, description="Impact score (1–5)")
    likelihood: int = Field(default=3, ge=1, le=5, description="Likelihood score (1–5)")


# ==========================================================
# Behavior Data
# ==========================================================
class BehaviorData(BaseModel):
    """Represents system behaviors, use cases, NFRs, and risks."""
    use_cases: List[str] = Field(default_factory=list, description="List of user stories or use cases")
    nfrs: Dict[str, List[str]] = Field(default_factory=dict, description="Non-functional requirements grouped by category")
    risks: List[RiskData] = Field(default_factory=list, description="List of identified risks and mitigations")
    diagram_plan: Dict[str, Any] = Field(default_factory=dict, description="Sequence diagram plan for use case modeling")


# ==========================================================
# Diagram Data
# ==========================================================
class DiagramData(BaseModel):
    """Holds generated diagram artifacts."""
    class_text: str = Field(default="", description="Mermaid syntax for class diagram")
    sequence_texts: List[str] = Field(default_factory=list, description="Mermaid syntax for sequence diagrams")
    sequence_titles: Dict[str, str] = Field(default_factory=dict, description="Use case titles for each sequence diagram (e.g., {'sequence_1': 'configure report templates'})")
    class_img_path: Optional[str] = Field(default=None, description="Rendered class diagram file path")
    seq_img_paths: List[str] = Field(default_factory=list, description="Rendered sequence diagram file paths")


# ==========================================================
# Output Data
# ==========================================================
class OutputData(BaseModel):
    """Stores all output artifacts of the workflow."""
    output_dir: str = Field(default="", description="Base output directory")
    hld_md_path: str = Field(default="", description="Path to generated HLD.md file")
    hld_html_path: str = Field(default="", description="Path to generated HLD.html file")
    diagrams_html_path: str = Field(default="", description="Path to Diagrams.html viewer")
    risk_heatmap_path: Optional[str] = Field(default=None, description="Path to generated risk heatmap image")


# ==========================================================
# HLDState (Main Workflow State)
# ==========================================================
class HLDState(BaseModel):
    """Main state container used throughout the LangGraph HLD workflow."""
    pdf_path: str = Field(default="", description="Input PDF file path")
    requirement_name: str = Field(default="", description="Requirement identifier or name")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration parameters")

    # Stage-wise processing states
    status: Dict[str, ProcessingStatus] = Field(default_factory=dict, description="Status for each workflow stage")

    # Step data
    extracted: Optional[ExtractedContent] = None
    authentication: Optional[AuthenticationData] = None
    integrations: List[IntegrationData] = Field(default_factory=list)
    domain: Optional[DomainData] = None
    behavior: Optional[BehaviorData] = None
    diagrams: Optional[DiagramData] = None
    output: Optional[OutputData] = None

    # Error and warning tracking
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # ======================================================
    # Validation & Utility Methods
    # ======================================================
    def has_errors(self) -> bool:
        """Return True if there are any recorded errors."""
        return bool(self.errors)

    def add_error(self, msg: str) -> None:
        """Append an error message to the state."""
        if msg:
            self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        """Append a warning message to the state."""
        if msg:
            self.warnings.append(msg)

    def update_status(self, stage: str, status: str, message: str = "", error: str = "") -> None:
        """Update processing status for a workflow stage."""
        self.status[stage] = ProcessingStatus(status=status, message=message or error)

    def set_status(self, stage: str, status: str, message: str = "", error: str = "") -> None:
        """Alias for update_status for compatibility with nodes."""
        self.update_status(stage, status, message, error)

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Override dict() to ensure serialization compatibility."""
        kwargs.setdefault("exclude_none", True)
        return super().dict(*args, **kwargs)

    @field_validator("pdf_path")
    def validate_pdf_path(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("pdf_path must be a valid string path")
        return v

    @field_validator("requirement_name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("requirement_name cannot be empty")
        return v

    class Config:
        """Pydantic model configuration"""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_schema_extra = {"version": "1.0.0", "author": "GenAI DesignMind Team"}