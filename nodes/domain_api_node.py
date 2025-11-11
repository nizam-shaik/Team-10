from __future__ import annotations
"""
Domain and API Design Node - Creates domain model and API specifications
"""

# TODO: Implement DomainAPINode extending BaseNode
# TODO: Import DomainAPIAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create DomainAPIAgent instance
#       - Call agent.process(state)
#       - Validate domain entities and APIs
#       - Update state.domain with DomainData
#       - Update stage status to completed
#       - Return updated state
# TODO: Add domain validation
#       - Ensure entities have name and attributes
#       - Verify each entity has ≥3 attributes
#       - Validate API specs have name, request, response
#       - Check request/response are dictionaries with fields
# TODO: Implement entity relationship mapping
#       - Extract relationships from diagram plan
#       - Create Mermaid-compatible relationship strings
#       - Validate cardinality notations
# TODO: API schema validation
#       - Validate request/response field types
#       - Check for required vs optional fields
#       - Ensure API names are REST-style
# TODO: Generate diagram plan
#       - Create class names from entities
#       - Define relationships in Mermaid format
#       - Include API endpoints in class diagram
# TODO: Log domain insights
#       - Number of entities designed
#       - Number of APIs defined
#       - Relationship count
"""
Domain and API Design Node - Creates domain model and API specifications
"""

import logging
import re
from typing import List, Dict, Any

from .base_node import BaseNode
from agent.domain_agent import DomainAPIAgent
from state.models import (
    HLDState,
    DomainData,
    EntityData,
    APIData,
)

logger = logging.getLogger(__name__)


class DomainAPINode(BaseNode):
    """
    Node responsible for domain modeling and API specification generation.
    """

    def __init__(self):
        super().__init__(name="domain_api_design", agent=DomainAPIAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Executes the DomainAPIAgent, validates results, and updates the workflow state.
        """
        logger.info("[DomainAPINode] Starting domain and API design analysis.")
        state.set_status("domain_api_design", "processing", "Designing domain model and APIs")

        try:
            # Execute agent to get structured domain model and APIs
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("Invalid result format returned by DomainAPIAgent")

            entities_data = result.get("entities", [])
            apis_data = result.get("apis", [])
            diagram_plan = result.get("diagram_plan", {})

            # --- Entity Validation ---
            validated_entities: List[EntityData] = []
            for e in entities_data:
                try:
                    name = str(e.get("name") or "").strip()
                    attrs = [a.strip() for a in (e.get("attributes") or []) if a.strip()]
                    if not name:
                        raise ValueError("Entity missing name")
                    if len(attrs) < 3:
                        logger.warning(f"[DomainAPINode] Entity '{name}' has fewer than 3 attributes.")
                    validated_entities.append(EntityData(name=name, attributes=attrs))
                except Exception as ex:
                    logger.warning(f"[DomainAPINode] Skipping invalid entity: {ex}")

            # --- API Validation ---
            validated_apis: List[APIData] = []
            for api in apis_data:
                try:
                    name = str(api.get("name") or "").strip()
                    if not name:
                        raise ValueError("API missing name")

                    # Validate request/response schemas
                    request = api.get("request") or {}
                    response = api.get("response") or {}
                    if not isinstance(request, dict) or not isinstance(response, dict):
                        raise ValueError(f"Invalid request/response for API: {name}")
                    if not request or not response:
                        logger.warning(f"[DomainAPINode] API '{name}' missing request/response fields.")

                    # Enforce REST naming style
                    if not re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+/.+", name, re.IGNORECASE):
                        name = f"POST /{name.replace(' ', '_').lower()}"

                    validated_apis.append(APIData(
                        name=name,
                        description=api.get("description", ""),
                        request=request,
                        response=response,
                    ))
                except Exception as ex:
                    logger.warning(f"[DomainAPINode] Skipping invalid API entry: {ex}")

            # --- Relationship Mapping ---
            rels = []
            if diagram_plan and "relations" in diagram_plan.get("class", {}):
                rels = diagram_plan["class"]["relations"]
            else:
                # Auto-generate simple relationships from entity cross-references
                for e in validated_entities:
                    for attr in e.attributes:
                        for other in validated_entities:
                            if e.name != other.name and other.name.lower() in attr.lower():
                                rels.append({"from": e.name, "to": other.name, "type": "association"})

            # --- Diagram Plan Generation ---
            class_plan = {
                "nodes": [{"name": e.name, "attributes": e.attributes} for e in validated_entities],
                "relations": rels,
            }
            diagram_plan = {"class": class_plan}

            # --- Log Insights ---
            logger.info(f"[DomainAPINode] Entities: {len(validated_entities)}")
            logger.info(f"[DomainAPINode] APIs: {len(validated_apis)}")
            logger.info(f"[DomainAPINode] Relationships: {len(rels)}")

            # --- Update State ---
            domain_data = DomainData(
                entities=validated_entities,
                apis=validated_apis,
                diagram_plan=diagram_plan
            )
            state.domain = domain_data
            state.set_status("domain_api_design", "completed", "Domain and API design completed")

            return state

        except Exception as e:
            logger.exception("[DomainAPINode] Domain and API design failed.")
            state.add_error(str(e))
            state.set_status("domain_api_design", "failed", str(e))
            return state
