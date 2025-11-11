from __future__ import annotations
"""
Authentication and Integrations Node - Analyzes security and external systems
"""

# TODO: Implement AuthIntegrationsNode extending BaseNode
# TODO: Import AuthIntegrationsAgent from agent module
# TODO: Implement __init__ to initialize with agent instance
# TODO: Implement execute(state) method:
#       - Create AuthIntegrationsAgent instance
#       - Call agent.process(state)
#       - Validate authentication and integration data
#       - Update state.authentication with AuthenticationData
#       - Update state.integrations with list of IntegrationData
#       - Update stage status to completed
#       - Return updated state
# TODO: Add data validation
#       - Validate authentication actors list is not empty
#       - Check flows, threats have meaningful content
#       - Validate integration systems are unique
#       - Ensure data_contract fields have inputs/outputs
# TODO: Implement error handling
#       - Handle cases where no auth/integration data found
#       - Log warnings for incomplete data
#       - Provide sensible defaults if needed
# TODO: Consider data enrichment
#       - Categorize threats by severity
#       - Map auth flows to standard types
#       - Link integrations to entities if possible
# TODO: Log security insights
#       - Number of security threats identified
#       - Count of external integrations
#       - Authentication mechanisms found
"""
Authentication and Integrations Node - Analyzes security and external systems
"""

import logging
from typing import List

from .base_node import BaseNode
from agent.auth_agent import AuthIntegrationsAgent
from state.models import (
    HLDState,
    AuthenticationData,
    IntegrationData,
)

logger = logging.getLogger(__name__)


class AuthIntegrationsNode(BaseNode):
    """
    Node responsible for analyzing authentication mechanisms
    and external integrations defined in the requirements.
    """

    def __init__(self):
        super().__init__(name="auth_integrations", agent=AuthIntegrationsAgent())

    def execute(self, state: HLDState) -> HLDState:
        """
        Run authentication and integration analysis using AuthIntegrationsAgent.
        Validates and updates the state with structured data.
        """
        logger.info("[AuthIntegrationsNode] Starting authentication and integrations analysis.")
        state.set_status("auth_integrations", "processing", "Analyzing authentication and integrations")

        try:
            # Run agent to analyze the extracted requirements content
            result = self.agent.process(state)
            if not isinstance(result, dict):
                raise ValueError("Agent did not return a valid dictionary result")

            auth_data = result.get("authentication") or {}
            integrations_data = result.get("integrations") or []

            # --- Data Validation & Normalization ---
            # Validate authentication structure
            actors = [a.strip() for a in (auth_data.get("actors") or []) if a.strip()]
            flows = [f.strip() for f in (auth_data.get("flows") or []) if f.strip()]
            threats = [t.strip() for t in (auth_data.get("threats") or []) if t.strip()]
            idp_options = [i.strip() for i in (auth_data.get("idp_options") or []) if i.strip()]

            # Don't add dummy defaults - the agent should extract from requirements
            if not actors:
                logger.error("[AuthIntegrationsNode] No authentication actors found even after agent processing.")
                raise ValueError("No authentication actors could be extracted from requirements.")

            if not flows:
                logger.error("[AuthIntegrationsNode] No authentication flows found even after agent processing.")
                raise ValueError("No authentication flows could be extracted from requirements.")

            # IDP options are truly optional - empty list is acceptable
            if not idp_options:
                logger.info("[AuthIntegrationsNode] No IDP options specified in requirements.")

            # Categorize threats by severity (basic heuristic)
            threat_levels = []
            for t in threats:
                lvl = "high" if any(k in t.lower() for k in ["csrf", "injection", "spoof", "bypass"]) else "medium"
                threat_levels.append(f"{t} ({lvl})")

            # Build AuthenticationData object
            auth_obj = AuthenticationData(
                actors=actors,
                flows=flows,
                idp_options=idp_options,
                threats=threat_levels,
            )

            # --- Integration Data ---
            validated_integrations: List[IntegrationData] = []
            seen_systems = set()
            for i in integrations_data:
                try:
                    system = str(i.get("system") or i.get("name") or "UnknownSystem").strip()
                    if system in seen_systems:
                        logger.warning(f"[AuthIntegrationsNode] Duplicate integration system skipped: {system}")
                        continue
                    seen_systems.add(system)

                    purpose = str(i.get("purpose") or "N/A").strip()
                    protocol = str(i.get("protocol") or "HTTPS").strip()
                    auth_method = str(i.get("auth") or "OAuth2.0").strip()
                    endpoints = [str(e).strip() for e in (i.get("endpoints") or []) if str(e).strip()]
                    data_contract = i.get("data_contract") or {"inputs": [], "outputs": []}

                    # Ensure data_contract has valid structure
                    if not isinstance(data_contract, dict):
                        data_contract = {"inputs": [], "outputs": []}
                    if "inputs" not in data_contract:
                        data_contract["inputs"] = []
                    if "outputs" not in data_contract:
                        data_contract["outputs"] = []

                    integ_obj = IntegrationData(
                        system=system,
                        purpose=purpose,
                        protocol=protocol,
                        auth=auth_method,
                        endpoints=endpoints,
                        data_contract=data_contract,
                    )
                    validated_integrations.append(integ_obj)

                except Exception as e:
                    logger.warning(f"[AuthIntegrationsNode] Skipped invalid integration entry: {e}")

            # Handle empty integration fallback
            if not validated_integrations:
                raise ValueError("[AuthIntegrationsNode] No integrations could be extracted from requirements.")

            # Update state
            state.authentication = auth_obj
            state.integrations = validated_integrations
            state.set_status("auth_integrations", "completed", "Authentication and integrations analysis completed")

            # --- Logging of Security Insights ---
            logger.info(f"[AuthIntegrationsNode] Actors identified: {len(actors)}")
            logger.info(f"[AuthIntegrationsNode] Authentication flows: {', '.join(flows)}")
            logger.info(f"[AuthIntegrationsNode] IDP options: {', '.join(idp_options)}")
            logger.info(f"[AuthIntegrationsNode] Threats: {len(threat_levels)} detected")
            logger.info(f"[AuthIntegrationsNode] External integrations: {len(validated_integrations)}")

            return state

        except Exception as e:
            logger.exception("AuthIntegrationsNode failed during execution.")
            state.add_error(f"AuthIntegrationsNode failed: {e}")
            state.set_status("auth_integrations", "failed", str(e))
            return state
