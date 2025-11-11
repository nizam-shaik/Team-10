from __future__ import annotations
"""
Authentication and Integrations Agent
"""

# TODO: Implement AuthIntegrationsAgent class extending BaseAgent
# TODO: Analyze requirements text for authentication mechanisms
# TODO: Extract authentication actors (users, systems, roles)
# TODO: Identify authentication flows (OAuth, JWT, session-based, API key, etc.)
# TODO: List potential identity provider options (Google, AWS IAM, Auth0, Azure AD)
# TODO: Identify security threats relevant to authentication
# TODO: Extract external system integrations from requirements
# TODO: For each integration, extract: system name, purpose, protocol, auth method
# TODO: Collect integration endpoints and data contracts (inputs/outputs)
# TODO: Normalize authentication data into AuthenticationData object
# TODO: Normalize integrations list into IntegrationData objects
# TODO: Handle missing or empty authentication/integration fields gracefully
# TODO: Map alternative field names to standard schema (e.g., service->system)
# TODO: Validate data structure and ensure required fields are present
# TODO: Create comprehensive error messages for parsing failures
"""
Authentication and Integrations Agent
"""

import re
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from state.models import (
    HLDState,
    AuthenticationData,
    IntegrationData,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class AuthIntegrationsAgent(BaseAgent):
    """
    Analyze extracted requirements to identify authentication flows,
    actors, identity provider options, threats, and external integrations.

    Approach:
      - Uses LangChain's ChatGoogleGenerativeAI for LLM calls (automatic retry)
      - Leverages ChatPromptTemplate for structured prompts
      - Uses JsonOutputParser for automatic JSON parsing
      - Falls back to heuristics if LLM output is incomplete
      - Returns normalized AuthenticationData and IntegrationData objects
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are a security & integrations analyst. Given a product requirements "
            "document, extract ALL authentication details (actors, flows, identity provider "
            "options, threats, security features, compliance requirements) and ALL external integrations. "
            "IMPORTANT: Extract ALL authentication actors, ALL authentication flows/methods, ALL identity providers mentioned, "
            "ALL security threats, and ALL external system integrations. "
            "Return valid JSON with keys:\n"
            "{\n"
            '  "authentication": {\n'
            '    "actors": ["User","Admin","Compliance Officer"],\n'
            '    "flows": ["OAuth2","JWT","MFA","OTP","Session-based"],\n'
            '    "idp_options": ["Auth0","Okta","Azure AD","AWS Cognito"],\n'
            '    "threats": ["CSRF","Session Fixation","XSS","SQL Injection"],\n'
            '    "compliance": ["GDPR","CCPA","PCI-DSS","SOC2"]\n'
            "  },\n"
            '  "integrations": [\n'
            '    { "system": "Onfido", "purpose":"identity verification", "protocol":"REST", "auth":"API Key", "endpoints":["/verify"], "data_contract":{"inputs":["id_image"],"outputs":["score"]} },\n'
            '    { "system": "Notification Service", "purpose":"SMS/Email OTP", "protocol":"REST", "auth":"API Key", "endpoints":["/send"], "data_contract":{"inputs":["phone","message"],"outputs":["status"]} }\n'
            "  ]\n"
            "}\n"
            "CRITICAL: Do not return empty arrays. Extract comprehensive authentication and integration information. "
            "If authentication details are not explicitly stated, infer common security practices for similar systems. "
            "If integrations are mentioned by name only, infer reasonable protocol and auth methods."
        )

    def process(self, state: HLDState) -> Dict[str, Any]:
        """
        Analyze the HLDState to populate authentication and integrations fields.
        Uses LangChain for LLM invocation with automatic retries and JSON parsing.
        Returns a dict containing normalized structures or an error entry on failure.
        """
        state.set_status("auth_integrations", "processing", "Analyzing authentication & integrations")
        try:
            text = ""
            if state.extracted and getattr(state.extracted, "markdown", None):
                text = state.extracted.markdown
            else:
                # Fallback to minimal text reference if extraction not present
                text = f"(no extracted markdown) {state.pdf_path or ''}"

            # Increased text limit to ensure full context
            prompt = f"Requirements Text:\n{text[:50000]}"

            # ✅ Use LangChain's call_llm (automatic retry, error handling)
            llm_resp = self.call_llm(prompt, parse_json=True)
            self.log_cost(llm_resp)

            parsed = llm_resp.parsed_json or {}
            auth_block = parsed.get("authentication") or {}
            integrations_block = parsed.get("integrations") or []

            # If the LLM returned nothing usable, fall back to heuristics
            if not auth_block and not integrations_block:
                logger.debug("LLM returned no structured auth/integrations; running heuristics.")
                auth_block = self._heuristic_extract_auth(text)
                integrations_raw = self._heuristic_extract_integrations(text)
            else:
                integrations_raw = integrations_block

            # Normalize authentication block
            auth_obj = AuthenticationData(
                actors=self._coerce_list(auth_block.get("actors") if isinstance(auth_block, dict) else auth_block),
                flows=self._coerce_list(auth_block.get("flows") if isinstance(auth_block, dict) else auth_block),
                idp_options=self._coerce_list(auth_block.get("idp_options") if isinstance(auth_block, dict) else auth_block),
                threats=self._coerce_list(auth_block.get("threats") if isinstance(auth_block, dict) else auth_block),
            )

            # Normalize integrations into IntegrationData objects
            integrations: List[IntegrationData] = []
            if isinstance(integrations_raw, list):
                for item in integrations_raw:
                    try:
                        if isinstance(item, dict):
                            integrations.append(self._integration_from_dict(item))
                        elif isinstance(item, str):
                            integrations.append(self._integration_from_line(item))
                        else:
                            # best-effort stringify
                            integrations.append(self._integration_from_line(str(item)))
                    except Exception as e:
                        logger.debug(f"Skipping malformed integration entry: {item} — {e}")
                        continue

            # If still empty, try heuristic object conversion
            if not integrations:
                integrations = self._heuristic_extract_integrations_as_objects(text)

            # Assign to state
            state.authentication = auth_obj
            state.integrations = integrations
            state.set_status("auth_integrations", "completed", "Authentication & integrations analysis completed")
            logger.info("[AuthIntegrationsAgent] Completed analysis using LangChain.")
            return {"authentication": auth_obj.dict(), "integrations": [i.dict() for i in integrations]}

        except Exception as exc:
            logger.exception("AuthIntegrationsAgent failed")
            state.add_error(str(exc))
            state.set_status("auth_integrations", "failed", str(exc))
            return {"error": str(exc)}

    # -----------------------
    # Normalization helpers
    # -----------------------
    def _coerce_list(self, value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self.normalize_string(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, str):
            # split common separators
            parts = [p.strip() for p in re.split(r"[,\n;|/]+", value) if p.strip()]
            return [self.normalize_string(p) for p in parts]
        # final fallback: single stringified value
        return [self.normalize_string(str(value))]

    def _integration_from_dict(self, d: Dict[str, Any]) -> IntegrationData:
        system = self.normalize_string(d.get("system") or d.get("name") or "Unknown")
        purpose = self.normalize_string(d.get("purpose") or "")
        protocol = self.normalize_string(d.get("protocol") or d.get("proto") or "")
        authm = self.normalize_string(d.get("auth") or d.get("authentication") or "")
        endpoints_raw = d.get("endpoints") or []
        endpoints = [self.normalize_string(e) for e in endpoints_raw] if isinstance(endpoints_raw, list) else []
        data_contract = {}
        if isinstance(d.get("data_contract"), dict):
            for k, v in d.get("data_contract").items():
                if isinstance(v, list):
                    data_contract[self.normalize_string(k)] = [self.normalize_string(x) for x in v]
                else:
                    data_contract[self.normalize_string(k)] = [self.normalize_string(v)]
        return IntegrationData(
            system=system,
            purpose=purpose,
            protocol=protocol,
            auth=authm,
            endpoints=endpoints,
            data_contract=data_contract,
        )

    def _integration_from_line(self, line: str) -> IntegrationData:
        # Parse patterns like:
        # "Onfido - identity verification - REST - API Key"
        parts = [p.strip() for p in re.split(r"[-–—|:;]", line) if p.strip()]
        name = parts[0] if parts else "Unknown"
        purpose = ""
        protocol = ""
        auth = ""
        endpoints: List[str] = []

        for p in parts[1:]:
            if re.search(r"\bREST\b|\bHTTP\b|\bHTTPS\b|\bAPI\b", p, flags=re.I):
                protocol = p
            elif re.search(r"\bKey\b|\bOAuth\b|\bJWT\b|\bSAML\b|\bAPI Key\b|\bAuth\b", p, flags=re.I):
                auth = p
            else:
                purpose = f"{purpose}; {p}" if purpose else p

        return IntegrationData(
            system=self.normalize_string(name),
            purpose=self.normalize_string(purpose),
            protocol=self.normalize_string(protocol),
            auth=self.normalize_string(auth),
            endpoints=[self.normalize_string(e) for e in endpoints],
            data_contract={},
        )

    # -----------------------
    # Heuristic extractors
    # -----------------------
    def _heuristic_extract_auth(self, text: str) -> Dict[str, List[str]]:
        actors = set()
        flows = set()
        idps = set()
        threats = set()

        # Actors
        for m in re.finditer(r"\b(customer|user|admin|operator|compliance officer|support|merchant)\b", text, flags=re.I):
            actors.add(m.group(1).title())

        # Flows / mechanisms
        for mech in ("OAuth2", "OAuth", "JWT", "SAML", "API Key", "MFA", "OTP", "Session"):
            if re.search(rf"\b{re.escape(mech)}\b", text, flags=re.I):
                flows.add(mech)

        # IDP mentions
        for provider in ("Auth0", "Okta", "Azure AD", "AWS Cognito", "Google", "Aadhaar", "Onfido"):
            if re.search(rf"\b{re.escape(provider)}\b", text, flags=re.I):
                idps.add(provider)

        # Threats
        for t in ("CSRF", "XSS", "session fixation", "replay attack", "credential stuffing", "fraud"):
            if re.search(rf"\b{re.escape(t)}\b", text, flags=re.I):
                threats.add(t)

        # MFA patterns
        if re.search(r"\b(MFA|multi-?factor|OTP|one[- ]time pass)\b", text, flags=re.I):
            flows.add("MFA/OTP")

        return {
            "actors": sorted(actors),
            "flows": sorted(flows),
            "idp_options": sorted(idps),
            "threats": sorted(threats),
        }

    def _heuristic_extract_integrations(self, text: str) -> List[str]:
        integrations = []

        # Find an "Integrations" section
        m = re.search(r"(?im)^#{0,2}\s*Integrations\s*[:\n]\s*(.*?)(?:\n#{1,3}\s|\Z)", text, flags=re.S)
        if m:
            block = m.group(1)
            for ln in [l.strip(" -•\t") for l in block.splitlines()]:
                if ln:
                    integrations.append(ln)
            if integrations:
                return integrations

        # Search for common service phrases
        for ln in re.findall(r"^.*(Identity Verification Service|Document OCR|Notification Service|E-?Signature|KYC provider|Onfido|Trulioo).*", text, flags=re.I | re.M):
            integrations.append(ln.strip())

        # Comma-separated inline line
        m2 = re.search(r"(?i)Integrations\s*[:\-]\s*(.+)", text)
        if m2:
            parts = [p.strip() for p in re.split(r"[;,]", m2.group(1)) if p.strip()]
            integrations.extend(parts)

        # Deduplicate preserving order
        return list(dict.fromkeys(integrations))

    def _heuristic_extract_integrations_as_objects(self, text: str) -> List[IntegrationData]:
        raw = self._heuristic_extract_integrations(text)
        objs: List[IntegrationData] = []
        for line in raw:
            name = line
            purpose = ""
            proto = ""
            auth = ""
            endpoints: List[str] = []
            m = re.match(r"^(?P<title>[^\(]+)\((?P<inside>.+)\)$", line)
            if m:
                name = m.group("title").strip()
                inside = m.group("inside")
                vendors = [v.strip() for v in re.split(r"[;,]", inside) if v.strip()]
                if vendors:
                    purpose = f"Providers: {', '.join(vendors[:3])}"
            if re.search(r"\bOCR\b|\bDocument OCR\b", line, flags=re.I):
                purpose = purpose or "Document OCR & validation"
                proto = proto or "REST"
            if re.search(r"\bNotification Service\b|\bSMS\b|\bEmail OTP\b", line, flags=re.I):
                purpose = purpose or "Notification (SMS/Email)"
                proto = proto or "REST"
            if re.search(r"\bE-?Signature\b|\bEsignature\b", line, flags=re.I):
                purpose = purpose or "E-Signature Provider"
                proto = proto or "REST"

            objs.append(
                IntegrationData(
                    system=self.normalize_string(name) or "Unknown",
                    purpose=self.normalize_string(purpose),
                    protocol=self.normalize_string(proto),
                    auth=self.normalize_string(auth),
                    endpoints=[self.normalize_string(e) for e in endpoints],
                    data_contract={},
                )
            )
        return objs
