from __future__ import annotations
"""
Behavior and Quality Agent
"""

# TODO: Implement BehaviorQualityAgent class extending BaseAgent
# TODO: Extract user stories and use cases from requirements
# TODO: Define non-functional requirements (NFRs) by categories
# TODO: Implement NFR categories: security, reliability, performance, operability
# TODO: Identify and assess risks with likelihood and impact scoring
# TODO: Extract assumptions underlying the requirements
# TODO: Define risk mitigation strategies for identified risks
# TODO: Create sequence diagram plans from use cases and workflows
# TODO: Define actors and message flows for sequence diagrams
# TODO: Normalize use cases into simple string format
# TODO: Build NFRs dictionary with category keys and requirement lists
# TODO: Normalize risks into RiskData objects with impact/likelihood (1-5 scale)
# TODO: Create diagram plan with sequences containing actors and steps
# TODO: Validate risk scores are in valid range (1-5)
# TODO: Ensure diagram sequences have proper actor and step definitions
# TODO: Handle missing or incomplete risk information gracefully
"""
Behavior and Quality Agent
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from state.models import HLDState, BehaviorData, RiskData, ProcessingStatus

logger = logging.getLogger(__name__)


class BehaviorQualityAgent(BaseAgent):
    """
    Generate use cases, NFRs, risks, and sequence diagram plans from requirements.
    - Extracts user stories / use cases
    - Categorizes NFRs into security, reliability, performance, operability
    - Identifies risks and scores them (impact, likelihood) on 1-5 scale
    - Builds sequence diagram plans (actors + steps) suitable for diagram_converter
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior behavior and quality engineer with extensive experience in enterprise systems. "
            "Given product requirements, perform a comprehensive analysis to extract:"
            "\n1. ALL user stories/use-cases (both explicit and implicit)"
            "\n   - Start with 'As a [role]' format"
            "\n   - Include primary flows and alternative paths"
            "\n   - Cover both functional and business requirements"
            "\n2. Detailed non-functional requirements across categories:"
            "\n   Security: Authentication, authorization, encryption, compliance, audit"
            "\n   Reliability: Availability, fault tolerance, backup/restore, data integrity"
            "\n   Performance: Response times, throughput, scalability, resource utilization"
            "\n   Operability: Monitoring, logging, deployment, maintenance, support"
            "\n3. Comprehensive risk assessment:"
            "\n   - Technical risks (integration, scalability, etc.)"
            "\n   - Business risks (user adoption, compliance, etc.)"
            "\n   - Operational risks (availability, data quality, etc.)"
            "\n   - Score each risk (impact/likelihood 1-5)"
            "\n   - Provide detailed mitigation strategies"
            "Return JSON with keys:\n"
            "{\n"
            ' "use_cases": ["As a user, I want ...", "As a customer, I want ..."],\n'
            ' "nfrs": {"security": ["..."], "performance": ["..."], "reliability": ["..."], "operability": ["..."]},\n'
            ' "risks": [ {"id":"R1","desc":"...","assumption":"...","mitigation":"...","impact":3,"likelihood":2} ]\n'
            "}\n"
            "IMPORTANT: Do NOT generate diagram_plan or sequences - those will be generated automatically from use cases.\n"
            "CRITICAL: Do not return empty arrays or objects. Extract comprehensive, detailed information. "
            "If use cases are not explicitly stated, infer them from features and requirements. "
            "If NFRs are not explicitly stated, infer them from performance, security, and operational requirements mentioned. "
            "If risks are not explicitly stated, identify common risks for similar systems."
        )

    def process(self, state: HLDState) -> Dict[str, Any]:
        """
        Analyze requirements text and populate behavior-related state:
          - state.behavior : BehaviorData
          - update processing status
        Returns: dict(serializable) or {'error': msg}
        """
        state.set_status("behavior_quality", "processing", "Analyzing behavior and quality")
        try:
            text = ""
            if state.extracted and getattr(state.extracted, "markdown", None):
                text = self.normalize_string(state.extracted.markdown)
            else:
                text = f"(no extracted markdown) {state.pdf_path or ''}"

            # Ask LLM for structured behavior output (prefer)
            # Increased text limit to ensure full context
            prompt = f"Requirements Text:\n{text[:50000]}"
            
            # ✅ Use LangChain's call_llm with automatic retry and JSON parsing
            llm_resp = self.call_llm(prompt, parse_json=True)
            self.log_cost(llm_resp)

            parsed = llm_resp.parsed_json or {}
            use_cases_raw = parsed.get("use_cases") or []
            nfrs_raw = parsed.get("nfrs") or {}
            risks_raw = parsed.get("risks") or []

            # If LLM outputs are empty, fallback to heuristics
            logger.info(f"[BehaviorQualityAgent] LLM returned {len(use_cases_raw)} use cases, {len(list(nfrs_raw.values()) if isinstance(nfrs_raw, dict) else nfrs_raw)} NFRs, {len(risks_raw)} risks")
            
            if not use_cases_raw or len(use_cases_raw) < 2:
                logger.info("[BehaviorQualityAgent] Insufficient use cases from LLM, using heuristics")
                use_cases_raw = self._heuristic_extract_use_cases(text)
                logger.info(f"[BehaviorQualityAgent] Heuristics extracted {len(use_cases_raw)} use cases")
                
            if not nfrs_raw or (isinstance(nfrs_raw, dict) and not any(nfrs_raw.values())):
                logger.info("[BehaviorQualityAgent] Insufficient NFRs from LLM, using heuristics")
                nfrs_raw = self._heuristic_extract_nfrs(text)
                logger.info(f"[BehaviorQualityAgent] Heuristics extracted NFRs: {[(k, len(v)) for k, v in nfrs_raw.items()]}")
                
            if not risks_raw or len(risks_raw) < 2:
                logger.info("[BehaviorQualityAgent] Insufficient risks from LLM, using heuristics")
                risks_raw = self._heuristic_extract_risks(text)
                logger.info(f"[BehaviorQualityAgent] Heuristics extracted {len(risks_raw)} risks")

            # Normalize use cases (list of strings)
            use_cases = [self.normalize_string(uc) for uc in use_cases_raw if isinstance(uc, str) and uc.strip()]
            # Fallback: if still empty, try to build from lines that start with "As a"
            if not use_cases:
                use_cases = self._heuristic_extract_use_cases(text)

            # Normalize NFRs into specific categories
            nfrs = {
                "security": [],
                "reliability": [],
                "performance": [],
                "operability": []
            }
            # Accept either dict with categories or flat list
            if isinstance(nfrs_raw, dict):
                for k in nfrs.keys():
                    entries = nfrs_raw.get(k) or nfrs_raw.get(k.capitalize()) or []
                    if isinstance(entries, list):
                        nfrs[k] = [self.normalize_string(x) for x in entries if x and str(x).strip()]
            elif isinstance(nfrs_raw, list):
                # attempt to classify each item by keyword
                for item in nfrs_raw:
                    txt = self.normalize_string(item)
                    cat = self._classify_nfr(txt)
                    nfrs[cat].append(txt)

            # Normalize risks into RiskData objects and validate scores
            risks: List[RiskData] = []
            if isinstance(risks_raw, list):
                for r in risks_raw:
                    try:
                        if isinstance(r, dict):
                            rid = self.normalize_string(r.get("id") or r.get("risk_id") or "")
                            desc = self.normalize_string(r.get("desc") or r.get("description") or r.get("risk") or "")
                            assumption = self.normalize_string(r.get("assumption") or r.get("assumptions") or "")
                            mitigation = self.normalize_string(r.get("mitigation") or r.get("mitigation_strategy") or r.get("response") or "")
                            impact = self._coerce_score(r.get("impact", 3))
                            likelihood = self._coerce_score(r.get("likelihood", 3))
                            risks.append(RiskData(id=rid or "", desc=desc or "", assumption=assumption, mitigation=mitigation, impact=impact, likelihood=likelihood))
                        elif isinstance(r, str):
                            # create minimal RiskData from string
                            risks.append(RiskData(id="", desc=self.normalize_string(r), assumption="", mitigation="", impact=3, likelihood=3))
                    except Exception as e:
                        logger.debug("Skipping malformed risk entry: %s (%s)", r, e)
                        continue

            # If still no risks identified, try to heuristically identify from text (common risk phrases)
            if not risks:
                heur_risks = self._heuristic_extract_risks(text)
                for hr in heur_risks:
                    risks.append(RiskData(
                        id=hr.get("id", ""),
                        desc=hr.get("desc", ""),
                        assumption=hr.get("assumption", ""),
                        mitigation=hr.get("mitigation", ""),
                        impact=self._coerce_score(hr.get("impact", 3)),
                        likelihood=self._coerce_score(hr.get("likelihood", 3))
                    ))

            # Build sequence diagram plans from top use cases (up to 10, processes up to 15 for coverage)
            sequences = self._build_sequence_plans(use_cases)

            # Validate risk scores (ensure 1..5)
            for r in risks:
                r.impact = self._coerce_score(r.impact)
                r.likelihood = self._coerce_score(r.likelihood)

            # Create BehaviorData object with improved sequences
            behavior = BehaviorData(
                use_cases=use_cases,
                nfrs=nfrs,
                risks=risks,
                diagram_plan={"sequences": sequences}
            )

            state.behavior = behavior
            state.set_status("behavior_quality", "completed", "Behavior & quality analysis completed")
            logger.info("[BehaviorQualityAgent] Completed behavior analysis.")
            
            # Return structure expected by BehaviorQualityNode
            return {
                "use_cases": use_cases,
                "nfrs": nfrs,
                "risks": [r.dict() for r in risks],
                "diagram_plan": {"sequences": sequences}
            }

        except Exception as exc:
            logger.exception("BehaviorQualityAgent failed")
            state.add_error(str(exc))
            state.set_status("behavior_quality", "failed", str(exc))
            return {"error": str(exc)}

    # -----------------------
    # Heuristics & Helpers
    # -----------------------
    def _heuristic_extract_use_cases(self, text: str) -> List[str]:
        """
        Extract lines starting with 'As a' or bulleted user stories.
        """
        out: List[str] = []
        
        # Pattern 1: Direct "As a" patterns (most common)
        for m in re.finditer(r"(?im)^\s*(?:-|\*|\•|–|>|\d+\.|\#{1,6})?\s*(As\s+a\s+[^\n]{10,200})", text):
            uc = m.group(1).strip()
            if uc and len(uc) > 15:  # Ensure it's substantial
                out.append(uc)
        
        # Pattern 2: Look specifically in "User Stories" section
        user_stories_match = re.search(r"(?im)##?\s*(?:\d+\.?)?\s*User\s+Stories\s*\n(.*?)(?:\n##|\Z)", text, re.DOTALL)
        if user_stories_match:
            stories_section = user_stories_match.group(1)
            # Extract bullet points in this section
            for line in stories_section.split('\n'):
                line = line.strip()
                if line.startswith(('*', '-', '•', '–', '>')):
                    story = line.lstrip('*-•–> \t')
                    if 'as a' in story.lower() and len(story) > 15:
                        out.append(story)
        
        # Pattern 3: "I want to" patterns (implicit user stories)
        for m in re.finditer(r"(?im)(?:customer|user|applicant|officer|admin)\s+(?:want[s]?|need[s]?|can)\s+to\s+([^\n]{10,150})", text):
            story = m.group(0).strip()
            if len(story) > 20:
                out.append(f"As a user, {story}")
        
        # fallback: look for feature headings and convert to user stories
        if len(out) < 3:
            for m in re.finditer(r"(?im)^#{1,3}\s*(?:\d+\.?\s*)?([A-Z][A-Za-z0-9 &\-/]+(?:Feature|Capability|Function|Story))\s*$", text):
                title = m.group(1).strip()
                out.append(f"As a user, I want to {title.lower()}.")
        
        # dedupe preserving order
        seen = set()
        result = []
        for item in out:
            normalized = self.normalize_string(item)
            if normalized and normalized not in seen and len(normalized) > 15:
                seen.add(normalized)
                result.append(normalized)
        
        return result

    def _heuristic_extract_nfrs(self, text: str) -> Dict[str, List[str]]:
        """
        Look for NFR headings and categorize content under them.
        Enhanced to find NFRs in structured sections.
        """
        categories = {"security": [], "reliability": [], "performance": [], "operability": []}
        
        # Pattern 1: Look for "Non-Functional Requirements" section
        nfr_section_match = re.search(r"(?im)##?\s*(?:\d+\.?)?\s*Non[- ]?Functional\s+Requirements.*?\n(.*?)(?:\n##|\Z)", text, re.DOTALL)
        if nfr_section_match:
            nfr_block = nfr_section_match.group(1)
            # Extract bullet points
            for line in nfr_block.split('\n'):
                line = line.strip()
                if line.startswith(('*', '-', '•', '–')):
                    nfr = line.lstrip('*-•– \t').strip()
                    if nfr and len(nfr) > 10:
                        cat = self._classify_nfr(nfr)
                        categories[cat].append(self.normalize_string(nfr))
        
        # Pattern 2: Find category-specific sections
        for cat in categories.keys():
            # Look for dedicated sections like "## Security" or "### Performance"
            cat_pattern = rf"(?im)^#{{{1,4}}}\s*{cat}\b.*?\n(.*?)(?:\n#{{1,4}}|\Z)"
            m = re.search(cat_pattern, text, flags=re.DOTALL)
            if m:
                block = m.group(1)
                lines = [ln.strip(" -•–\t*") for ln in block.splitlines() if ln.strip()]
                for line in lines:
                    if line and len(line) > 10 and not line.startswith('#'):
                        categories[cat].append(self.normalize_string(line))
        
        # Pattern 3: Extract from common NFR keyword patterns
        nfr_patterns = {
            "performance": [
                r"(?i)(?:response|latency|throughput|processing)\s+time[:\s]+([^\n\.]{10,100})",
                r"(?i)(?:performance|speed)[:\s]+([^\n\.]{10,100})",
                r"(?i)(?:≤|<=|less than|within)\s+\d+\s+(?:second|minute|ms|millisecond)",
            ],
            "security": [
                r"(?i)(?:authentication|encryption|security)[:\s]+([^\n\.]{10,100})",
                r"(?i)(?:OAuth|JWT|TLS|AES|MFA|2FA)[^\n\.]{0,80}",
                r"(?i)(?:access control|authorization|secure)[^\n\.]{10,80}",
            ],
            "reliability": [
                r"(?i)(?:availability|uptime)[:\s]+([^\n\.]{10,100})",
                r"(?i)\d+\.?\d*%\s+(?:uptime|availability|reliable)",
                r"(?i)(?:fault.?toleran|redundan|backup|disaster recovery)[^\n\.]{10,80}",
            ],
            "operability": [
                r"(?i)(?:monitoring|logging|observability|alert)[^\n\.]{10,80}",
                r"(?i)(?:deployment|devops|cicd|maintenance)[^\n\.]{10,80}",
            ]
        }
        
        for cat, patterns in nfr_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, str) and len(match) > 10:
                        categories[cat].append(self.normalize_string(match))
        
        # Remove duplicates while preserving order
        for cat in categories:
            seen = set()
            unique = []
            for item in categories[cat]:
                if item and item not in seen:
                    seen.add(item)
                    unique.append(item)
            categories[cat] = unique[:10]  # Limit to 10 per category
        
        return categories

    def _heuristic_extract_risks(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify risks from risk tables and risk-related sentences.
        Enhanced to parse markdown tables and risk sections.
        """
        found: List[Dict[str, Any]] = []
        
        # Pattern 1: Find "Risks & Mitigations" section with tables
        risks_section_match = re.search(r"(?im)##?\s*(?:\d+\.?)?\s*Risks?\s*(?:&|and)?\s*Mitigations?.*?\n(.*?)(?:\n##|\Z)", text, re.DOTALL)
        if risks_section_match:
            risks_block = risks_section_match.group(1)
            
            # Try to parse markdown table
            table_rows = []
            for line in risks_block.split('\n'):
                # Skip separator lines (---|---|---)
                if re.match(r'^\s*\|?\s*[-:]+\s*\|', line):
                    continue
                # Match table rows
                if '|' in line:
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if len(cells) >= 2 and cells[0].lower() not in ['risk', 'risks', '']:
                        table_rows.append(cells)
            
            # Parse table rows (usually: Risk | Impact | Mitigation)
            for idx, row in enumerate(table_rows):
                if len(row) >= 2:
                    risk_desc = row[0].strip()
                    mitigation = row[2].strip() if len(row) > 2 else "Mitigation not specified"
                    impact_text = row[1].strip() if len(row) > 1 else ""
                    
                    # Try to extract impact score from text
                    impact = 3
                    if any(word in impact_text.lower() for word in ['high', 'critical', 'severe']):
                        impact = 4
                    elif any(word in impact_text.lower() for word in ['low', 'minor']):
                        impact = 2
                    
                    found.append({
                        "id": f"RISK-{idx+1:02d}",
                        "desc": risk_desc,
                        "assumption": "",
                        "mitigation": mitigation,
                        "impact": impact,
                        "likelihood": 3
                    })
        
        # Pattern 2: Find risk sentences in text
        for s in re.split(r'(?<=[.!?])\s+', text):
            if re.search(r'\b(risk|threat|failure|vulnerability|weakness|concern)\b', s, flags=re.I):
                desc = self.normalize_string(s)
                if len(desc) > 20:
                    # Determine impact/likelihood from keywords
                    impact = 4 if re.search(r'\b(financial|critical|high.?impact|severe|loss|breach)\b', s, flags=re.I) else 3
                    likelihood = 3 if re.search(r'\b(common|likely|possible|may|could)\b', s, flags=re.I) else 2
                    
                    # Extract mitigation if mentioned
                    mitigation = "Mitigation strategy needed"
                    m = re.search(r'(?i)(mitigat(?:e|ion)|prevent|address|control|manage|monitor)[:\s-]*(.+?)(?:\.|$)', s)
                    if m and len(m.group(2)) > 10:
                        mitigation = m.group(2).strip()
                    
                    found.append({
                        "id": f"RISK-{len(found)+1:02d}",
                        "desc": desc[:200],
                        "assumption": "",
                        "mitigation": mitigation[:200],
                        "impact": impact,
                        "likelihood": likelihood
                    })
        
        # Pattern 3: Look for specific risk keywords and create risks
        risk_keywords = {
            "downtime": ("System downtime or unavailability", "Implement monitoring and redundancy"),
            "breach": ("Data breach or security incident", "Apply encryption and access controls"),
            "compliance": ("Regulatory non-compliance", "Regular audits and compliance checks"),
            "performance": ("Performance degradation under load", "Load testing and scaling strategy"),
            "bias": ("Model bias in scoring or decisions", "Regular bias audits and fairness metrics"),
        }
        
        for keyword, (desc_template, mitigation_template) in risk_keywords.items():
            if re.search(rf'\b{keyword}\b', text, flags=re.I):
                # Avoid duplicates
                if not any(keyword in r['desc'].lower() for r in found):
                    found.append({
                        "id": f"RISK-{len(found)+1:02d}",
                        "desc": desc_template,
                        "assumption": f"Risk identified from {keyword} references in requirements",
                        "mitigation": mitigation_template,
                        "impact": 3,
                        "likelihood": 3
                    })
        
        # Remove duplicates based on description similarity
        unique_risks = []
        seen_descs = set()
        for risk in found:
            desc_key = risk['desc'][:50].lower()
            if desc_key not in seen_descs:
                seen_descs.add(desc_key)
                unique_risks.append(risk)
        
        return unique_risks[:10]  # Limit to 10 most important risks

    def _classify_nfr(self, text: str) -> str:
        """
        Simple keyword-based NFR classification.
        """
        t = text.lower()
        if any(k in t for k in ["encrypt", "auth", "mfa", "oauth", "jwt", "privacy", "gdpr", "compliance", "secure"]):
            return "security"
        if any(k in t for k in ["uptime", "availability", "redund", "failover", "reliab", "backup"]):
            return "reliability"
        if any(k in t for k in ["latency", "response", "performance", "throughput", "concurrent", "scale"]):
            return "performance"
        return "operability"

    def _coerce_score(self, value: Optional[Any]) -> int:
        """
        Ensure a score is an int in 1..5 (default 3).
        Accepts numeric or common textual values.
        """
        try:
            if value is None:
                return 3
            if isinstance(value, (int, float)):
                v = int(round(value))
                return max(1, min(5, v))
            s = str(value).strip().lower()
            if s.isdigit():
                v = int(s)
                return max(1, min(5, v))
            # map common words
            mapping = {"low": 2, "medium": 3, "med": 3, "high": 4, "very high": 5, "critical": 5}
            for k, vv in mapping.items():
                if k in s:
                    return vv
        except Exception:
            pass
        return 3

    def _build_sequence_plans(self, use_cases: List[str]) -> List[Dict[str, Any]]:
        """
        Create simple sequence diagram plans from use cases. Each sequence:
          { "title": str, "actors": [], "steps": [ {"from":"User","to":"System","message":"..."} ] }
        Produces up to 15 use cases (to ensure we get at least 10 valid sequences).
        
        Generates proper action-based sequence flows instead of full user story text.
        NOTE: We leave actors list empty - Mermaid will auto-detect from steps.
        """
        sequences: List[Dict[str, Any]] = []
        for idx, uc in enumerate(use_cases[:15]):  # Process more use cases to get better coverage
            # Extract actor from "As a [actor]" pattern
            actor_match = re.search(r"(?i)as\s+an?\s+([^,]+?)(?:\s+I\s+want|,)", uc)
            primary_actor = actor_match.group(1).strip().title() if actor_match else "User"
            
            # Extract the action/goal from "I want to [action]" pattern
            action_match = re.search(r"(?i)I\s+want\s+to\s+([^,\.]+?)(?:\s+so\s+that|\.|\,|$)", uc)
            if not action_match:
                # Alternative patterns: "wants to", "needs to", "can"
                action_match = re.search(r"(?i)(?:wants?|needs?|can)\s+to\s+([^,\.]+?)(?:\s+so\s+that|\.|\,|$)", uc)
            
            if action_match:
                action = action_match.group(1).strip()
                
                # Generate meaningful sequence steps from the action
                steps = self._generate_sequence_steps(primary_actor, action)
                
                # Create title from the action (keep it concise)
                title = action[:60] if len(action) < 60 else action[:57] + "..."
                
                # DON'T collect actors - let diagram_converter infer from steps
                # This prevents the issue of declaring unused actors
                sequences.append({
                    "title": title,
                    "actors": [],  # Empty - actors will be inferred from steps
                    "steps": steps
                })
        
        # If no valid sequences generated, return empty list (validation will catch this)
        # The node layer should fail with proper error message instead of showing dummy sequence
        
        return sequences[:10]  # Limit to 10 sequences (best quality ones)
    
    def _generate_sequence_steps(self, actor: str, action: str) -> List[Dict[str, str]]:
        """
        Generate meaningful sequence diagram steps from an action description.
        Converts natural language actions into proper sequence flow with multiple actors.
        """
        steps = []
        action_lower = action.lower()
        
        # Common patterns for different types of actions
        if any(word in action_lower for word in ["sign up", "register", "create account", "onboard"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "submits registration"},
                {"from": "ApplicationModule", "to": "AuthenticationService", "message": "creates user account"},
                {"from": "AuthenticationService", "to": "Database", "message": "stores user data"},
                {"from": "Database", "to": "AuthenticationService", "message": "confirms registration"},
                {"from": "AuthenticationService", "to": actor, "message": "sends confirmation"}
            ]
        elif any(word in action_lower for word in ["upload", "submit document", "provide document"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "uploads document"},
                {"from": "ApplicationModule", "to": "StorageService", "message": "stores document"},
                {"from": "StorageService", "to": "ValidationService", "message": "validates document"},
                {"from": "ValidationService", "to": "ApplicationModule", "message": "returns validation result"},
                {"from": "ApplicationModule", "to": actor, "message": "displays confirmation"}
            ]
        elif any(word in action_lower for word in ["verify", "validate", "check"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "requests verification"},
                {"from": "ApplicationModule", "to": "VerificationService", "message": "initiates verification"},
                {"from": "VerificationService", "to": "ExternalAPI", "message": "validates identity"},
                {"from": "ExternalAPI", "to": "VerificationService", "message": "returns result"},
                {"from": "VerificationService", "to": actor, "message": "confirms verification"}
            ]
        elif any(word in action_lower for word in ["login", "authenticate", "sign in"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "enters credentials"},
                {"from": "ApplicationModule", "to": "AuthenticationService", "message": "validates credentials"},
                {"from": "AuthenticationService", "to": "Database", "message": "retrieves user data"},
                {"from": "Database", "to": "AuthenticationService", "message": "returns user profile"},
                {"from": "AuthenticationService", "to": actor, "message": "grants access token"}
            ]
        elif any(word in action_lower for word in ["view", "see", "check status", "monitor"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "requests information"},
                {"from": "ApplicationModule", "to": "Database", "message": "queries data"},
                {"from": "Database", "to": "ApplicationModule", "message": "returns results"},
                {"from": "ApplicationModule", "to": actor, "message": "displays information"}
            ]
        elif any(word in action_lower for word in ["approve", "review", "assess"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "reviews request"},
                {"from": "ApplicationModule", "to": "WorkflowEngine", "message": "fetches details"},
                {"from": "WorkflowEngine", "to": "ApplicationModule", "message": "returns data"},
                {"from": "ApplicationModule", "to": actor, "message": "shows decision form"},
                {"from": actor, "to": "WorkflowEngine", "message": "submits decision"}
            ]
        elif any(word in action_lower for word in ["apply", "request", "submit application"]):
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": "submits application"},
                {"from": "ApplicationModule", "to": "CreditBureau", "message": "requests credit report"},
                {"from": "CreditBureau", "to": "ApplicationModule", "message": "returns report"},
                {"from": "ApplicationModule", "to": "RiskEngine", "message": "requests risk score"},
                {"from": "RiskEngine", "to": "ApplicationModule", "message": "returns score"},
                {"from": "ApplicationModule", "to": actor, "message": "sends confirmation"}
            ]
        else:
            # Generic fallback for unknown actions
            # Extract key verbs and create simple flow
            verb = action.split()[0] if action.split() else "process"
            steps = [
                {"from": actor, "to": "ApplicationModule", "message": f"{verb}s request"},
                {"from": "ApplicationModule", "to": "BackendService", "message": f"processes {action[:30]}"},
                {"from": "BackendService", "to": "Database", "message": "fetches/updates data"},
                {"from": "Database", "to": "BackendService", "message": "returns result"},
                {"from": "BackendService", "to": "ApplicationModule", "message": "returns response"},
                {"from": "ApplicationModule", "to": actor, "message": "displays result"}
            ]
        
        return steps
