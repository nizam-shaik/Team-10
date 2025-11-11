from __future__ import annotations

"""
Domain and API Design Agent
"""

# TODO: Implement DomainAPIAgent class extending BaseAgent
# TODO: Analyze requirements for core business entities
# TODO: Extract entity attributes and relationships
# TODO: Design REST API endpoints based on requirements
# TODO: Define API request and response schemas
# TODO: Create database schema suggestions from entities
# TODO: Generate Mermaid-compatible class diagram plan
# TODO: Extract entity relationships in Mermaid format (A --> B : uses)
# TODO: Normalize domain entities into EntityData objects with name and attributes
# TODO: Normalize API specifications into APIData objects with request/response
# TODO: Build diagram plan with classes and relationships references
# TODO: Handle entity cardinality and relationship types
# TODO: Validate that entities have meaningful attributes (≥3)
# TODO: Ensure API specs have both request and response schemas
# TODO: Create DomainData object aggregating entities, APIs, and diagram plan
# TODO: Handle cases where requirements are incomplete or ambiguous
"""
Domain and API Design Agent
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from state.models import (
    HLDState,
    EntityData,
    APIData,
    DomainData,
    ProcessingStatus,
)

logger = logging.getLogger(__name__)


class DomainAPIAgent(BaseAgent):
    """
    Analyze requirements to produce:
      - Domain entities (name + attributes)
      - API specifications (endpoints, request/response schemas)
      - Database schema suggestions
      - Mermaid-compatible class diagram plan

    Strategy:
      1. Prefer structured output from the LLM (JSON with entities + apis)
      2. If no valid structured output, use heuristics to extract entities & attributes
      3. Validate and enrich entities (ensure meaningful attributes)
      4. Generate API specs using entities & use-cases
      5. Build a diagram_plan dict compatible with utils/diagram_converter.diagr am_plan_to_text
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior system architect assistant. Given product requirements text, "
            "identify ALL domain entities, their attributes, relationships, and propose comprehensive REST API "
            "endpoints with request/response schemas. "
            "CRITICALLY IMPORTANT: Analyze the entire text thoroughly and extract EVERY SINGLE entity mentioned, "
            "both explicitly stated and implicitly referenced. For each entity:"
            "\n- List at least 8-12 attributes that would be necessary in a real system"
            "\n- Include all standard fields (id, created_at, updated_at, status, etc.)"
            "\n- Add domain-specific fields based on the business context"
            "\n- Ensure proper data types and relationships are captured"
            "\nCreate comprehensive REST API endpoints covering:"
            "\n- Full CRUD operations for each entity"
            "\n- Business-specific operations and workflows"
            "\n- Batch operations where relevant"
            "\n- Search and filter endpoints"
            "\nReturn JSON with keys:\n"
            "{\n"
            '  "entities": [ { "name": "User", "attributes": ["id","email","created_at","mobile_number","status"] }, ... ],\n'
            '  "relationships": [ { "from":"User","to":"Account","type":"aggregation","label":"owns"} ],\n'
            '  "apis": [ { "name": "/api/accounts", "description":"create account", "request": {"email":"string","mobile":"string"}, "response": {"account_id":"string","status":"string"} } ]\n'
            "}\n"
            "CRITICAL: Do not return empty arrays. Extract as many entities and APIs as possible from the requirements. "
            "Be thorough and comprehensive. If uncertain about attributes, infer reasonable ones based on the entity name and context."
        )

    # -----------------------
    # Public processing entry
    # -----------------------
    def process(self, state: HLDState) -> Dict[str, Any]:
        state.set_status("domain_api_design", "processing", "Designing domain model and APIs")
        try:
            src_text = ""
            if state.extracted and getattr(state.extracted, "markdown", None):
                src_text = state.extracted.markdown
            else:
                src_text = f"(no extracted markdown available) {state.pdf_path or ''}"

            # Ask LLM for structured domain + api JSON (prefer)
            # Increased text limit to ensure full context
            prompt = f"Requirements Text:\n{src_text[:50000]}"
            
            # ✅ Use LangChain's call_llm with automatic retry and JSON parsing
            llm_resp = self.call_llm(prompt, parse_json=True)
            self.log_cost(llm_resp)

            parsed = llm_resp.parsed_json or {}

            entities_raw = parsed.get("entities") or []
            relationships_raw = parsed.get("relationships") or []
            apis_raw = parsed.get("apis") or []

            # ALWAYS use heuristics as a fallback or supplement
            logger.info(f"[DomainAPIAgent] LLM returned {len(entities_raw)} entities, {len(apis_raw)} apis")
            
            # If LLM didn't return structured results, use heuristics
            if not entities_raw or len(entities_raw) < 3:
                logger.info("LLM returned insufficient entities; using heuristics to supplement.")
                heuristic_entities = self._heuristic_extract_entities(src_text)
                # Merge with LLM results (avoid duplicates)
                existing_names = {e.get("name", "").lower() for e in entities_raw if isinstance(e, dict)}
                for he in heuristic_entities:
                    if he.get("name", "").lower() not in existing_names:
                        entities_raw.append(he)
                        
            if not apis_raw or len(apis_raw) < 3:
                logger.info("LLM returned insufficient APIs; using heuristics to supplement.")
                heuristic_apis = self._heuristic_design_apis(entities_raw, src_text)
                apis_raw.extend(heuristic_apis)
                
            if not relationships_raw:
                logger.info("No relationships from LLM; extracting via heuristics.")
                relationships_raw = self._heuristic_extract_relationships(src_text, entities_raw)

            # Normalize entities
            entities: List[EntityData] = []
            for e in entities_raw:
                try:
                    if isinstance(e, dict):
                        name = self.normalize_string(e.get("name") or e.get("entity") or "Entity")
                        attrs = []
                        raw_attrs = e.get("attributes") or e.get("fields") or []
                        if isinstance(raw_attrs, list):
                            attrs = [self.normalize_string(a) for a in raw_attrs if a and str(a).strip()]
                        elif isinstance(raw_attrs, str):
                            attrs = [self.normalize_string(x) for x in re.split(r"[,\n;|]+", raw_attrs) if x.strip()]
                        # Enforce minimum attribute count (attempt to synthesize if <8)
                        if len(attrs) < 8:
                            attrs = self._enrich_attributes(name, attrs)
                        entities.append(EntityData(name=name or "Entity", attributes=attrs))
                except Exception as ex:
                    logger.debug("Skipping malformed entity: %s (%s)", e, ex)
                    continue

            logger.info(f"[DomainAPIAgent] Processed {len(entities)} entities")
            
            # Validate entities list
            if not entities or len(entities) < 3:
                logger.warning(f"[DomainAPIAgent] Only {len(entities)} entities found, adding fallback entities")
                # fallback: ensure at least 5 entities
                fallback_names = ["Customer", "Account", "Transaction", "Profile", "Document"]
                existing_names = {e.name.lower() for e in entities}
                for fname in fallback_names:
                    if fname.lower() not in existing_names:
                        entities.append(EntityData(
                            name=fname, 
                            attributes=self._generate_entity_attributes(fname, src_text)
                        ))

            # Normalize APIs
            apis: List[APIData] = []
            for a in apis_raw:
                try:
                    if isinstance(a, dict):
                        name = self.normalize_string(a.get("name") or a.get("endpoint") or "/unnamed")
                        description = self.normalize_string(a.get("description") or a.get("purpose") or "")
                        request = {}
                        response = {}
                        if isinstance(a.get("request"), dict):
                            request = {self.normalize_string(k): self.normalize_string(v) for k, v in a.get("request").items()}
                        if isinstance(a.get("response"), dict):
                            response = {self.normalize_string(k): self.normalize_string(v) for k, v in a.get("response").items()}
                        # Ensure both request and response have something
                        request, response = self._ensure_request_response(request, response, entities)
                        apis.append(APIData(name=name, description=description, request=request, response=response))
                except Exception as ex:
                    logger.debug("Skipping malformed API spec: %s (%s)", a, ex)
                    continue

            logger.info(f"[DomainAPIAgent] Processed {len(apis)} APIs from LLM/heuristics")
            
            # If LLM didn't provide enough APIs, design them automatically
            if not apis or len(apis) < 5:
                logger.info(f"[DomainAPIAgent] Supplementing APIs (current: {len(apis)})")
                additional_apis = self._heuristic_design_apis(entities, src_text)
                # Avoid duplicates
                existing_api_names = {api.name.lower() for api in apis}
                for new_api in additional_apis:
                    if new_api.name.lower() not in existing_api_names:
                        apis.append(new_api)
                        
            logger.info(f"[DomainAPIAgent] Final API count: {len(apis)}")

            # Build diagram plan compatible with diagram_converter
            diagram_plan = self._build_diagram_plan(entities, relationships_raw)

            # Database suggestions
            db_suggestions = self._generate_db_suggestions(entities)

            domain = DomainData(
                entities=entities,
                apis=apis,
                diagram_plan=diagram_plan
            )

            # attach to state
            state.domain = domain
            # store db suggestions in meta for downstream (not formal model)
            if hasattr(state, "extracted") and state.extracted and hasattr(state.extracted, "meta"):
                state.extracted.meta["db_suggestions"] = db_suggestions
            state.set_status("domain_api_design", "completed", "Domain & API design completed")
            logger.info("[DomainAPIAgent] Domain design complete.")
            
            # Return structure expected by DomainAPINode
            return {
                "entities": [e.dict() for e in entities],
                "apis": [a.dict() for a in apis],
                "diagram_plan": diagram_plan,
                "db_suggestions": db_suggestions
            }

        except Exception as exc:
            logger.exception("DomainAPIAgent failed")
            state.add_error(str(exc))
            state.set_status("domain_api_design", "failed", str(exc))
            return {"error": str(exc)}

    # -----------------------
    # Heuristics & Helpers
    # -----------------------
    def _heuristic_extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Attempt to extract entities from the text using headings and list patterns.
        Returns a list of dicts: { "name": str, "attributes": [str,...] }
        """
        entities: List[Dict[str, Any]] = []

        # Enhanced keyword-based entity detection
        # Common banking/finance domain entities
        domain_keywords = {
            "Customer": ["customer", "user", "client", "borrower"],
            "Loan": ["loan", "lending", "credit", "borrow"],
            "Account": ["account", "banking", "deposit"],
            "Transaction": ["transaction", "payment", "transfer"],
            "Interest": ["interest", "rate", "pricing", "apr"],
            "Fee": ["fee", "charge", "penalty"],
            "Profile": ["profile", "kyc", "identity"],
            "Document": ["document", "file", "upload"],
            "Application": ["application", "apply", "request"],
            "Repayment": ["repayment", "installment", "emi"],
            "Score": ["score", "rating", "credit score", "risk"],
            "Campaign": ["campaign", "promotion", "offer"],
            "Configuration": ["configuration", "setting", "rule"],
            "Compliance": ["compliance", "regulation", "audit"],
        }

        # Detect entities by keyword frequency
        found_entities = {}
        for entity_name, keywords in domain_keywords.items():
            count = sum(len(re.findall(rf"\b{kw}\b", text, re.IGNORECASE)) for kw in keywords)
            if count >= 2:  # Entity must be mentioned at least twice
                found_entities[entity_name] = count

        # Create entities with domain-appropriate attributes
        for entity_name in sorted(found_entities.keys(), key=lambda k: found_entities[k], reverse=True):
            attrs = self._generate_entity_attributes(entity_name, text)
            entities.append({"name": entity_name, "attributes": attrs})

        # Look for sections likely describing entities (e.g., "Profile", "Account", "Customer")
        # Heuristic: headings with singular nouns
        for m in re.finditer(r"(?im)^#{1,3}\s*([A-Z][A-Za-z0-9 &\-]+)\s*$", text):
            title = m.group(1).strip()
            # Skip if already found
            if any(e["name"].lower() == title.lower() for e in entities):
                continue
            # capture a short block after heading for attributes list
            block_start = m.end()
            block = text[block_start:block_start + 800]
            attrs = []
            # find lines like "- id, email, created_at" or "attributes: id, email"
            list_matches = re.findall(r"[-•]\s*([A-Za-z0-9 _\-/]+(?:, *[A-Za-z0-9 _\-/]+)*)", block)
            for lm in list_matches:
                for part in re.split(r"[,\n;|]+", lm):
                    if part.strip():
                        attrs.append(self.normalize_string(part))
            # fallback: look for inline "fields: a, b, c"
            fm = re.search(r"(?i)(fields|attributes)\s*[:\-]\s*([A-Za-z0-9_,\s\-]+)", block)
            if fm:
                for part in re.split(r"[,\n;|]+", fm.group(2)):
                    if part.strip():
                        attrs.append(self.normalize_string(part))
            if title and (attrs or len(entities) < 10):
                entities.append({"name": title, "attributes": attrs if attrs else self._generate_entity_attributes(title, text)})
        
        # Ensure we have at least some entities
        if not entities:
            # fallback to basic entities
            entities = [
                {"name": "Customer", "attributes": self._generate_entity_attributes("Customer", text)},
                {"name": "Transaction", "attributes": self._generate_entity_attributes("Transaction", text)},
                {"name": "Account", "attributes": self._generate_entity_attributes("Account", text)},
            ]
        
        return entities

    def _heuristic_extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Very light-weight extraction of relationships: look for words like 'owns', 'has', 'belongs to'.
        Returns relationships as dicts compatible with diagram_plan.
        """
        rels: List[Dict[str, Any]] = []
        names = [e["name"] for e in entities if e.get("name")]
        for a in names:
            for b in names:
                if a == b:
                    continue
                # search "A owns B" or "B belongs to A"
                if re.search(rf"\b{a}\b.*\b(owns|owns the|has)\b.*\b{b}\b", text, flags=re.I):
                    rels.append({"from": a, "to": b, "type": "aggregation"})
                if re.search(rf"\b{b}\b.*\b(belongs to|owned by)\b.*\b{a}\b", text, flags=re.I):
                    rels.append({"from": a, "to": b, "type": "association"})
        return rels

    def _enrich_attributes(self, entity_name: str, existing: List[str]) -> List[str]:
        """
        Ensure an entity has at least three sensible attributes.
        If insufficient, synthesize common fields.
        """
        attrs = list(existing or [])
        
        # Use domain-specific attributes first, then fall back to generic
        attrs = self._generate_entity_attributes(entity_name, "", existing)
        
        return attrs

    def _generate_entity_attributes(self, entity_name: str, context_text: str = "", existing: List[str] = None) -> List[str]:
        """
        Generate appropriate attributes for an entity based on its name and context.
        """
        attrs = list(existing or [])
        entity_lower = entity_name.lower()
        
        # Domain-specific attribute mappings
        attribute_map = {
            "customer": ["customer_id", "name", "email", "mobile_number", "date_of_birth", "kyc_status", "risk_tier", "created_at", "updated_at", "status"],
            "loan": ["loan_id", "customer_id", "principal_amount", "interest_rate", "tenure_months", "emi_amount", "loan_type", "disbursement_date", "status", "created_at"],
            "interest": ["interest_id", "loan_id", "base_rate", "margin", "risk_adjustment", "effective_rate", "apr", "calculation_method", "created_at", "updated_at"],
            "account": ["account_id", "customer_id", "account_number", "account_type", "balance", "currency", "status", "opened_date", "created_at", "updated_at"],
            "transaction": ["transaction_id", "account_id", "amount", "transaction_type", "timestamp", "status", "reference_number", "description", "created_at"],
            "profile": ["profile_id", "customer_id", "full_name", "address", "city", "country", "postal_code", "phone_number", "created_at", "updated_at"],
            "document": ["document_id", "customer_id", "document_type", "file_path", "file_size", "upload_date", "verification_status", "expiry_date", "created_at"],
            "application": ["application_id", "customer_id", "loan_type", "requested_amount", "application_date", "status", "decision", "approved_by", "created_at"],
            "repayment": ["repayment_id", "loan_id", "installment_number", "due_date", "payment_date", "amount", "principal_component", "interest_component", "status"],
            "fee": ["fee_id", "loan_id", "fee_type", "amount", "calculation_basis", "applied_date", "status", "created_at", "updated_at"],
            "score": ["score_id", "customer_id", "credit_score", "risk_tier", "score_date", "bureau_name", "factors", "validity_period", "created_at"],
            "campaign": ["campaign_id", "campaign_name", "discount_rate", "start_date", "end_date", "eligibility_criteria", "status", "created_at", "updated_at"],
            "configuration": ["config_id", "config_name", "config_type", "value", "description", "updated_by", "created_at", "updated_at", "status"],
            "compliance": ["compliance_id", "rule_name", "rule_type", "description", "applicable_from", "applicable_to", "status", "created_at", "updated_at"],
        }
        
        # Find best match
        for key, suggested_attrs in attribute_map.items():
            if key in entity_lower:
                # Use suggested attributes, keeping existing ones
                for attr in suggested_attrs:
                    if attr not in attrs:
                        attrs.append(attr)
                        if len(attrs) >= 10:  # Limit to 10 attributes
                            break
                break
        
        # If no match found, use generic attributes
        if len(attrs) < 5:
            defaults = ["id", "name", "description", "created_at", "updated_at", "status", "created_by", "modified_by"]
            for d in defaults:
                if d not in attrs:
                    attrs.append(d)
                if len(attrs) >= 8:
                    break
        
        return attrs[:12]  # Cap at 12 attributes

    def _ensure_request_response(self, req: Dict[str, str], resp: Dict[str, str], entities: List[EntityData]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Ensure request and response schemas are non-empty and sensible.
        If empty, infer from first entity.
        """
        if req and resp:
            return req, resp
        # pick primary entity to infer
        primary = entities[0] if entities else None
        if primary:
            # infer simple request/response
            if not req:
                # for create endpoints, require minimal fields
                req = {a: "string" for a in (primary.attributes[:3] if primary.attributes else ["id", "name", "email"])}
            if not resp:
                resp = {"id": "string", **({a: "string" for a in (primary.attributes[:3] if primary.attributes else [])})}
        else:
            if not req:
                req = {"payload": "object"}
            if not resp:
                resp = {"result": "object"}
        return req, resp

    def _heuristic_design_apis(self, entities_input: List[Dict] | List[EntityData], text: str) -> List[APIData]:
        """
        Build REST-style API surface from entities.
        Produces CRUD endpoints for main entities and some domain-specific endpoints.
        """
        apis: List[APIData] = []
        # normalize to entity names + attributes
        normalized = []
        for e in entities_input:
            if isinstance(e, dict):
                name = self.normalize_string(e.get("name") or "entity")
                attrs = [self.normalize_string(a) for a in (e.get("attributes") or [])]
            else:
                name = e.name
                attrs = e.attributes
            normalized.append((name, attrs))

        # Generate comprehensive CRUD APIs for each entity
        for name, attrs in normalized[:10]:  # Process up to 10 entities
            base_path = f"/api/{name.lower()}s".replace(" ", "-")
            entity_lower = name.lower()
            
            # Identify key attributes for requests/responses
            id_field = next((a for a in attrs if "id" in a.lower()), "id")
            key_attrs = [a for a in attrs[:5] if a != id_field]  # First 5 non-id attributes
            
            # CREATE (POST)
            create_req = {a: "string" for a in key_attrs[:4]} if key_attrs else {"name": "string"}
            create_resp = {id_field: "string", "status": "string", "created_at": "timestamp"}
            apis.append(APIData(
                name=f"POST {base_path}", 
                description=f"Create a new {name}", 
                request=create_req, 
                response=create_resp
            ))
            
            # READ by ID (GET)
            get_resp = {id_field: "string"}
            for attr in attrs[:6]:
                get_resp[attr] = "string"
            apis.append(APIData(
                name=f"GET {base_path}/{{id}}", 
                description=f"Retrieve {name} by ID", 
                request={}, 
                response=get_resp
            ))
            
            # LIST (GET)
            list_resp = {
                "items": "array",
                "total_count": "integer",
                "page": "integer",
                "page_size": "integer"
            }
            apis.append(APIData(
                name=f"GET {base_path}", 
                description=f"List all {name}s with pagination", 
                request={"page": "integer", "page_size": "integer"}, 
                response=list_resp
            ))
            
            # UPDATE (PUT)
            update_req = {a: "string" for a in key_attrs[:4]} if key_attrs else {"name": "string"}
            update_resp = {id_field: "string", "status": "string", "updated_at": "timestamp"}
            apis.append(APIData(
                name=f"PUT {base_path}/{{id}}", 
                description=f"Update {name}", 
                request=update_req, 
                response=update_resp
            ))
            
            # DELETE (DELETE)
            apis.append(APIData(
                name=f"DELETE {base_path}/{{id}}", 
                description=f"Delete {name}", 
                request={}, 
                response={"status": "string", "message": "string"}
            ))

        # Domain-specific business APIs based on context
        text_lower = text.lower()
        
        if "loan" in text_lower or "credit" in text_lower:
            apis.append(APIData(
                name="POST /api/loans/calculate-emi", 
                description="Calculate EMI for loan", 
                request={"principal_amount": "decimal", "interest_rate": "decimal", "tenure_months": "integer"}, 
                response={"emi_amount": "decimal", "total_interest": "decimal", "total_amount": "decimal"}
            ))
            apis.append(APIData(
                name="POST /api/loans/apply", 
                description="Submit loan application", 
                request={"customer_id": "string", "loan_type": "string", "amount": "decimal", "tenure": "integer"}, 
                response={"application_id": "string", "status": "string", "reference_number": "string"}
            ))
            
        if "interest" in text_lower or "pricing" in text_lower or "rate" in text_lower:
            apis.append(APIData(
                name="GET /api/interest/rates", 
                description="Get applicable interest rates", 
                request={"customer_id": "string", "loan_type": "string", "amount": "decimal"}, 
                response={"base_rate": "decimal", "risk_adjustment": "decimal", "effective_rate": "decimal", "apr": "decimal"}
            ))
            apis.append(APIData(
                name="POST /api/interest/calculate", 
                description="Calculate interest for loan", 
                request={"loan_id": "string", "calculation_date": "date"}, 
                response={"interest_amount": "decimal", "calculation_method": "string"}
            ))
            
        if "penalty" in text_lower or "fee" in text_lower:
            apis.append(APIData(
                name="POST /api/fees/calculate-penalty", 
                description="Calculate late payment penalty", 
                request={"loan_id": "string", "days_overdue": "integer"}, 
                response={"penalty_amount": "decimal", "fee_breakdown": "object"}
            ))
            
        if "kyc" in text_lower or "verification" in text_lower or "identity" in text_lower:
            apis.append(APIData(
                name="POST /api/kyc/verify", 
                description="Submit documents for KYC verification", 
                request={"customer_id": "string", "documents": "array", "document_type": "string"}, 
                response={"verification_id": "string", "status": "string"}
            ))
            apis.append(APIData(
                name="GET /api/kyc/status/{customer_id}", 
                description="Get KYC verification status", 
                request={}, 
                response={"customer_id": "string", "kyc_status": "string", "verified_date": "timestamp"}
            ))
            
        if "repayment" in text_lower or "payment" in text_lower or "installment" in text_lower:
            apis.append(APIData(
                name="POST /api/repayments/schedule", 
                description="Generate repayment schedule", 
                request={"loan_id": "string"}, 
                response={"installments": "array", "total_installments": "integer"}
            ))
            apis.append(APIData(
                name="POST /api/repayments/pay", 
                description="Process repayment", 
                request={"loan_id": "string", "amount": "decimal", "payment_method": "string"}, 
                response={"transaction_id": "string", "status": "string", "receipt": "object"}
            ))
            
        if "campaign" in text_lower or "promotion" in text_lower or "offer" in text_lower:
            apis.append(APIData(
                name="GET /api/campaigns/active", 
                description="Get active promotional campaigns", 
                request={"customer_id": "string"}, 
                response={"campaigns": "array", "eligible_offers": "array"}
            ))

        return apis

    def _build_diagram_plan(self, entities: List[EntityData], relationships_raw: Any) -> Dict[str, Any]:
        """
        Build a diagram_plan dict suitable for utils/diagram_converter.diagram_plan_to_text
        Structure:
          { "class": { "nodes": [ {name, attributes}, ... ], "relations": [ {from,to,type,label}, ... ] },
            "sequences": [...] }
        """
        nodes = []
        for e in entities:
            nodes.append({"name": e.name, "attributes": e.attributes})

        rels = []
        # accept either list of dicts or simple strings in relationships_raw
        if isinstance(relationships_raw, list):
            for r in relationships_raw:
                if isinstance(r, dict):
                    frm = r.get("from") or r.get("source") or r.get("a")
                    to = r.get("to") or r.get("target") or r.get("b")
                    rtype = r.get("type") or r.get("relation") or "association"
                    label = r.get("label") or ""
                    if frm and to:
                        rels.append({"from": frm, "to": to, "type": rtype, "label": label})
                elif isinstance(r, str):
                    # try simple "A -> B : label" parse
                    s = r.strip()
                    parts = re.split(r"\s*[:]\s*", s, maxsplit=1)
                    relation_part = parts[0]
                    label = parts[1] if len(parts) > 1 else ""
                    arrow_match = re.search(r"([A-Za-z0-9_]+)\s*[-=\.]*[>\-]+\s*([A-Za-z0-9_]+)", relation_part)
                    if arrow_match:
                        rels.append({"from": arrow_match.group(1), "to": arrow_match.group(2), "type": "association", "label": label})
        return {"class": {"nodes": nodes, "relations": rels}, "sequences": []}

    def _generate_db_suggestions(self, entities: List[EntityData]) -> Dict[str, Any]:
        """
        Produce a simple DB schema suggestion list: table name, columns with types, indexes.
        """
        suggestions = {}
        for e in entities:
            cols = {}
            for a in e.attributes:
                # simple heuristic: fields containing "id" -> UUID, amount->decimal, created_at->timestamp, email->string, default->string
                key = a.lower()
                if "id" in key and key != "id":
                    cols[a] = "string /* uuid */"
                elif "amount" in key or "balance" in key or "money" in key:
                    cols[a] = "decimal"
                elif "created" in key or "updated" in key or "timestamp" in key:
                    cols[a] = "timestamp"
                elif "email" in key:
                    cols[a] = "string /* email */"
                else:
                    cols[a] = "string"
            # ensure primary key
            if "id" not in cols:
                cols = {"id": "uuid PRIMARY KEY", **cols}
            suggestions[e.name] = {"table": e.name.lower() + "s", "columns": cols, "indexes": ["id"]}
        return suggestions
