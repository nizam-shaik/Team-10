import json
from typing import Dict, List, Any

def diagram_plan_to_text(diagram_plan: Dict) -> Dict:
    """
    ONE FUNCTION ONLY.
    Converts a (sometimes messy) diagram plan into Mermaid text.
    Accepts dicts or JSON strings for sub-parts; tolerates strings in node/relation/step lists.
    Returns:
      {"class_text": str, "sequence_texts": [str, ...]}  OR  {"error": str}
    """

    # --- local helpers (scoped here; no imports/side effects) ---
    def _as_dict(v: Any) -> Dict:
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                j = json.loads(v)
                return j if isinstance(j, dict) else {}
            except Exception:
                return {}
        return {}

    def _as_list(v: Any) -> List:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                j = json.loads(v)
                return j if isinstance(j, list) else []
            except Exception:
                return []
        return []

    def _clean_name(s: Any, default: str = "") -> str:
        s = (str(s or "").strip())
        # Mermaid identifiers: keep [A-Za-z0-9_ -], then drop spaces
        s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", " "))
        return s.replace(" ", "") or default

    def _escape_field(s: Any) -> str:
        # For class fields we output "type name"
        s = str(s or "").strip().replace('"', "'")
        return s[:200]

    def _escape_msg(s: Any) -> str:
        s = str(s or "").strip()
        return s.replace(":", r"\:").replace('"', "'")[:240]

    def _relation_arrow(rtype: str) -> str:
        r = (rtype or "").lower().strip()
        mapping = {
            "aggregation": "o--", "aggregate": "o--",
            "composition": "*--", "composes": "*--",
            "inherits": "<|--",  "inheritance": "<|--", "extends": "<|--",
            "implements": "<|..",
            "uses": "..>", "use": "..>",
            "association": "--", "assoc": "--", "has": "--",
            "dependency": "<.."
        }
        return mapping.get(r, "--")

    if not isinstance(diagram_plan, dict):
        return {"error": "diagram_plan must be a dict"}

    # Optional name -> attributes map if caller provided entities
    entity_attrs = {}
    for ent in _as_list(diagram_plan.get("entities")):
        if isinstance(ent, dict):
            name = _clean_name(ent.get("name"))
            attrs = [str(a).strip() for a in _as_list(ent.get("attributes") or ent.get("fields")) if str(a).strip()]
            if name:
                entity_attrs[name] = attrs

    # ---------- CLASS ----------
    class_lines: List[str] = ["classDiagram"]

    # Support both shapes: class.{nodes,relations} OR class.{classes,relationships}
    cplan = _as_dict(diagram_plan.get("class")) or _as_dict(_as_dict(diagram_plan.get("diagram_plan")).get("class"))
    nodes_like = _as_list(cplan.get("nodes")) or _as_list(cplan.get("classes"))
    rels_like  = _as_list(cplan.get("relations")) or _as_list(cplan.get("relationships"))

    any_node = False
    for n in nodes_like:
        if isinstance(n, dict):
            name = _clean_name(n.get("name"), "Unnamed")
            fields = _as_list(n.get("fields") or n.get("attributes"))
        else:
            name = _clean_name(n, "Unnamed")
            fields = []

        if not name:
            continue
        any_node = True

        # Enrich empty fields with entity attributes (if provided)
        if not fields and name in entity_attrs:
            fields = [{"name": a, "type": "string"} for a in entity_attrs[name][:12]]

        if fields:
            class_lines.append(f"class {name} {{")
            for f in fields:
                if isinstance(f, dict):
                    fname = _clean_name(f.get("name"), "field")
                    ftype = _escape_field(f.get("type") or "string")
                    class_lines.append(f"  {ftype} {fname}")
                else:  # string field name
                    fname = _escape_field(f)
                    if fname:
                        class_lines.append(f"  string {fname}")
            class_lines.append("}")
        else:
            class_lines.append(f"class {name}")

    any_rel = False
    for r in rels_like:
        # Allow raw Mermaid relation lines like "A --> B : uses"
        if isinstance(r, str):
            s = r.strip()
            if any(tok in s for tok in ("--", "..", "<|", "|>", "--|>", "<|--")):
                any_rel = True
                class_lines.append(s)
                continue
            # Else try a light "A->B: label" parse
            frm = to = label = ""
            if "->" in s:
                left, right = s.split("->", 1)
                frm = _clean_name(left)
                if ":" in right:
                    _to, _label = [x.strip() for x in right.split(":", 1)]
                    to = _clean_name(_to)
                    label = _escape_field(_label)
                else:
                    to = _clean_name(right)
            if frm and to:
                any_rel = True
                class_lines.append(f"{frm} -- {to}" + (f" : {label}" if label else ""))
            continue

        # Dict relation
        frm = _clean_name(r.get("from") or r.get("source"))
        to  = _clean_name(r.get("to")   or r.get("target"))
        rtype = (r.get("type") or "association")
        label = _escape_field(r.get("label") or "")
        edge = _relation_arrow(rtype)
        if frm and to:
            any_rel = True
            class_lines.append(f"{frm} {edge} {to}" + (f" : {label}" if label else ""))

    if not any_node and not any_rel:
        class_lines.append("class Placeholder")
        class_lines.append("Placeholder : note: No entities/relations were generated")

    class_text = "\n".join(class_lines)

    # ---------- SEQUENCES ----------
    sequence_texts: List[str] = []
    raw_sequences = diagram_plan.get("sequences")
    if not raw_sequences and "diagram_plan" in diagram_plan:
        raw_sequences = _as_dict(diagram_plan["diagram_plan"]).get("sequences")
    seq_list = _as_list(raw_sequences)

    # If the plan is a flat list of step dicts, treat it as ONE sequence
    if seq_list and all(isinstance(x, dict) and any(k in x for k in
        ("from", "to", "message", "actor", "source", "target", "receiver", "action", "text")) for x in seq_list):
        lines = ["sequenceDiagram"]
        any_step = False
        for step in seq_list:
            frm = _clean_name(step.get("from") or step.get("actor") or step.get("source"))
            to  = _clean_name(step.get("to")   or step.get("target") or step.get("receiver"))
            msg = _escape_msg(step.get("message") or step.get("action") or step.get("text"))
            if frm and to and msg:
                any_step = True
                lines.append(f"{frm}->>{to}: {msg}")
            elif msg:
                any_step = True
                lines.append(f"Note over {frm or 'System'}: {msg}")
        if not any_step:
            lines.append("Note over System: No steps were generated")
        sequence_texts.append("\n".join(lines))
    else:
        # List of sequences; each has steps[]
        for s in seq_list:
            if isinstance(s, dict):
                # Get actors from the sequence plan (may be empty)
                actors_from_plan = [_clean_name(a) for a in _as_list(s.get("actors")) if _clean_name(a)]
                steps = _as_list(s.get("steps") or s.get("lines") or s.get("sequence") or [])
                
                # Extract actors from steps if not provided in plan
                if not actors_from_plan and steps:
                    actors_set = set()
                    for step in steps:
                        if isinstance(step, dict):
                            frm = _clean_name(step.get("from") or step.get("actor") or step.get("source"))
                            to = _clean_name(step.get("to") or step.get("target") or step.get("receiver"))
                            if frm:
                                actors_set.add(frm)
                            if to:
                                actors_set.add(to)
                    actors_from_plan = list(actors_set)
                
                actors = actors_from_plan
            else:
                actors = []
                steps = [{"from": "System", "to": "System", "message": str(s).strip()}]

            lines = ["sequenceDiagram"]
            
            # Classify actors vs participants based on name patterns
            if actors:
                # Common human role keywords
                human_keywords = [
                    'user', 'customer', 'admin', 'manager', 'officer', 'agent', 
                    'employee', 'staff', 'clerk', 'operator', 'analyst', 'developer',
                    'tester', 'viewer', 'guest', 'member', 'subscriber', 'buyer',
                    'seller', 'vendor', 'client', 'patient', 'doctor', 'nurse'
                ]
                
                # System/technical keywords (participants)
                system_keywords = [
                    'system', 'service', 'api', 'database', 'server', 'module',
                    'component', 'engine', 'gateway', 'controller', 'handler',
                    'processor', 'manager', 'repository', 'dao', 'facade'
                ]
                
                human_actors = []
                system_participants = []
                
                for a in actors:
                    a_lower = a.lower()
                    # Check if it's a human actor
                    is_human = any(keyword in a_lower for keyword in human_keywords)
                    # Check if it's explicitly a system
                    is_system = any(keyword in a_lower for keyword in system_keywords)
                    
                    if is_human and not is_system:
                        human_actors.append(a)
                    else:
                        system_participants.append(a)
                
                # Declare human actors first (using 'actor' keyword)
                for actor in human_actors:
                    lines.append(f"actor {actor}")
                
                # Then declare system participants (using 'participant' keyword)
                for participant in system_participants:
                    lines.append(f"participant {participant}")

            any_step = False
            for step in steps:
                if isinstance(step, dict):
                    frm = _clean_name(step.get("from") or step.get("actor") or step.get("source"))
                    to  = _clean_name(step.get("to")   or step.get("target") or step.get("receiver"))
                    msg = _escape_msg(step.get("message") or step.get("action") or step.get("text"))
                else:
                    sstep = str(step)
                    frm = to = ""
                    msg = _escape_msg(sstep)
                    if "->" in sstep:
                        left, right = sstep.split("->", 1)
                        frm = _clean_name(left)
                        if ":" in right:
                            _to, _msg = [x.strip() for x in right.split(":", 1)]
                            to = _clean_name(_to); msg = _escape_msg(_msg)
                        else:
                            to = _clean_name(right)

                if frm and to and msg:
                    any_step = True
                    lines.append(f"{frm}->>{to}: {msg}")
                elif msg:
                    any_step = True
                    lines.append(f"Note over {frm or 'System'}: {msg}")

            if not any_step:
                lines.append("Note over System: No steps were generated")

            sequence_texts.append("\n".join(lines))

    if not sequence_texts:
        sequence_texts = ["sequenceDiagram\nNote over System: No sequences were generated"]

    return {"class_text": class_text, "sequence_texts": sequence_texts}