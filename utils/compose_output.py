# utils/compose_output.py
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, date
import base64
import json

def hld_to_markdown(
    requirement_name: str,
    prd_markdown: str,
    authentication,
    integrations,
    entities,
    apis,
    use_cases,
    nfrs,
    risks,
    class_mermaid_text: Optional[str],
    sequence_mermaid_texts: Optional[List[str]],
    # optional image paths (can be absolute or relative)
    class_img: Optional[str] = None,
    seq_imgs: Optional[List[Optional[str]]] = None,
    # base directory where HLD files are written (for resolving relative image paths)
    hld_base_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Compose the final HLD markdown/HTML-friendly text.
    If class_img / seq_imgs exist, they are embedded as base64/inline; otherwise Mermaid code is used.
    """

    # ---------- helpers scoped to this function ----------
    def _section(title: str, body: str) -> str:
        return f"## {title}\n\n{body}\n\n"

    def _to_jsonable(o):
        # make common non-JSON types serializable
        try:
            json.dumps(o)  # fast path
            return o
        except TypeError:
            pass
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(map(str, o))
        # last resort
        return str(o)

    def _json_block(obj) -> str:
        return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False, default=_to_jsonable) + "\n```\n"

    def _resolve_img_path(img_path: str) -> Optional[Path]:
        if not img_path:
            return None
        p = Path(img_path)
        if p.is_absolute():
            return p if p.exists() else None
        # try relative to CWD first
        if p.exists():
            return p
        # then relative to the HLD output directory
        if hld_base_dir:
            hp = Path(hld_base_dir) / p
            if hp.exists():
                return hp
        return None

    def _embed_image_tag(resolved_path: Path, alt: str = "") -> str:
        ext = resolved_path.suffix.lower().lstrip(".")  # "png" | "svg" | etc.
        data = resolved_path.read_bytes()
        if ext == "svg":
            # Prefer raw inline SVG so it stays crisp in PDF
            try:
                svg_text = data.decode("utf-8")
                if "<svg" in svg_text:
                    return f"<div class='diagram' aria-label='{alt}'>{svg_text}</div>"
            except Exception:
                pass
            b64 = base64.b64encode(data).decode("utf-8")
            return f"<img alt='{alt}' style='max-width:100%' src='data:image/svg+xml;base64,{b64}'/>"
        mime = f"image/{ext if ext != 'jpg' else 'jpeg'}"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"<img alt='{alt}' style='max-width:100%' src='data:{mime};base64,{b64}'/>"

    def _diagram_block(title: str, img_path: Optional[str], mermaid_text: Optional[str]) -> str:
        if img_path:
            resolved = _resolve_img_path(img_path)
            if resolved and resolved.exists():
                return f"### {title}\n\n{_embed_image_tag(resolved, alt=title)}\n\n"
        if mermaid_text:
            return f"### {title}\n\n```mermaid\n{mermaid_text}\n```\n\n"
        return f"### {title}\n\n*(No diagram available)*\n\n"
    # -----------------------------------------------------

    seq_imgs = seq_imgs or []
    sequence_mermaid_texts = sequence_mermaid_texts or []

    header = (
        f"# High-Level Design — {requirement_name}\n\n"
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
    )

    parts: List[str] = [header]

    # PRD extract
    parts.append(_section("Extracted Requirements (from PRD)", prd_markdown or ""))

    # Auth & Integrations
    parts.append(_section("Authentication", _json_block(authentication)))
    parts.append(_section("Integrations", _json_block(integrations)))

    # Domain model & APIs
    parts.append(_section("Domain Entities", _json_block(entities)))
    parts.append(_section("APIs", _json_block(apis)))

    # Diagrams
    parts.append("## Diagrams\n\n")
    parts.append(_diagram_block("Class Diagram", class_img, class_mermaid_text))

    # Render images even if mermaid text missing (and vice versa)
    if sequence_mermaid_texts or seq_imgs:
        parts.append("#### Sequence Diagrams\n\n")
        n = max(len(sequence_mermaid_texts), len(seq_imgs))
        for idx in range(n):
            mermaid = sequence_mermaid_texts[idx] if idx < len(sequence_mermaid_texts) else None
            img_for_idx = seq_imgs[idx] if idx < len(seq_imgs) else None
            parts.append(_diagram_block(f"Sequence #{idx+1}", img_for_idx, mermaid))

    # Behavior & Quality
    parts.append(_section("Use Cases", _json_block(use_cases)))
    parts.append(_section("Non-Functional Requirements (NFRs)", _json_block(nfrs)))
    parts.append(_section("Risks & Assumptions", _json_block(risks)))

    return "".join(parts)


def save_markdown(md_text: str, out_path: Union[str, Path]) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_text, encoding="utf-8")
    return path