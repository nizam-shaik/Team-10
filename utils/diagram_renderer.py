# utils/diagram_renderer.py
# Backend-only renderer. One public function: render_diagrams(...)
from __future__ import annotations
from pathlib import Path
from typing import Dict
import os
import logging

logger = logging.getLogger(__name__)

def render_diagrams(
    mermaid_map: Dict[str, str],
    out_dir: str,
    want_images: bool = True,
    image_fmt: str = "png",       # "svg" | "png"
    save_sources: bool = True,    # write <n>.mmd
) -> Dict[str, Dict[str, str]]:
    """
    Writes .mmd sources and renders images using mmdc (Mermaid CLI).
    
    Flow:
    1. Writes .mmd files to out_dir/
    2. If want_images=True, renders .mmd → images using mmdc into out_dir/img/
    
    Returns: {"mmd": {name: path}, "images": {name: path}}
    """

    # --- helpers INSIDE the function (no extra public APIs) ---

    def _write_sources(_base: Path) -> Dict[str, str]:
        out = {}
        if not save_sources:
            return out
        for name, code in mermaid_map.items():
            p = _base / f"{name}.mmd"
            p.write_text(code, encoding="utf-8")
            out[name] = str(p)
        return out

    def _render_images_mmdc(_img_dir: Path, _base: Path) -> Dict[str, str]:
        """
        Render images using Mermaid Ink API (no npm/node required!).
        Falls back to mmdc CLI if available, otherwise uses free public API.
        """
        import subprocess
        import requests
        import base64
        from urllib.parse import quote
        
        out = {}
        
        # Check if mmdc is available
        mmdc_available = False
        try:
            subprocess.run(["mmdc", "--version"], check=True, 
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            mmdc_available = True
            logger.info(f"Using mmdc CLI to render {len(mermaid_map)} diagrams")
        except:
            logger.info(f"mmdc not found, using Mermaid Ink API to render {len(mermaid_map)} diagrams")
        
        for name, code in mermaid_map.items():
            out_img = _img_dir / f"{name}.{image_fmt}"
            
            try:
                if mmdc_available:
                    # Use mmdc CLI
                    in_mmd = _base / f"{name}.mmd"
                    cmd = ["mmdc", "-i", str(in_mmd), "-o", str(out_img)]
                    if image_fmt == "png":
                        cmd += ["-t", "default"]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                else:
                    # Use Mermaid Ink API (free, no installation needed)
                    # Encode mermaid code to base64
                    encoded = base64.b64encode(code.encode('utf-8')).decode('utf-8')
                    
                    # Mermaid Ink API endpoint
                    if image_fmt == "svg":
                        url = f"https://mermaid.ink/svg/{encoded}"
                    else:
                        url = f"https://mermaid.ink/img/{encoded}"
                    
                    # Download rendered image
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Save to file
                    out_img.write_bytes(response.content)
                
                out[name] = str(out_img)
                logger.info(f"✓ Successfully rendered {name}.{image_fmt}")
                
            except Exception as e:
                logger.error(f"✗ Failed to render {name}: {e}")
                continue
        
        logger.info(f"Rendered {len(out)}/{len(mermaid_map)} diagrams successfully")
        return out

    # --- main flow ---
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, str]] = {"mmd": {}, "images": {}}
    
    # Step 1: Always write .mmd source files first
    results["mmd"] = _write_sources(base)

    # Step 2: Optionally render images from the saved .mmd files
    if not want_images:
        return results

    img_dir = base / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Use mmdc CLI to render from already-saved .mmd files
    results["images"] = _render_images_mmdc(img_dir, base)

    return results