"""
Base Node class for HLD workflow nodes
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableLambda
from state.models import HLDState
from agent.base_agent import BaseAgent
logger = logging.getLogger(__name__)

class BaseNode(ABC):
    """
    Abstract base class for workflow nodes.
    Each node encapsulates an Agent and manages its execution, error handling, and I/O paths.
    """

    def __init__(self, name: str, agent: BaseAgent):
        self.name = name
        self.agent = agent
        logger.debug(f"[INIT] Node '{self.name}' initialized with agent {agent.__class__.__name__}")

    # ------------------------
    # Abstract core execution
    # ------------------------
    @abstractmethod
    def execute(self, state: HLDState) -> HLDState:
        """
        Execute node logic using its associated Agent.
        Must be implemented by derived classes to customize data flow.
        """
        pass

    # ------------------------
    # Runnable wrapper for LangGraph integration
    # ------------------------
    def get_runnable(self):
        """
        Returns a LangChain-compatible Runnable that executes this node's logic.
        It accepts a dict-like HLDState and returns the updated dict form.
        """
        def _runnable(state_dict: Dict[str, Any]) -> Dict[str, Any]:
            try:
                state = HLDState(**state_dict)
                logger.info("=" * 80)
                logger.info(f"NODE: {self.name.upper()}")
                logger.info("=" * 80)
                logger.info(f"Starting execution with agent: {self.agent.__class__.__name__}")
                start = time.time()
                updated_state = self.execute(state)
                elapsed = round(time.time() - start, 2)
                logger.info(f"✓ Node execution completed successfully!")
                logger.info(f"⏱ Duration: {elapsed}s")
                logger.info("=" * 80)
                return updated_state.dict()
            except Exception as e:
                logger.exception(f"✗ Node execution failed: {e}")
                logger.info("=" * 80)
                state = HLDState(**state_dict)
                state.add_error(f"Node {self.name} failed: {e}")
                state.set_status(self.name, "failed", str(e))
                return state.dict()

        return RunnableLambda(_runnable)

    # ------------------------
    # Helper methods
    # ------------------------
    def _get_output_dir(self, state: HLDState) -> Path:
        """
        Determine the base output directory for this workflow.
        Always use Project/output as base directory.
        """
        project_dir = Path(__file__).resolve().parent.parent
        if state.output and getattr(state.output, "output_dir", None):
            base = Path(state.output.output_dir)
        else:
            base = project_dir / "output" / (state.requirement_name or "unknown")
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _ensure_output_dirs(self, output_dir: Path) -> Dict[str, Path]:
        """
        Create standard output subdirectories.
        Returns dict of created subpaths.
        """
        subdirs = {
            "json": output_dir / "json",
            "diagrams": output_dir / "diagrams",
            "hld": output_dir / "hld"
        }
        for d in subdirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return subdirs

    def _get_relative_path(self, from_path: Path, to_path: Path) -> str:
        """
        Calculate relative path between two paths (for embedding in reports).
        Returns relative string; falls back to absolute on failure.
        """
        try:
            return str(to_path.relative_to(from_path))
        except Exception:
            return str(to_path.resolve())

    # ------------------------
    # Execution monitoring
    # ------------------------
    def _log_start(self):
        logger.info(f"[Node:{self.name}] Execution started.")

    def _log_end(self, duration: float):
        logger.info(f"[Node:{self.name}] Execution completed in {duration:.2f}s.")

    def _log_error(self, error: str):
        logger.error(f"[Node:{self.name}] Error: {error}")

    def _monitor_execution(self, func, *args, **kwargs):
        """
        Monitor runtime performance and catch exceptions.
        Returns (result, duration, error)
        """
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            return result, duration, None
        except Exception as e:
            duration = round(time.time() - start, 2)
            self._log_error(str(e))
            return None, duration, e
