from __future__ import annotations
"""
HLD Workflow - Main workflow orchestrator for HLD generation
"""

# TODO: Import time for performance tracking
# TODO: Import Runnable from langchain_core
# TODO: Import HLDState from state.models
# TODO: Import WorkflowInput, WorkflowOutput, ConfigSchema from state.schema
# TODO: Implement HLDWorkflow class
# TODO: Implement __init__(workflow_type: str = "sequential") method
#       - Store workflow_type parameter
#       - Call _create_graph() to build appropriate graph
# TODO: Implement _create_graph() -> Runnable method
#       - Select graph builder based on workflow_type
#       - Return compiled LangGraph runnable
#       - Call create_workflow_graph, create_parallel_workflow_graph, etc.
# TODO: Implement run(input_data: WorkflowInput) -> WorkflowOutput method
#       - Record start time
#       - Create initial state using create_initial_state()
#       - Invoke graph with state dictionary
#       - Convert result back to HLDState
#       - Calculate processing time
#       - Collect output paths from final state
#       - Return WorkflowOutput with results
# TODO: Implement arun(input_data: WorkflowInput) -> WorkflowOutput method
#       - Async version of run()
#       - Use ainvoke instead of invoke
#       - Same return and error handling as run()
# TODO: Implement stream(input_data: WorkflowInput) -> AsyncIterator method
#       - Stream workflow execution for real-time updates
#       - Yield state updates from astream()
#       - Handle errors and yield error messages
# TODO: Implement get_workflow_info() -> Dict[str, Any] method
#       - Return workflow configuration details
#       - Include type, nodes, capabilities
# TODO: Implement module-level factory functions
#       - create_hld_workflow(workflow_type: str) -> HLDWorkflow
#       - create_sequential_workflow() -> HLDWorkflow
#       - create_parallel_workflow() -> HLDWorkflow
#       - create_conditional_workflow() -> HLDWorkflow
# TODO: Add error handling
#       - Catch graph execution errors
#       - Provide meaningful error messages
#       - Return error status in WorkflowOutput
# TODO: Consider performance monitoring
#       - Track execution time per stage
#       - Log performance metrics
#       - Implement progress tracking
"""
HLD Workflow - Main workflow orchestrator for HLD generation
"""

import time
import asyncio
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, AsyncIterator

logger = logging.getLogger(__name__)

# Local lightweight WorkflowOutput dataclass to avoid depending on a possibly-unimplemented state.schema
@dataclass
class WorkflowOutput:
    success: bool
    state: Dict[str, Any]
    output_paths: Dict[str, Any]
    processing_time: float
    errors: Optional[list] = None
    warnings: Optional[list] = None


class HLDWorkflow:
    """
    High-level orchestrator for HLD generation.

    Usage:
        wf = HLDWorkflow(workflow_type="sequential")
        out = wf.run({"pdf_path": "data/Requirement-1.pdf", "config": {...}})
    """

    def __init__(self, workflow_type: str = "sequential"):
        self.workflow_type = workflow_type or "sequential"
        self.node_manager = None
        self._graph_callable = None
        self._compiled_graph = None
        
        logger.info("=" * 80)
        logger.info("HLD WORKFLOW INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Workflow Type: {workflow_type}")
        logger.info("=" * 80)
        
        self._create_graph()

    # -------------------------
    # Graph creation / binding
    # -------------------------
    def _create_graph(self) -> None:
        """
        Build an execution callable depending on workflow_type.
        Prefer compiled LangGraph runnables when available (e.g., from parallel_safe).
        Fallback to the NodeManager sequential executor.
        """
        # Lazy imports to avoid import errors if modules are not ready
        try:
            from nodes.node_manager import NodeManager
            self.node_manager = NodeManager()
        except Exception as e:
            logger.exception("Failed to import NodeManager. Sequential execution will not be available: %s", e)
            self.node_manager = None

        # Try to use a prebuilt parallel graph if requested and available
        if self.workflow_type in ("parallel", "conditional"):
            try:
                # prefer parallel_safe compiled runnable if present
                from workflow import parallel_safe  # Fixed import path!
                if self.workflow_type == "parallel" and hasattr(parallel_safe, "create_safe_parallel_workflow"):
                    logger.info("✓ Using parallel workflow with concurrent node execution")
                    logger.info("  Auth, Domain, and Behavior nodes will run in parallel")
                    runnable = parallel_safe.create_safe_parallel_workflow()
                    self._compiled_graph = runnable
                    self._graph_callable = self._invoke_compiled_graph
                    return
                if self.workflow_type == "conditional" and hasattr(parallel_safe, "create_conditional_workflow"):
                    logger.info("✓ Using conditional workflow with intelligent routing")
                    logger.info("  Workflow will route based on content analysis")
                    runnable = parallel_safe.create_conditional_workflow()
                    self._compiled_graph = runnable
                    self._graph_callable = self._invoke_compiled_graph
                    return
                if self.workflow_type == "parallel" and hasattr(parallel_safe, "create_batch_parallel_workflow"):
                    logger.info("✓ Using batch parallel workflow")
                    runnable = parallel_safe.create_batch_parallel_workflow()
                    self._compiled_graph = runnable
                    self._graph_callable = self._invoke_compiled_graph
                    return
            except Exception as e:
                logger.warning(f"Failed to load {self.workflow_type} workflow: {e}. Falling back to sequential.")
                logger.debug(f"{self.workflow_type.capitalize()} workflow error details:", exc_info=True)

        # Fallback: sequential callable using NodeManager
        if self.node_manager:
            logger.info("✓ Using sequential workflow with step-by-step node execution")
            self._graph_callable = self._run_sequential
            return

        # Last resort: no executor available — set a no-op that returns input state
        logger.warning("[HLDWorkflow] No executor available; workflow will be a no-op.")
        self._graph_callable = lambda state_dict: state_dict

    # -------------------------
    # Low-level invocation helpers
    # -------------------------
    def _invoke_compiled_graph(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a compiled LangGraph runnable (if available).
        This expects the compiled graph object to support `invoke` (synchronous).
        """
        if not self._compiled_graph:
            raise RuntimeError("No compiled graph available to invoke.")
        # Some LangGraph runnables expose `invoke` or are callable
        try:
            if hasattr(self._compiled_graph, "invoke"):
                return self._compiled_graph.invoke(state_dict)
            # fall back to calling directly
            return self._compiled_graph(state_dict)
        except Exception as e:
            logger.exception("Compiled graph invocation failed: %s", e)
            raise

    def _run_sequential(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run nodes sequentially using NodeManager.execute_all_sequential.
        Expects self.node_manager to be initialized.
        """
        if not self.node_manager:
            raise RuntimeError("NodeManager not initialized for sequential execution.")
        try:
            # Build HLDState-like object if available; but NodeManager.execute_all_sequential accepts HLDState
            from state.models import HLDState  # may raise if models not implemented
            # Try to construct HLDState from dict; if not possible, pass dict through and let nodes handle it
            try:
                state_obj = HLDState(**state_dict)
            except Exception:
                # Create minimal HLDState manually if constructor signature is incompatible
                state_obj = HLDState(pdf_path=state_dict.get("pdf_path"), requirement_name=state_dict.get("requirement_name") or "")
            # Execute all nodes sequentially and return final dict
            final_state = self.node_manager.execute_all_sequential(state_obj)
            return final_state.dict()
        except Exception as e:
            # If state.models not available or NodeManager fails, attempt a best-effort node loop
            logger.exception("Sequential execution via NodeManager failed: %s", e)
            # As a last resort, echo back the input state
            return state_dict

    # -------------------------
    # Public API
    # -------------------------
    def run(self, input_data: Any) -> WorkflowOutput:
        """
        Synchronous workflow execution.

        input_data: either a dict containing 'pdf_path' and optional 'config', or an object with those attributes.
        Returns WorkflowOutput dataclass with final state and metadata.
        """
        start = time.time()
        # Prepare initial state dict
        try:
            pdf_path = None
            config = None
            if isinstance(input_data, dict):
                pdf_path = input_data.get("pdf_path")
                config = input_data.get("config", {})
            else:
                pdf_path = getattr(input_data, "pdf_path", None)
                config = getattr(input_data, "config", {}) or {}

            # Create a conservative initial state dict
            init_state = {
                "pdf_path": pdf_path,
                "requirement_name": None,
                "config": config,
                "status": {},
                "errors": [],
                "warnings": [],
                # allow other fields to exist downstream
            }

            # Prefer create_initial_state if present
            try:
                from state.schema import create_initial_state  # if implemented
                state_obj = create_initial_state(pdf_path=pdf_path, config=config)
                state_dict = state_obj.dict()
            except Exception:
                # Fallback to building minimal HLDState dict
                try:
                    from state.models import HLDState
                    state_obj = HLDState(pdf_path=pdf_path, requirement_name=(pdf_path and __import__("pathlib").Path(pdf_path).stem) or "requirement", config=config)
                    state_dict = state_obj.dict()
                except Exception:
                    state_dict = init_state

            # Invoke the configured graph / executor
            logger.info("=" * 80)
            logger.info("STARTING WORKFLOW EXECUTION")
            logger.info("=" * 80)
            result_state_dict = self._graph_callable(state_dict)

            # Normalize to dict
            if hasattr(result_state_dict, "dict"):
                result_state = result_state_dict.dict()
            else:
                result_state = dict(result_state_dict or {})

            # Collect outputs (best-effort)
            output_paths = {}
            try:
                out = result_state.get("output") or {}
                # if out is object-like, attempt dict()
                if hasattr(out, "dict"):
                    out = out.dict()
                if isinstance(out, dict):
                    output_paths = out
                else:
                    output_paths = {"output_state": out}
            except Exception:
                output_paths = {}

            end = time.time()
            proc_time = end - start

            errors = result_state.get("errors") or []
            warnings = result_state.get("warnings") or []
            success = not bool(errors)
            
            logger.info("=" * 80)
            logger.info("WORKFLOW EXECUTION COMPLETED")
            logger.info("=" * 80)
            logger.info(f"✓ Success: {success}")
            logger.info(f"⏱ Total Duration: {proc_time:.2f}s")
            logger.info(f"Errors: {len(errors)}")
            logger.info(f"Warnings: {len(warnings)}")
            logger.info("=" * 80)

            return WorkflowOutput(
                success=success,
                state=result_state,
                output_paths=output_paths,
                processing_time=proc_time,
                errors=errors,
                warnings=warnings,
            )
        except Exception as exc:
            logger.exception("HLDWorkflow.run() failed: %s", exc)
            end = time.time()
            return WorkflowOutput(success=False, state={}, output_paths={}, processing_time=end - start, errors=[str(exc)], warnings=[])

    async def arun(self, input_data: Any) -> WorkflowOutput:
        """
        Async version of run(). Uses asyncio.to_thread to execute synchronous run() without blocking.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.run(input_data))
    async def stream(self, input_data: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream workflow execution node-by-node.
        Yields progress updates after each node executes.
        Designed for real-time UI integration (e.g., Streamlit progress display).
        """
        try:
            from state.models import HLDState
        except Exception:
            HLDState = None

        # Parse input
        pdf_path = input_data.get("pdf_path") if isinstance(input_data, dict) else getattr(input_data, "pdf_path", None)
        config = input_data.get("config", {}) if isinstance(input_data, dict) else getattr(input_data, "config", {}) or {}

        # Build initial state
        state_dict = {"pdf_path": pdf_path, "config": config}
        try:
            from state.schema import create_initial_state
            state_obj = create_initial_state(pdf_path=pdf_path, config=config)
        except Exception:
            if HLDState:
                from pathlib import Path
                state_obj = HLDState(
                    pdf_path=pdf_path, 
                    requirement_name=Path(pdf_path).stem if pdf_path else "requirement",
                    config=config  # ← ADD CONFIG HERE!
                )
            else:
                state_obj = type("StateMock", (), {"pdf_path": pdf_path, "config": config, "errors": [], "warnings": [], "status": {}})()
        
        # Log configuration being used
        logger.info(f"Streaming with config: {config}")
        yield {"type": "started", "timestamp": time.time(), "node": None, "message": "Workflow started"}
        
        logger.info("=" * 80)
        logger.info("STREAMING WORKFLOW EXECUTION")
        logger.info("=" * 80)

        # Check if we should use compiled graph for parallel/conditional workflows
        if self.workflow_type in ("parallel", "conditional") and self._compiled_graph:
            logger.info(f"Using compiled LangGraph for {self.workflow_type} execution")
            
            # For compiled graphs, we need to stream differently
            # LangGraph's astream yields node outputs, not individual node events
            try:
                if hasattr(self._compiled_graph, "astream"):
                    async for chunk in self._compiled_graph.astream(state_obj.dict()):
                        # LangGraph yields the full state after each node
                        yield {
                            "type": "state_update",
                            "state": chunk,
                            "timestamp": time.time(),
                        }
                else:
                    # Fallback: invoke synchronously and yield result
                    result = self._compiled_graph.invoke(state_obj.dict())
                    yield {
                        "type": "completed",
                        "timestamp": time.time(),
                        "state": result,
                        "message": f"{self.workflow_type.capitalize()} workflow completed"
                    }
                return
            except Exception as e:
                logger.exception(f"Compiled graph streaming failed: {e}")
                yield {"type": "error", "message": f"Compiled graph execution failed: {e}", "timestamp": time.time()}
                return

        # Fallback to sequential execution for sequential workflow or if compiled graph not available
        if not self.node_manager:
            yield {"type": "error", "message": "NodeManager not initialized"}
            return

        logger.info("Using sequential node-by-node execution")
        # Execute sequentially node-by-node
        for node_name in self.node_manager.get_execution_order():
            start = time.time()
            yield {"type": "node_start", "node": node_name, "timestamp": start}

            try:
                updated_state = self.node_manager.execute_node(node_name, state_obj)
                duration = round(time.time() - start, 2)
                status = updated_state.status.get(node_name).status if hasattr(updated_state, "status") else "unknown"

                yield {
                    "type": "node_complete",
                    "node": node_name,
                    "status": status,
                    "duration": duration,
                    "timestamp": time.time(),
                }

                # If any errors occurred, yield error info and break
                if hasattr(updated_state, "has_errors") and updated_state.has_errors():
                    yield {
                        "type": "error",
                        "node": node_name,
                        "errors": getattr(updated_state, "errors", []),
                        "timestamp": time.time(),
                    }
                    break

                state_obj = updated_state

            except Exception as e:
                yield {
                    "type": "error",
                    "node": node_name,
                    "message": str(e),
                    "timestamp": time.time(),
                }
                break

        yield {
            "type": "completed",
            "timestamp": time.time(),
            "node": None,
            "message": "Workflow completed",
            # include final state snapshot for UI consumption
            "state": getattr(state_obj, "dict", lambda: {} )(),
        }


# -------------------------
# Factory helpers
# -------------------------
def create_hld_workflow(workflow_type: str = "sequential") -> HLDWorkflow:
    return HLDWorkflow(workflow_type=workflow_type)


def create_sequential_workflow() -> HLDWorkflow:
    return create_hld_workflow("sequential")


def create_parallel_workflow() -> HLDWorkflow:
    return create_hld_workflow("parallel")


def create_conditional_workflow() -> HLDWorkflow:
    return create_hld_workflow("conditional")
