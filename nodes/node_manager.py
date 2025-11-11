from __future__ import annotations
"""
Node Manager - Central orchestration of all workflow nodes
"""

# TODO: Implement NodeManager class
# TODO: Import all node classes (PDFExtractionNode, AuthIntegrationsNode, etc.)
# TODO: Implement __init__ method
#       - Initialize all 6 nodes with their agents
#       - Create nodes dictionary mapping name to node instance
#       - Define execution order list
# TODO: Implement get_node_runnables() -> Dict[str, Runnable]
#       - Return dictionary of node name to runnable
#       - Each runnable wraps node.execute() for LangGraph
#       - Handle state dict conversion
# TODO: Implement get_execution_order() -> list
#       - Return list of node names in execution sequence
#       - Order: pdf_extraction, auth_integrations, domain_api_design,
#         behavior_quality, diagram_generation, output_composition
# TODO: Implement get_nodes_info() -> Dict[str, Dict[str, Any]]
#       - Return detailed info about each node
#       - Include: name, description, inputs, outputs, status
# TODO: Implement get_node(name: str) -> BaseNode
#       - Retrieve specific node by name
#       - Raise error if node not found
# TODO: Implement node execution methods
#       - execute_node(name: str, state: HLDState) -> HLDState
#       - execute_all_sequential(state: HLDState) -> HLDState
# TODO: Add validation
#       - Validate all nodes are initialized
#       - Check node dependencies
#       - Verify state transitions
# TODO: Consider monitoring and logging
#       - Track node execution time
#       - Log state transitions
#       - Monitor error conditions
"""
Node Manager - Central orchestration of all workflow nodes
"""

import logging
import time
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda

from .pdf_extraction_node import PDFExtractionNode
from .auth_integrations_node import AuthIntegrationsNode
from .domain_api_node import DomainAPINode
from .behavior_quality_node import BehaviorQualityNode
from .diagram_generation_node import DiagramGenerationNode
from .output_composition_node import OutputCompositionNode

from state.models import HLDState
from .base_node import BaseNode

logger = logging.getLogger(__name__)


class NodeManager:
    """
    Orchestrates all workflow nodes used in the LangGraph pipeline.
    Handles initialization, execution order, and LangGraph runnable integration.
    """

    def __init__(self):
        logger.info("[NodeManager] Initializing all workflow nodes...")

        # Initialize all 6 nodes
        self.pdf_extraction = PDFExtractionNode()
        self.auth_integrations = AuthIntegrationsNode()
        self.domain_api_design = DomainAPINode()
        self.behavior_quality = BehaviorQualityNode()
        self.diagram_generation = DiagramGenerationNode()
        self.output_composition = OutputCompositionNode()

        # Node registry
        self.nodes: Dict[str, BaseNode] = {
            "pdf_extraction": self.pdf_extraction,
            "auth_integrations": self.auth_integrations,
            "domain_api_design": self.domain_api_design,
            "behavior_quality": self.behavior_quality,
            "diagram_generation": self.diagram_generation,
            "output_composition": self.output_composition,
        }

        # Execution order for sequential workflows
        self.execution_order: List[str] = [
            "pdf_extraction",
            "auth_integrations",
            "domain_api_design",
            "behavior_quality",
            "diagram_generation",
            "output_composition",
        ]

        logger.info(f"[NodeManager] Registered nodes: {list(self.nodes.keys())}")

    # -------------------------------------------------------------------------
    # Runnable Integration
    # -------------------------------------------------------------------------
    def get_node_runnables(self) -> Dict[str, RunnableLambda]:
        """
        Return dictionary mapping node names to LangGraph-compatible runnables.
        Each runnable wraps node.execute() with state conversion.
        """
        logger.info("[NodeManager] Building node runnables for LangGraph integration.")

        def _wrap(node: BaseNode):
            def _runner(state_dict: Dict[str, Any]):
                try:
                    state = HLDState(**state_dict)
                    updated = node.execute(state)
                    return updated.dict()
                except Exception as e:
                    logger.exception(f"[NodeManager] Error executing node {node.name}: {e}")
                    raise
            return RunnableLambda(_runner)

        return {name: _wrap(node) for name, node in self.nodes.items()}

    # -------------------------------------------------------------------------
    # Node Accessors
    # -------------------------------------------------------------------------
    def get_execution_order(self) -> List[str]:
        """Return the standard sequential execution order."""
        return self.execution_order

    def get_nodes_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Return detailed metadata about all registered nodes:
        name, description, inputs, outputs, and status.
        """
        info = {}
        for name, node in self.nodes.items():
            info[name] = {
                "name": node.name,
                "description": node.__doc__.strip() if node.__doc__ else "No description.",
                "inputs": "HLDState",
                "outputs": "HLDState",
                "status": "initialized",
            }
        return info

    def get_node(self, name: str) -> BaseNode:
        """
        Retrieve a node by name.
        Raises ValueError if node is not found.
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in NodeManager.")
        return self.nodes[name]

    # -------------------------------------------------------------------------
    # Execution Utilities
    # -------------------------------------------------------------------------
    def execute_node(self, name: str, state: HLDState) -> HLDState:
        """
        Execute a specific node by name on the given state.
        Returns the updated HLDState.
        """
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' not found in NodeManager.")

        node = self.nodes[name]
        logger.info(f"[NodeManager] Executing node: {name}")

        start = time.time()
        try:
            updated_state = node.execute(state)
            duration = round(time.time() - start, 2)
            logger.info(f"[NodeManager] Node '{name}' completed in {duration}s with status: "
                        f"{updated_state.status.get(name).status}")
            return updated_state
        except Exception as e:
            logger.exception(f"[NodeManager] Node '{name}' execution failed: {e}")
            state.add_error(str(e))
            state.set_status(name, "failed", str(e))
            return state

    def execute_all_sequential(self, state: HLDState) -> HLDState:
        """
        Execute all nodes in the defined order sequentially.
        Returns the final HLDState after full pipeline execution.
        """
        logger.info("[NodeManager] Executing all nodes sequentially...")
        for node_name in self.execution_order:
            state = self.execute_node(node_name, state)
            if state.has_errors():
                logger.warning(f"[NodeManager] Stopping execution: errors detected after '{node_name}'.")
                break
        logger.info("[NodeManager] Sequential node execution completed.")
        return state

    # -------------------------------------------------------------------------
    # Validation and Monitoring
    # -------------------------------------------------------------------------
    def validate_nodes(self) -> bool:
        """
        Validate that all nodes are initialized and ready.
        Returns True if all nodes are valid.
        """
        valid = all(isinstance(n, BaseNode) for n in self.nodes.values())
        if not valid:
            logger.error("[NodeManager] One or more nodes are invalid.")
        else:
            logger.info("[NodeManager] All nodes validated successfully.")
        return valid
