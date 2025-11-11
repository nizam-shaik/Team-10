from __future__ import annotations
"""
LangGraph Workflow Graph Definition
Defines the graph structure and execution flow for HLD generation
Located at root level for easy access and graph visualization
"""

# TODO: Import StateGraph and END from langgraph.graph
# TODO: Import NodeManager for access to all workflow nodes
# TODO: Create WorkflowGraph class with __init__ method
# TODO: Implement create_sequential_workflow_graph() method
#       - Create StateGraph with Dict[str, Any] as state type
#       - Add all node runnables to the graph
#       - Set entry point to "pdf_extraction"
#       - Add sequential edges: pdf_extraction -> auth_integrations -> domain_api_design
#         -> behavior_quality -> diagram_generation -> output_composition -> END
#       - Compile and return the graph
# TODO: Implement create_parallel_workflow_graph() method
#       - Create similar structure to sequential (for compatibility)
#       - Add optimized execution order
#       - Compile and return the graph
# TODO: Implement create_conditional_workflow_graph() method
#       - Create StateGraph and add all nodes
#       - Define routing functions for each node (route_after_pdf, etc.)
#       - Route based on _node_result flag in state
#       - Allow retry logic if node execution fails
#       - Add conditional edges with routing functions
#       - Add final edge from output_composition to END
#       - Compile and return the graph
# TODO: Implement create_graph(graph_type: str) factory method
#       - Validate graph_type is one of: sequential, parallel, conditional
#       - Raise ValueError for unknown types
#       - Return appropriate workflow graph
# TODO: Implement get_execution_order() -> list
#       - Return list of node names in execution sequence
# TODO: Implement get_nodes_info() -> Dict[str, Dict[str, Any]]
#       - Retrieve detailed information about all nodes
# TODO: Implement visualize() -> str
#       - Return ASCII representation of graph flow
#       - Example: "pdf_extraction -> auth_integrations -> ... -> END"
# TODO: Create module-level convenience functions
#       - create_workflow_graph() -> creates sequential graph
#       - create_parallel_workflow_graph() -> creates parallel graph
#       - create_conditional_workflow_graph() -> creates conditional graph
# TODO: Add graph compilation and validation
# TODO: Handle state type definition and persistence
# TODO: Consider graph visualization and debugging capabilities
"""
LangGraph Workflow Graph Definition
Defines the graph structure and execution flow for HLD generation
Located at root level for easy access and graph visualization
"""

import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, END
from nodes.node_manager import NodeManager

logger = logging.getLogger(__name__)


class WorkflowGraph:
    """
    Defines and compiles LangGraph workflow graphs (sequential, parallel, conditional)
    for HLD generation.
    """

    def __init__(self, graph_type: str = "sequential"):
        self.graph_type = graph_type.lower().strip()
        self.node_manager = NodeManager()
        self.graph = None

    # -------------------------------------------------------------------------
    # Sequential Graph
    # -------------------------------------------------------------------------
    def create_sequential_workflow_graph(self):
        """Create a simple sequential LangGraph workflow."""
        logger.info("[WorkflowGraph] Building sequential workflow graph.")
        runnables = self.node_manager.get_node_runnables()
        workflow = StateGraph(Dict[str, Any])

        # Add nodes
        for name, runnable in runnables.items():
            workflow.add_node(name, runnable)

        # Set entry point
        workflow.set_entry_point("pdf_extraction")

        # Add sequential edges
        workflow.add_edge("pdf_extraction", "auth_integrations")
        workflow.add_edge("auth_integrations", "domain_api_design")
        workflow.add_edge("domain_api_design", "behavior_quality")
        workflow.add_edge("behavior_quality", "diagram_generation")
        workflow.add_edge("diagram_generation", "output_composition")
        workflow.add_edge("output_composition", END)

        compiled = workflow.compile()
        self.graph = compiled
        return compiled

    # -------------------------------------------------------------------------
    # Parallel Graph
    # -------------------------------------------------------------------------
    def create_parallel_workflow_graph(self):
        """
        Parallel variant for potential concurrency (future extension).
        Currently mirrors sequential execution for compatibility.
        """
        logger.info("[WorkflowGraph] Building parallel workflow graph.")
        return self.create_sequential_workflow_graph()

    # -------------------------------------------------------------------------
    # Conditional Graph
    # -------------------------------------------------------------------------
    def create_conditional_workflow_graph(self):
        """
        Conditional graph with routing based on node results.
        Nodes can decide which path to take next dynamically.
        """
        logger.info("[WorkflowGraph] Building conditional workflow graph.")
        runnables = self.node_manager.get_node_runnables()
        workflow = StateGraph(Dict[str, Any])

        for name, runnable in runnables.items():
            workflow.add_node(name, runnable)

        workflow.set_entry_point("pdf_extraction")

        # Routing functions
        def route_after_pdf(state: Dict[str, Any]):
            if state.get("skip_auth"):
                return "domain_api_design"
            return "auth_integrations"

        def route_after_domain(state: Dict[str, Any]):
            if state.get("_behavior_skipped"):
                return "diagram_generation"
            return "behavior_quality"

        # Edges with conditions
        workflow.add_conditional_edges("pdf_extraction", route_after_pdf)
        workflow.add_edge("auth_integrations", "domain_api_design")
        workflow.add_conditional_edges("domain_api_design", route_after_domain)
        workflow.add_edge("behavior_quality", "diagram_generation")
        workflow.add_edge("diagram_generation", "output_composition")
        workflow.add_edge("output_composition", END)

        compiled = workflow.compile()
        self.graph = compiled
        return compiled

    # -------------------------------------------------------------------------
    # Factory Method
    # -------------------------------------------------------------------------
    def create_graph(self, graph_type: str = None):
        """Factory method to build a workflow graph by type."""
        gtype = (graph_type or self.graph_type).lower()
        if gtype == "sequential":
            return self.create_sequential_workflow_graph()
        elif gtype == "parallel":
            return self.create_parallel_workflow_graph()
        elif gtype == "conditional":
            return self.create_conditional_workflow_graph()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    # -------------------------------------------------------------------------
    # Metadata & Visualization
    # -------------------------------------------------------------------------
    def get_execution_order(self):
        """Return the standard node execution order."""
        return self.node_manager.get_execution_order()

    def get_nodes_info(self):
        """Return detailed metadata about each node."""
        return self.node_manager.get_nodes_info()

    def visualize(self) -> str:
        """
        Return a simple ASCII visualization of the workflow flow.
        Example: pdf_extraction -> auth_integrations -> ... -> END
        """
        order = self.get_execution_order()
        return " -> ".join(order + ["END"])


# -------------------------------------------------------------------------
# Module-Level Convenience Functions
# -------------------------------------------------------------------------
def create_workflow_graph() -> Any:
    """Default sequential workflow graph."""
    return WorkflowGraph("sequential").create_graph("sequential")


def create_parallel_workflow_graph() -> Any:
    """Parallel workflow graph (for concurrency-ready execution)."""
    return WorkflowGraph("parallel").create_graph("parallel")


def create_conditional_workflow_graph() -> Any:
    """Conditional workflow graph (dynamic routing)."""
    return WorkflowGraph("conditional").create_graph("conditional")
