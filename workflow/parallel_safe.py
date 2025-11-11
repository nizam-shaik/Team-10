"""
Safe Parallel Workflow Implementation for LangGraph
"""

import logging
from typing import Dict, Any, Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from operator import add
import operator

from state.models import HLDState
from nodes import NodeManager

logger = logging.getLogger(__name__)

# Custom reducer to merge dictionaries
def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with right values taking precedence"""
    if not left:
        return right
    if not right:
        return left
    result = left.copy()
    result.update(right)
    return result

# Custom reducer to merge lists of dictionaries (for integrations)
def merge_list_items(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge two lists, avoiding duplicates based on 'system' key"""
    if not left:
        return right
    if not right:
        return left
    result = left.copy()
    existing_systems = {item.get("system") for item in left if isinstance(item, dict)}
    for item in right:
        if isinstance(item, dict) and item.get("system") not in existing_systems:
            result.append(item)
    return result

# Define state schema with proper annotations for parallel updates
class ParallelWorkflowState(TypedDict, total=False):
    """
    State schema that supports parallel updates.
    All fields use Annotated with reducers to handle concurrent updates.
    """
    # Core fields (can be updated by any node)
    pdf_path: str
    requirement_name: str
    config: Dict[str, Any]
    
    # Fields updated by specific nodes - use merge_dicts for dictionary fields
    extracted: Annotated[Dict[str, Any], merge_dicts]
    authentication: Annotated[Dict[str, Any], merge_dicts]
    integrations: Annotated[List[Dict[str, Any]], merge_list_items]  # List of integration dicts
    domain: Annotated[Dict[str, Any], merge_dicts]
    behavior: Annotated[Dict[str, Any], merge_dicts]
    diagrams: Annotated[Dict[str, Any], merge_dicts]
    output: Annotated[Dict[str, Any], merge_dicts]
    
    # Status tracking - merge status updates from parallel nodes
    status: Annotated[Dict[str, Any], merge_dicts]
    
    # Parallel-safe accumulator fields - concatenate lists
    errors: Annotated[List[str], add]
    warnings: Annotated[List[str], add]

def create_safe_parallel_workflow() -> Runnable:
    """
    Create a truly parallel workflow using LangGraph's recommended patterns
    This avoids the INVALID_CONCURRENT_GRAPH_UPDATE error
    """

    node_manager = NodeManager()
    
    # Create state graph with proper typed state schema
    workflow = StateGraph(ParallelWorkflowState)
    
    # Add PDF extraction node (runs first, sequentially)
    node_runnables = node_manager.get_node_runnables()
    workflow.add_node("pdf_extraction", node_runnables["pdf_extraction"])

    # Get individual nodes for parallel execution
    auth_node = node_manager.get_node("auth_integrations")
    domain_node = node_manager.get_node("domain_api_design")
    behavior_node = node_manager.get_node("behavior_quality")

    # Create parallel processing nodes that return properly typed state updates
    def auth_parallel_node(state: ParallelWorkflowState) -> ParallelWorkflowState:
        """Auth node that properly uses node.execute() for consistency"""
        hld_state = HLDState(**state)
        updated_state = auth_node.execute(hld_state)
        
        # Return only fields this node updates (LangGraph will merge)
        return {
            "authentication": updated_state.authentication,
            "integrations": [i.dict() for i in (updated_state.integrations or [])],
            "status": updated_state.status,
            "errors": updated_state.errors,
            "warnings": updated_state.warnings
        }
    
    def domain_parallel_node(state: ParallelWorkflowState) -> ParallelWorkflowState:
        """Domain node that properly uses node.execute() for consistency"""
        hld_state = HLDState(**state)
        updated_state = domain_node.execute(hld_state)
        
        # Return only fields this node updates (LangGraph will merge)
        return {
            "domain": updated_state.domain,
            "status": updated_state.status,
            "errors": updated_state.errors,
            "warnings": updated_state.warnings
        }
    
    def behavior_parallel_node(state: ParallelWorkflowState) -> ParallelWorkflowState:
        """Behavior node that properly uses node.execute() for consistency"""
        hld_state = HLDState(**state)
        updated_state = behavior_node.execute(hld_state)
        
        # Return only fields this node updates (LangGraph will merge)
        return {
            "behavior": updated_state.behavior,
            "status": updated_state.status,
            "errors": updated_state.errors,
            "warnings": updated_state.warnings
        }
    
    def parallel_coordinator(state: ParallelWorkflowState) -> ParallelWorkflowState:
        """Coordinate the results from parallel execution"""
        # All merging is handled by Annotated reducers
        # This node just verifies completion and passes state through
        logger.info("[ParallelCoordinator] All parallel nodes completed successfully")
        # Must return at least one field - return empty status to satisfy LangGraph
        return {"status": {}}
    
    # Add nodes to workflow
    workflow.add_node("auth_parallel", auth_parallel_node)
    workflow.add_node("domain_parallel", domain_parallel_node)
    workflow.add_node("behavior_parallel", behavior_parallel_node)
    workflow.add_node("parallel_coordinator", parallel_coordinator)
    workflow.add_node("diagram_generation", node_runnables["diagram_generation"])
    workflow.add_node("output_composition", node_runnables["output_composition"])
    
    # Set entry point
    workflow.set_entry_point("pdf_extraction")
    
    # Sequential: PDF extraction first
    workflow.add_edge("pdf_extraction", "auth_parallel")
    workflow.add_edge("pdf_extraction", "domain_parallel")
    workflow.add_edge("pdf_extraction", "behavior_parallel")
    
    # All parallel nodes feed into coordinator
    workflow.add_edge("auth_parallel", "parallel_coordinator")
    workflow.add_edge("domain_parallel", "parallel_coordinator")
    workflow.add_edge("behavior_parallel", "parallel_coordinator")
    
    # Sequential: Continue after parallel processing
    workflow.add_edge("parallel_coordinator", "diagram_generation")
    workflow.add_edge("diagram_generation", "output_composition")
    workflow.add_edge("output_composition", END)
    
    return workflow.compile()

def create_batch_parallel_workflow() -> Runnable:
    """
    Alternative parallel approach using batch processing
    """

    node_manager = NodeManager()
    workflow = StateGraph(Dict[str, Any])

    # Single batch processing node that handles multiple operations
    def batch_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process auth, domain, and behavior analysis in a single node"""
        hld_state = HLDState(**state)

        # Get individual nodes
        auth_node = node_manager.get_node("auth_integrations")
        domain_node = node_manager.get_node("domain_api_design")
        behavior_node = node_manager.get_node("behavior_quality")

        # Process all three analyses
        auth_result = auth_node.agent.process(hld_state)
        domain_result = domain_node.agent.process(hld_state)
        behavior_result = behavior_node.agent.process(hld_state)
        
        # Update state with all results
        updated_state = hld_state.dict()
        updated_state["_batch_results"] = {
            "auth": auth_result,
            "domain": domain_result,
            "behavior": behavior_result
        }
        
        return updated_state
    
    # Add nodes
    node_runnables = node_manager.get_node_runnables()
    workflow.add_node("pdf_extraction", node_runnables["pdf_extraction"])
    workflow.add_node("batch_analysis", batch_analysis_node)
    workflow.add_node("diagram_generation", node_runnables["diagram_generation"])
    workflow.add_node("output_composition", node_runnables["output_composition"])
    
    # Set entry point and edges
    workflow.set_entry_point("pdf_extraction")
    workflow.add_edge("pdf_extraction", "batch_analysis")
    workflow.add_edge("batch_analysis", "diagram_generation")
    workflow.add_edge("diagram_generation", "output_composition")
    workflow.add_edge("output_composition", END)
    
    return workflow.compile()

def create_conditional_workflow() -> Runnable:
    """
    Create a conditional workflow with dynamic routing based on content analysis.
    Routes differently based on what's detected in the extracted requirements.
    """
    node_manager = NodeManager()
    workflow = StateGraph(Dict[str, Any])
    
    # Add all nodes
    node_runnables = node_manager.get_node_runnables()
    workflow.add_node("pdf_extraction", node_runnables["pdf_extraction"])
    workflow.add_node("auth_integrations", node_runnables["auth_integrations"])
    workflow.add_node("domain_api_design", node_runnables["domain_api_design"])
    workflow.add_node("behavior_quality", node_runnables["behavior_quality"])
    workflow.add_node("diagram_generation", node_runnables["diagram_generation"])
    workflow.add_node("output_composition", node_runnables["output_composition"])
    
    # Set entry point
    workflow.set_entry_point("pdf_extraction")
    
    # Always go to auth after PDF extraction
    workflow.add_edge("pdf_extraction", "auth_integrations")
    
    # Conditional routing after auth: decide if we need domain analysis
    def route_after_auth(state: Dict[str, Any]) -> str:
        """Route based on content analysis"""
        extracted_md = ""
        try:
            extracted = state.get("extracted", {})
            if isinstance(extracted, dict):
                extracted_md = extracted.get("markdown", "").lower()
            else:
                extracted_md = str(extracted).lower()
        except Exception:
            extracted_md = ""
        
        # Check for API/domain keywords
        has_api_content = any(keyword in extracted_md for keyword in [
            "api", "endpoint", "rest", "service", "entity", "database", "schema"
        ])
        
        if has_api_content:
            logger.info("[ConditionalWorkflow] API content detected → routing to domain_api_design")
            return "domain_api_design"
        else:
            logger.info("[ConditionalWorkflow] No API content → skipping to behavior_quality")
            return "behavior_quality"
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "auth_integrations",
        route_after_auth,
        {
            "domain_api_design": "domain_api_design",
            "behavior_quality": "behavior_quality"
        }
    )
    
    # Domain analysis leads to behavior
    workflow.add_edge("domain_api_design", "behavior_quality")
    
    # Conditional routing after behavior: decide if we need diagrams
    def route_after_behavior(state: Dict[str, Any]) -> str:
        """Route based on whether we have domain data for diagrams"""
        domain = state.get("domain")
        
        # If we have domain entities, generate diagrams
        if domain and isinstance(domain, dict):
            entities = domain.get("entities", [])
            if entities and len(entities) > 0:
                logger.info("[ConditionalWorkflow] Domain entities found → generating diagrams")
                return "diagram_generation"
        
        # Skip diagrams if no domain data
        logger.info("[ConditionalWorkflow] No domain entities → skipping diagrams")
        return "output_composition"
    
    workflow.add_conditional_edges(
        "behavior_quality",
        route_after_behavior,
        {
            "diagram_generation": "diagram_generation",
            "output_composition": "output_composition"
        }
    )
    
    # Diagrams always lead to output
    workflow.add_edge("diagram_generation", "output_composition")
    workflow.add_edge("output_composition", END)
    
    return workflow.compile()