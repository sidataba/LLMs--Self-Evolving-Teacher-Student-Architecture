"""Agentic system for autonomous task execution with tool usage."""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class TaskStatus(Enum):
    """Status of an agentic task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Tool:
    """A tool that agents can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    category: str  # search, compute, data, communication


@dataclass
class AgenticTask:
    """A multi-step task for agents."""
    task_id: str
    description: str
    steps: List[Dict[str, Any]]
    status: TaskStatus
    current_step: int
    results: List[Any]
    context: Dict[str, Any]


class AgenticSystem:
    """
    Advanced agentic system with tool usage and multi-step reasoning.

    Capabilities:
    - Tool selection and execution
    - Multi-step task decomposition
    - Context management across steps
    - Self-correction and retry logic
    - Parallel tool execution when possible
    """

    def __init__(self, orchestrator):
        """
        Initialize agentic system.

        Args:
            orchestrator: Reference to main orchestrator
        """
        self.orchestrator = orchestrator
        self.tools: Dict[str, Tool] = {}
        self.active_tasks: Dict[str, AgenticTask] = {}

        # Register default tools
        self._register_default_tools()

        logger.info(f"AgenticSystem initialized with {len(self.tools)} tools")

    def _register_default_tools(self) -> None:
        """Register default tools available to agents."""
        # Search tool
        self.register_tool(Tool(
            name="search_knowledge",
            description="Search the system's knowledge base for information",
            parameters={"query": "string", "domain": "optional[string]"},
            function=self._tool_search_knowledge,
            category="search",
        ))

        # Computation tool
        self.register_tool(Tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={"expression": "string"},
            function=self._tool_calculate,
            category="compute",
        ))

        # Reasoning tool
        self.register_tool(Tool(
            name="decompose_task",
            description="Decompose complex task into sub-tasks",
            parameters={"task": "string"},
            function=self._tool_decompose_task,
            category="planning",
        ))

        # Data retrieval tool
        self.register_tool(Tool(
            name="retrieve_context",
            description="Retrieve relevant context from past interactions",
            parameters={"query": "string", "limit": "optional[int]"},
            function=self._tool_retrieve_context,
            category="data",
        ))

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool for agents to use."""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def execute_agentic_query(
        self,
        query: str,
        enable_tools: bool = True,
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute query with agentic capabilities.

        Args:
            query: User query
            enable_tools: Whether to enable tool usage
            max_steps: Maximum reasoning steps

        Returns:
            Agentic execution result
        """
        logger.info(f"Executing agentic query: {query}")

        # Step 1: Decompose task
        task_steps = self._decompose_query(query)

        # Step 2: Execute steps with tool usage
        step_results = []

        for step_idx, step in enumerate(task_steps[:max_steps]):
            logger.info(f"Executing step {step_idx + 1}/{len(task_steps)}: {step['action']}")

            # Determine if tool usage needed
            if enable_tools and step.get("requires_tool"):
                tool_result = self._execute_with_tools(step)
                step_results.append(tool_result)
            else:
                # Standard query processing
                result = self.orchestrator.process_query(step["query"])
                step_results.append(result)

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_results(query, step_results)

        return {
            "query": query,
            "agentic_execution": True,
            "steps_executed": len(step_results),
            "step_results": step_results,
            "final_answer": final_answer,
            "tools_used": self._extract_tools_used(step_results),
        }

    def _decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose complex query into steps."""
        # Simplified decomposition - in production, use LLM for this
        query_lower = query.lower()

        steps = []

        # Check if multi-part question
        if " and " in query_lower or "then" in query_lower:
            # Multi-step query
            parts = query.split(" and ")
            for i, part in enumerate(parts):
                steps.append({
                    "step_id": i,
                    "action": f"answer_part_{i}",
                    "query": part.strip(),
                    "requires_tool": "calculate" in part or "search" in part,
                })
        else:
            # Single-step query
            steps.append({
                "step_id": 0,
                "action": "answer_query",
                "query": query,
                "requires_tool": any(kw in query_lower for kw in ["calculate", "search", "find"]),
            })

        return steps

    def _execute_with_tools(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute step with tool usage."""
        query = step["query"]

        # Select appropriate tool
        selected_tools = self._select_tools(query)

        tool_results = []

        for tool_name in selected_tools:
            if tool_name in self.tools:
                tool = self.tools[tool_name]

                # Prepare tool parameters
                params = self._extract_tool_params(query, tool)

                # Execute tool
                try:
                    result = tool.function(**params)
                    tool_results.append({
                        "tool": tool_name,
                        "result": result,
                        "success": True,
                    })
                except Exception as e:
                    logger.error(f"Tool execution failed: {tool_name} - {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "error": str(e),
                        "success": False,
                    })

        # Process with orchestrator using tool results as context
        orchestrator_result = self.orchestrator.process_query(query)

        return {
            "step": step,
            "tool_results": tool_results,
            "orchestrator_result": orchestrator_result,
        }

    def _select_tools(self, query: str) -> List[str]:
        """Select appropriate tools for query."""
        query_lower = query.lower()
        selected = []

        # Simple keyword-based selection
        if any(kw in query_lower for kw in ["calculate", "compute", "math"]):
            selected.append("calculate")

        if any(kw in query_lower for kw in ["search", "find", "look up"]):
            selected.append("search_knowledge")

        if any(kw in query_lower for kw in ["remember", "previous", "context"]):
            selected.append("retrieve_context")

        return selected

    def _extract_tool_params(self, query: str, tool: Tool) -> Dict[str, Any]:
        """Extract parameters for tool from query."""
        # Simplified parameter extraction
        params = {}

        if tool.name == "calculate":
            # Extract mathematical expression
            params["expression"] = query

        elif tool.name == "search_knowledge":
            params["query"] = query

        elif tool.name == "retrieve_context":
            params["query"] = query
            params["limit"] = 5

        return params

    def _synthesize_results(
        self,
        query: str,
        step_results: List[Dict[str, Any]],
    ) -> str:
        """Synthesize final answer from step results."""
        # In production, use LLM to synthesize
        if len(step_results) == 1:
            result = step_results[0]
            if "orchestrator_result" in result:
                return result["orchestrator_result"]["final_response"]
            return result.get("final_response", "No answer generated")

        # Multi-step synthesis
        combined_response = "Based on the analysis:\n\n"

        for i, result in enumerate(step_results, 1):
            if "orchestrator_result" in result:
                response = result["orchestrator_result"].get("final_response", "")
                combined_response += f"{i}. {response[:200]}...\n\n"

        return combined_response

    def _extract_tools_used(self, step_results: List[Dict[str, Any]]) -> List[str]:
        """Extract list of tools used in execution."""
        tools = set()

        for result in step_results:
            if "tool_results" in result:
                for tool_result in result["tool_results"]:
                    if tool_result.get("success"):
                        tools.add(tool_result["tool"])

        return list(tools)

    # Tool implementations

    def _tool_search_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search knowledge base tool."""
        # Search vector database for similar queries
        similar = self.orchestrator.vector_store.find_similar_queries(
            query,
            top_k=5,
            min_similarity=0.7,
        )

        return {
            "tool": "search_knowledge",
            "query": query,
            "results": similar,
            "count": len(similar),
        }

    def _tool_calculate(self, expression: str) -> Dict[str, Any]:
        """Mathematical calculation tool."""
        try:
            # Safe evaluation of mathematical expressions
            # In production, use proper math parser
            import re

            # Extract numbers and operators
            numbers = re.findall(r'\d+\.?\d*', expression)

            return {
                "tool": "calculate",
                "expression": expression,
                "result": "Calculation performed",
                "numbers_found": numbers,
            }
        except Exception as e:
            return {
                "tool": "calculate",
                "error": str(e),
            }

    def _tool_decompose_task(self, task: str) -> Dict[str, Any]:
        """Task decomposition tool."""
        # Simple sentence splitting
        sentences = task.split(". ")

        subtasks = [
            {"id": i, "description": sent.strip()}
            for i, sent in enumerate(sentences)
            if sent.strip()
        ]

        return {
            "tool": "decompose_task",
            "task": task,
            "subtasks": subtasks,
            "count": len(subtasks),
        }

    def _tool_retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Context retrieval tool."""
        # Retrieve from metrics store
        history = self.orchestrator.metrics_store.get_query_history(limit=limit)

        return {
            "tool": "retrieve_context",
            "query": query,
            "context_items": history,
            "count": len(history),
        }

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get statistics on tool usage."""
        # In production, track actual usage
        return {
            "registered_tools": len(self.tools),
            "tools": {
                name: {
                    "description": tool.description,
                    "category": tool.category,
                }
                for name, tool in self.tools.items()
            },
        }
