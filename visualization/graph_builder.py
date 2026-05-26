"""
GraphBuilder — converts AgentState history (or SystemTrace) into
execution graphs with Mermaid and optional networkx output.

Mermaid is zero-dependency.
networkx is optional; import fails gracefully.
"""
from __future__ import annotations

from pydantic import BaseModel
from core.state import AgentState

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ── Data model ────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    label: str
    node_type: str   # "input" | "agent" | "decision" | "terminal"
    iteration: int | None = None
    confidence: float | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str = ""


class ExecutionGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    # ── Mermaid rendering ─────────────────────────────────────────────────────

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]

        for node in self.nodes:
            shape = self._mermaid_shape(node.node_type)
            label = node.label
            if node.confidence is not None:
                label += f"\\nconf={node.confidence:.2f}"
            safe_id = node.id.replace(":", "_").replace("-", "_")
            lines.append(f'    {safe_id}{shape[0]}"{label}"{shape[1]}')

        for edge in self.edges:
            src = edge.source.replace(":", "_").replace("-", "_")
            tgt = edge.target.replace(":", "_").replace("-", "_")
            arrow = f" -->|{edge.label}|" if edge.label else " -->"
            lines.append(f"    {src}{arrow} {tgt}")

        return "\n".join(lines)

    @staticmethod
    def _mermaid_shape(node_type: str) -> tuple[str, str]:
        return {
            "input":    ("[", "]"),
            "agent":    ("(", ")"),
            "decision": ("{", "}"),
            "terminal": ("([", "])"),
        }.get(node_type, ("(", ")"))

    # ── networkx rendering ────────────────────────────────────────────────────

    def to_networkx(self):
        if not _HAS_NX:
            raise ImportError(
                "networkx is not installed. Run: pip install networkx"
            )
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(
                node.id,
                label=node.label,
                node_type=node.node_type,
                iteration=node.iteration,
                confidence=node.confidence,
            )
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)
        return G


# ── Builder ───────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Constructs an ExecutionGraph from an AgentState's history.

    Structure:
      User Input → Goal Interpreter → [per-iteration agent chain] → Terminal
    """

    @classmethod
    def from_state(cls, state: AgentState) -> ExecutionGraph:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        # Source node
        nodes.append(GraphNode(id="user_input", label="User Input", node_type="input"))
        nodes.append(GraphNode(id="goal_interpreter", label="Goal Interpreter",
                               node_type="agent"))
        edges.append(GraphEdge(source="user_input", target="goal_interpreter"))

        prev_node = "goal_interpreter"

        for record in state.history:
            i = record.iteration
            prefix = f"i{i}"

            # Determine which agents ran (infer from which fields are populated)
            agents_ran: list[tuple[str, str]] = []
            if record.plan is not None:
                agents_ran.append((f"{prefix}_planner", "Planner"))
            agents_ran.append((f"{prefix}_executor", "Executor"))
            agents_ran.append((f"{prefix}_critic", "Critic"))
            agents_ran.append((f"{prefix}_reflection", "Reflection"))
            agents_ran.append((f"{prefix}_memory", "Memory Manager"))

            for node_id, label in agents_ran:
                conf = record.agent_confidences.get(label.lower().replace(" ", "_"))
                nodes.append(GraphNode(
                    id=node_id, label=f"{label}\\n[iter {i}]",
                    node_type="agent", iteration=i, confidence=conf,
                ))
                edges.append(GraphEdge(source=prev_node, target=node_id))
                prev_node = node_id

            # Decision node
            action = record.control_decision.action if record.control_decision else "?"
            decision_id = f"{prefix}_controller"
            nodes.append(GraphNode(
                id=decision_id,
                label=f"Controller\\n[iter {i}]\\n→ {action}",
                node_type="decision", iteration=i,
            ))
            edges.append(GraphEdge(source=prev_node, target=decision_id))

            # Branch based on action
            if action == "terminate":
                terminal_id = f"{prefix}_terminal"
                nodes.append(GraphNode(
                    id=terminal_id, label="Terminate", node_type="terminal", iteration=i
                ))
                edges.append(GraphEdge(source=decision_id, target=terminal_id,
                                       label="terminate"))
                prev_node = terminal_id
            elif action == "replan":
                next_planner = f"i{i+1}_planner"
                # Edge will connect when next iteration is processed
                edges.append(GraphEdge(source=decision_id, target=next_planner,
                                       label="replan"))
                prev_node = decision_id
            elif action == "specialist":
                spec_type = (
                    record.control_decision.specialist_type or "?"
                    if record.control_decision else "?"
                )
                spec_id = f"{prefix}_specialist_{spec_type}"
                nodes.append(GraphNode(
                    id=spec_id, label=f"Specialist\\n[{spec_type}]",
                    node_type="agent", iteration=i,
                ))
                edges.append(GraphEdge(source=decision_id, target=spec_id,
                                       label="specialist"))
                prev_node = spec_id
            else:  # continue
                prev_node = decision_id

        # Ensure the graph ends with a terminal if the last action was terminate
        last = state.history[-1] if state.history else None
        if not last or (last.control_decision and
                        last.control_decision.action != "terminate"):
            nodes.append(GraphNode(id="end", label="End", node_type="terminal"))
            edges.append(GraphEdge(source=prev_node, target="end"))

        return ExecutionGraph(nodes=nodes, edges=edges)
