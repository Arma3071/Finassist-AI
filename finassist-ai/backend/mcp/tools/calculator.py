"""Calculator MCP tool: safely evaluates arithmetic expressions."""

import ast
import operator

from pydantic import BaseModel, Field

from backend.mcp.base import BaseTool, ToolError

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(description="A math expression, e.g. '(120.5 + 30) * 1.07'")


class CalculatorTool(BaseTool):
    """Evaluates arithmetic expressions safely (no eval/exec)."""

    name = "calculator"
    description = "Evaluate a mathematical expression involving +, -, *, /, %, ** and parentheses."
    args_schema = CalculatorArgs

    def _run(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
            return self._eval_node(tree.body)
        except (SyntaxError, ZeroDivisionError, TypeError, KeyError) as exc:
            raise ToolError(f"Could not evaluate expression: {exc}") from exc

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](self._eval_node(node.operand))
        raise ToolError(f"Unsupported expression element: {ast.dump(node)}")
