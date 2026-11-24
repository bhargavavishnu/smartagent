"""Safe AST-based math expression evaluator tool (never uses raw eval)."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": pow,
    "min": min,
    "max": max,
}

_ALLOWED_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTANTS:
        return _ALLOWED_CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent magnitude exceeds safe limit (1000).")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCS or node.keywords:
            raise ValueError(f"Disallowed or unsupported function '{func_name}'.")
        args = [_eval_node(arg) for arg in node.args]
        return float(_ALLOWED_FUNCS[func_name](*args))
    raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression safely using an AST whitelist."""
    cleaned = expression.strip().replace("^", "**").replace(",", "")
    if not cleaned:
        return {"ok": False, "error": "Expression is empty."}
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_node(tree)
        formatted = int(value) if value.is_integer() else round(value, 6)
        return {"ok": True, "expression": expression, "result": formatted}
    except ZeroDivisionError:
        return {"ok": False, "expression": expression, "error": "Division by zero."}
    except Exception as exc:
        return {"ok": False, "expression": expression, "error": str(exc)}
