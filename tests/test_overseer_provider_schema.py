from __future__ import annotations

from navirl.overseer.provider import _strict_json_schema_for_codex


def _assert_object_nodes_are_strict(node):
    if isinstance(node, dict):
        node_type = node.get("type")
        is_object = (
            node_type == "object"
            or (isinstance(node_type, list) and "object" in node_type)
            or ("properties" in node and "type" not in node)
        )
        if is_object:
            assert node.get("additionalProperties") is False
            props = node.get("properties")
            assert isinstance(props, dict)
            assert node.get("required") == [str(k) for k in props]
        for value in node.values():
            _assert_object_nodes_are_strict(value)
    elif isinstance(node, list):
        for value in node:
            _assert_object_nodes_are_strict(value)


def test_codex_schema_strict_object_nodes_closed_and_fully_required():
    schema = {
        "type": "object",
        "required": ["overall_pass", "violations"],
        "properties": {
            "overall_pass": {"type": "boolean"},
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "evidence"],
                    "properties": {
                        "type": {"type": "string"},
                        "evidence": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                },
            },
            "notes": {"type": "string"},
        },
    }

    strict = _strict_json_schema_for_codex(schema)
    _assert_object_nodes_are_strict(strict)
