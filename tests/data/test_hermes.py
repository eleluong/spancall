"""Unit tests for the Hermes function-calling V1 loader (all 5 configs)."""

from __future__ import annotations

import json

import pytest

from spancall.data.schema import GroundTruthCall, Split, ToolDefinition, UnifiedSample
from spancall.data.sources import HERMES_CONFIGS, HermesLoader
from spancall.data.partition import partition_schema_disjoint

# ---------------------------------------------------------------------------
# Fixture rows — one per config, HF-shape, no network required
# ---------------------------------------------------------------------------

TOOLS_JSON = """[{"type": "function", "function": {
    "name": "get_camera_live_feed",
    "description": "Retrieves the live feed from a specified security camera.",
    "parameters": {"type": "object", "properties": {
        "camera_id": {"type": "string", "description": "The unique identifier for the camera."},
        "stream_quality": {"type": "string", "enum": ["720p", "1080p", "4k"]}
    }, "required": ["camera_id"]}
}}]"""

SINGLETURN_ROW = {
    "id": "fixture-single-1",
    "conversations": [
        {"from": "system", "value": f"You are a function calling AI model.\n<tools>\n{TOOLS_JSON}\n</tools>"},
        {"from": "human", "value": "Show me the 4k feed of camera cam-1."},
        {"from": "gpt", "value": '<tool_call>{"name": "get_camera_live_feed", "arguments": {"camera_id": "cam-1", "stream_quality": "4k"}}</tool_call>'},
    ],
    "tools": TOOLS_JSON,
    "category": "Security", "subcategory": "Cameras", "task": "capture",
}

MULTITURN_ROW = {
    "id": "fixture-multi-1",
    "conversations": [
        {"from": "system", "value": f"You are a function calling AI model.\n<tools>\n{TOOLS_JSON}\n</tools>"},
        {"from": "human", "value": "List all cameras."},
        {"from": "gpt", "value": '{"function": "list_all_cameras", "arguments": {}}'},
        {"from": "human", "value": "Now show the 1080p feed of cam-2."},
        {"from": "gpt", "value": '<tool_call>{"name": "get_camera_live_feed", "arguments": {"camera_id": "cam-2", "stream_quality": "1080p"}}</tool_call>'},
    ],
    "tools": TOOLS_JSON,
    "category": "Security", "subcategory": "Cameras", "task": "capture",
}

GLAIVE_ROW = {
    "id": "fixture-glaive-1",
    "conversations": [
        {"from": "system", "value": "<INSTRUCTION>Use the provided function.</INSTRUCTION>\n<TOOLS>" + TOOLS_JSON + "</TOOLS>"},
        {"from": "human", "value": "Turn on camera 5 and tell me the stream quality"},
        {"from": "gpt", "value": '{"function": "get_camera_live_feed", "arguments": {"camera_id": "cam-5", "stream_quality": "720p"}}'},
    ],
    "category": "Security", "subcategory": "Cameras", "task": "capture",
}

JSON_SCHEMA = """{"title": "PatentPortfolio", "type": "object", "properties": {
    "patentID": {"title": "Patent ID", "type": "string"},
    "patentStatus": {"title": "Status", "type": "string", "enum": ["Filed", "Published", "Granted", "Expired", "Withdrawn"]}
}, "required": ["patentID", "patentStatus"]}"""

JSON_SINGLETURN_ROW = {
    "id": "fixture-json-st-1",
    "conversations": [
        {"from": "system", "value": "Answer in JSON.\n<schema>\n" + JSON_SCHEMA + "\n</schema>"},
        {"from": "human", "value": "Create the portfolio for patent ID 'US9876543B2', status 'Granted'."},
        {"from": "gpt", "value": '{"patentID": "US9876543B2", "patentStatus": "Granted"}'},
    ],
    "schema": JSON_SCHEMA,
    "category": "Pharma", "subcategory": "Biotech", "task": "extract",
}

JSON_AGENTIC_ROW = {
    "id": "fixture-json-ag-1",
    "conversations": [
        {"from": "system", "value": "Answer in JSON.\n<schema>\n" + JSON_SCHEMA + "\n</schema>"},
        {"from": "human", "value": "First portfolio: patent ID 'US123', status 'Filed'."},
        {"from": "gpt", "value": '{"patentID": "US123", "patentStatus": "Filed"}'},
        {"from": "human", "value": "Now update it to 'Published'."},
        {"from": "gpt", "value": '{"patentID": "US123", "patentStatus": "Published"}'},
    ],
    "schema": JSON_SCHEMA,
    "category": "Pharma", "subcategory": "Biotech", "task": "extract",
}

FIXTURES = {
    "func_calling_singleturn": SINGLETURN_ROW,
    "func_calling": MULTITURN_ROW,
    "glaive_func_calling": GLAIVE_ROW,
    "json_mode_singleturn": JSON_SINGLETURN_ROW,
    "json_mode_agentic": JSON_AGENTIC_ROW,
}

# --- Real hub shapes (multi-turn configs end with a natural-language summary,
# not a <tool_call>; the tags live in earlier assistant turns). These guard
# against regressions where the loader only scans the LAST assistant turn.
MULTITURN_SUMMARY_ROW = {
    "id": "fixture-multi-summary-1",
    "conversations": [
        {"from": "system", "value": f"You are a function calling AI model.\n<tools>\n{TOOLS_JSON}\n</tools>"},
        {"from": "human", "value": "I recently installed a new security system. Check the live feed from the front door camera."},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera_live_feed", "arguments": {"camera_id": "front_door", "stream_quality": "1080p"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"name": "get_camera_live_feed", "content": {"camera_id": "front_door", "stream_quality": "1080p", "live_feed_url": "https://homecam.example.com/live/front_door_1080p"}}\n</tool_response>'},
        {"from": "gpt", "value": "I have successfully executed all the requested functions for your security cameras. Here are the results:\n\n- Live Feed from front_door camera: 1080p."},
    ],
    "tools": TOOLS_JSON,
    "category": "Security", "subcategory": "Cameras", "task": "capture",
}

GLAIVE_SUMMARY_ROW = {
    "id": "fixture-glaive-summary-1",
    "conversations": [
        {"from": "system", "value": f"You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.\n<tools>\n{TOOLS_JSON}\n</tools>"},
        {"from": "human", "value": "Hi, can you tell me the current stock price of Apple?"},
        {"from": "gpt", "value": '<tool_call>\n{"name": "get_camera_live_feed", "arguments": {"camera_id": "cam-1", "stream_quality": "720p"}}\n</tool_call>'},
        {"from": "tool", "value": '<tool_response>\n{"name": "get_camera_live_feed", "content": {"camera_id": "cam-1", "stream_quality": "720p"}}\n</tool_response>'},
        {"from": "gpt", "value": "The live feed for cam-1 is now streaming at 720p."},
        {"from": "human", "value": "Thank you for the information."},
        {"from": "gpt", "value": "You're welcome! If you have any other questions, feel free to ask."},
    ],
    "tools": TOOLS_JSON,
    "category": "Security", "subcategory": "Cameras", "task": "capture",
}

# The multi-turn config has ONE <tool_call> (the final, correct one) as TARGET.
EXPECTED_TOOL_CALLS = {
    "func_calling_singleturn": 1,
    "func_calling": 1,          # only the final tagged call is a target
    "glaive_func_calling": 1,
    "json_mode_singleturn": 1,
    "json_mode_agentic": 1,
}


def _write_config_rows(tmp_path, config: str, rows) -> str:
    """Write rows as a JSONL file and return its path (loader local_path)."""
    path = tmp_path / f"{config}.jsonl"
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(path)


@pytest.mark.parametrize("config", HERMES_CONFIGS)
def test_load_each_config(tmp_path, config: str) -> None:
    """Every config yields exactly one processable UnifiedSample from a local file."""
    path = _write_config_rows(tmp_path, config, [FIXTURES[config]])
    loader = HermesLoader(config_name=config, local_path=path)
    samples = list(loader.load())
    assert len(samples) == 1, f"config={config} stats={loader.stats}"
    s = samples[0]
    assert s.sample_id.startswith(config)
    assert s.source_dataset == "hermes-function-calling-v1"
    assert len(s.ground_truth_calls) >= 1
    assert s.metadata["config"] == config

    # function-calling configs must parse the tools into ToolDefinition
    if config not in ("json_mode_agentic", "json_mode_singleturn"):
        assert s.tools, f"config={config} parsed no tools"
        assert any(t.name == "get_camera_live_feed" for t in s.tools)
    else:
        # json-mode: GT uses the synthetic json_output tool; the tools list
        # holds the schema-derived tool (named by the JSON Schema title) or the
        # synthetic json_output fallback when no schema exists.
        assert s.metadata["mode"] == "json"
        assert s.ground_truth_calls[0].tool_name == "json_output"
        assert s.tools, f"config={config} parsed no tools"


def test_singleturn_tools_and_gt() -> None:
    from spancall.data.sources.hermes import _parse_tools_json

    tools = _parse_tools_json(TOOLS_JSON)
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "get_camera_live_feed"
    params = {p.name: p for p in t.parameters}
    assert params["camera_id"].required is True
    assert params["camera_id"].param_type == "string"
    assert params["stream_quality"].enum == ["720p", "1080p", "4k"]
    assert params["stream_quality"].required is False


def test_multiturn_recent_context(tmp_path) -> None:
    path = _write_config_rows(tmp_path, "func_calling", [MULTITURN_ROW])
    loader = HermesLoader(config_name="func_calling", local_path=path)
    (s,) = list(loader.load())
    # Earlier human turn + earlier assistant reply except the final GT pair
    ctx = s.recent_context
    assert len(ctx) >= 1
    assert any(row["source"] == "conversation" for row in ctx)
    # The utterance is the LAST human turn
    assert s.utterance == "Now show the 1080p feed of cam-2."
    assert s.ground_truth_calls[0].tool_name == "get_camera_live_feed"


def test_glaive_gt_parsing(tmp_path) -> None:
    path = _write_config_rows(tmp_path, "glaive_func_calling", [GLAIVE_ROW])
    loader = HermesLoader(config_name="glaive_func_calling", local_path=path)
    (s,) = list(loader.load())
    assert s.ground_truth_calls[0].tool_name == "get_camera_live_feed"
    assert s.ground_truth_calls[0].arguments["camera_id"] == "cam-5"


def test_json_singleturn_synthetic_gt(tmp_path) -> None:
    path = _write_config_rows(tmp_path, "json_mode_singleturn", [JSON_SINGLETURN_ROW])
    loader = HermesLoader(config_name="json_mode_singleturn", local_path=path)
    (s,) = list(loader.load())
    assert s.metadata["mode"] == "json"
    assert s.ground_truth_calls[0].tool_name == "json_output"
    assert s.ground_truth_calls[0].arguments["patentID"] == "US9876543B2"
    # The JSON Schema -> ToolDefinition mapping yields a param with the enum
    assert any(p.name == "patentStatus" and p.enum for t in s.tools for p in t.parameters)


def test_json_agentic_uses_last_turn(tmp_path) -> None:
    path = _write_config_rows(tmp_path, "json_mode_agentic", [JSON_AGENTIC_ROW])
    loader = HermesLoader(config_name="json_mode_agentic", local_path=path)
    (s,) = list(loader.load())
    assert s.utterance == "Now update it to 'Published'."
    assert s.ground_truth_calls[0].arguments["patentStatus"] == "Published"


def test_multiturn_tool_call_in_earlier_turn(tmp_path) -> None:
    """func_calling rows end with a summary; the <tool_call> lives in an
    earlier assistant turn — the loader must scan ALL assistant turns."""
    path = _write_config_rows(tmp_path, "func_calling", [MULTITURN_SUMMARY_ROW])
    loader = HermesLoader(config_name="func_calling", local_path=path)
    (s,) = list(loader.load())
    assert s.ground_truth_calls[0].tool_name == "get_camera_live_feed"
    assert s.ground_truth_calls[0].arguments["camera_id"] == "front_door"
    # The last human turn is still the utterance
    assert s.utterance.startswith("I recently installed a new security system.")


def test_glaive_summary_ends_conversation(tmp_path) -> None:
    """glaive rows can end with a 'Thank you' human turn + closing assistant
    turn; the <tool_call> is in an earlier assistant turn."""
    path = _write_config_rows(tmp_path, "glaive_func_calling", [GLAIVE_SUMMARY_ROW])
    loader = HermesLoader(config_name="glaive_func_calling", local_path=path)
    (s,) = list(loader.load())
    assert len(s.ground_truth_calls) >= 1
    assert s.ground_truth_calls[0].tool_name == "get_camera_live_feed"
    assert s.utterance == "Thank you for the information."
    # All earlier human/assistant turns populate recent context, so the model
    # still sees the actual request
    assert any(row["value"].startswith("Hi, can you tell me") for row in s.recent_context)


def test_json_array_output_wrapped_as_items(tmp_path) -> None:
    """json-mode rows whose GT is a JSON ARRAY (e.g. flight/booking lists)
    must be wrapped under 'items' instead of dropped."""
    import json as _json

    row = {
        "id": "fixture-json-array-1",
        "conversations": [
            {"from": "system", "value": "Answer in JSON.\n<schema>\n" + JSON_SCHEMA + "\n</schema>"},
            {"from": "human", "value": "Track the status of flights AA123 and UA456."},
            {"from": "gpt", "value": _json.dumps([
                {"flightNumber": "AA123", "status": "On Time"},
                {"flightNumber": "UA456", "status": "Delayed"},
            ])},
        ],
        "schema": JSON_SCHEMA,
        "category": "Aviation", "subcategory": "Flights", "task": "track",
    }
    path = _write_config_rows(tmp_path, "json_mode_singleturn", [row])
    loader = HermesLoader(config_name="json_mode_singleturn", local_path=path)
    (s,) = list(loader.load())
    assert s.ground_truth_calls[0].tool_name == "json_output"
    assert isinstance(s.ground_truth_calls[0].arguments["items"], list)
    assert s.ground_truth_calls[0].arguments["items"][0]["flightNumber"] == "AA123"
    assert loader.stats["dropped_no_gt"] == 0


def test_unknown_config_rejected() -> None:
    with pytest.raises(ValueError):
        HermesLoader(config_name="nope")


def test_partition_schema_disjoint() -> None:
    """Two samples with the same schema must land in the SAME split."""
    a = UnifiedSample(
        sample_id="a", source_dataset="t", utterance="u",
        tools=[ToolDefinition(name="t1", description="")],
        ground_truth_calls=[GroundTruthCall(tool_name="t1", arguments={})],
    )
    b = UnifiedSample(
        sample_id="b", source_dataset="t", utterance="u2",
        tools=[ToolDefinition(name="t1", description="")],
        ground_truth_calls=[GroundTruthCall(tool_name="t1", arguments={})],
    )
    splits, report = partition_schema_disjoint([a, b], seed=0)
    # both share schema ("t1",) so they must be together in one split
    for sp, items in splits.items():
        ids = {s.sample_id for s in items}
        assert not ({"a", "b"} & ids) or {"a", "b"} <= ids
    assert report["total_samples"] == 2
    assert report["distinct_schemas"] == 1
    assert report["schema_disjoint"] is False  # fallback path (1 schema)
    assert report["warning"] is not None


def test_partition_report_counts() -> None:
    a = UnifiedSample(
        sample_id="a", source_dataset="t", utterance="u",
        tools=[ToolDefinition(name="t1", description="")],
        ground_truth_calls=[GroundTruthCall(tool_name="t1", arguments={})],
        metadata={"config": "func_calling_singleturn"},
    )
    b = UnifiedSample(
        sample_id="b", source_dataset="t", utterance="u2",
        tools=[ToolDefinition(name="t2", description="")],
        ground_truth_calls=[GroundTruthCall(tool_name="t2", arguments={})],
        metadata={"config": "json_mode_agentic"},
    )
    splits, report = partition_schema_disjoint([a, b], seed=0)
    assert report["per_config"]["func_calling_singleturn"] == 1
    assert report["per_config"]["json_mode_agentic"] == 1
    assert sum(report["per_split"].values()) == 2
    assert set(splits.keys()) == {Split.TRAIN, Split.DEV, Split.TEST_ID}