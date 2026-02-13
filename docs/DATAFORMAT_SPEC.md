# DataFormatSpec (Draft)

NavIRL trace bundle layout:

```text
out/<run_id>/bundle/
  scenario.yaml
  state.jsonl
  events.jsonl
  summary.json
  frames/
    frame_0000.png
    ...
    video.mp4   # optional
```

## `state.jsonl`

Each line:

```json
{
  "step": 12,
  "time_s": 1.2,
  "agents": [
    {
      "id": 0,
      "kind": "robot",
      "x": 0.12,
      "y": -0.03,
      "vx": 0.45,
      "vy": 0.01,
      "goal_x": 1.4,
      "goal_y": 0.0,
      "radius": 0.18,
      "max_speed": 0.9,
      "behavior": "GO_TO",
      "metadata": {}
    }
  ]
}
```

## `events.jsonl`

Each line:

```json
{
  "step": 14,
  "time_s": 1.4,
  "event_type": "door_token_acquire",
  "agent_id": 2,
  "payload": {"token": "doorway"}
}
```

## `summary.json`

Contains scenario/run metadata, computed metrics, artifact pointers, and event
counts.
