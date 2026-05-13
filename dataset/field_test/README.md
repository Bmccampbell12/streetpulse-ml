# Field Test Dataset — streetpulse-ml

Labeled IMU telemetry events for training and evaluating the severity scorer.

## Files

- `sample_telemetry.jsonl` — 13 labeled events (8 real from rad.txt, 5 synthetic)

## Schema

Each line is a JSON object with:
- `id`: unique event identifier
- `source`: `"rad.txt"` or `"synthetic"`
- `peak_g`: peak G-force magnitude
- `duration_us`: event duration in **microseconds** (device native unit)
- `duration_ms`: corrected duration in milliseconds (duration_us / 1000)
- `ground_truth_tier`: noise / negligible / low / moderate / severe / critical
- `ground_truth_type`: pothole / crack / rough_pavement / expansion_joint / normal
- `gps_valid`: whether GPS coordinates are trustworthy

## Real Device Notes (rad.txt session)

- GPS lock failed — all real events at (44.87, -93.04)
- Noise floor: 1.41–1.44G (gravity + bench vibration at rest)
- Only 4 of 200+ events are genuine road defects (peak_g > 5G)
- Duration from device is in **microseconds** — backend currently treats as ms (1000x bug)

## Usage

```python
from processing.severity import score_event

with open("dataset/field_test/sample_telemetry.jsonl") as f:
    for line in f:
        event = json.loads(line)
        result = score_event(event["peak_g"], event["duration_us"])
        assert result.tier == event["ground_truth_tier"], f"Mismatch on {event[\"id\"]}"
```

