
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


class LayerTrace:
    """Record inputs, outputs, computations and interpretations per stage."""

    def __init__(self, output_dir: str | Path, max_items: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self.records: list[dict[str, Any]] = []

    def record(
        self,
        stage: str,
        input_data: Any,
        output_data: Any,
        *,
        computation: str,
        interpretation: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record a stage after its real computation has completed."""
        entry = {
            "stage": stage,
            "computation": computation,
            "input": self.summarize(input_data),
            "output": self.summarize(output_data),
            "metrics": metrics or {},
            "interpretation": interpretation,
        }
        self.records.append(entry)
        self._write_json()
        self._write_markdown()

    def summarize(self, value: Any) -> Any:
        """Keep representative samples, shapes and counts."""
        if value is None or isinstance(value, (str, int, bool)):
            return value

        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        # NumPy arrays and PyTorch tensors, if present.
        if hasattr(value, "detach") and hasattr(value, "shape"):
            tensor = value.detach().cpu()
            return {
                "type": type(value).__name__,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sample": self.summarize(tensor.tolist()),
            }

        if hasattr(value, "shape") and hasattr(value, "tolist"):
            return {
                "type": type(value).__name__,
                "shape": list(value.shape),
                "sample": self.summarize(value.tolist()),
            }

        # Pandas DataFrame / Series.
        if hasattr(value, "to_dict") and hasattr(value, "columns"):
            return {
                "type": type(value).__name__,
                "shape": list(value.shape),
                "columns": [str(c) for c in value.columns],
                "sample": self.summarize(
                    value.head(self.max_items).to_dict(orient="records")
                ),
            }

        if isinstance(value, dict):
            items = list(value.items())
            return {
                "type": "dict",
                "count": len(items),
                "sample": {
                    str(k): self.summarize(v)
                    for k, v in items[:self.max_items]
                },
            }

        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "count": len(value),
                "sample": [
                    self.summarize(x)
                    for x in value[:self.max_items]
                ],
            }

        # Sets, paths and other small objects.
        if isinstance(value, (set, frozenset)):
            return {
                "type": type(value).__name__,
                "count": len(value),
                "sample": [
                    self.summarize(x)
                    for x in list(value)[:self.max_items]
                ],
            }

        if isinstance(value, Path):
            return str(value)

        # Avoid serializing arbitrary model internals.
        return {"type": type(value).__name__, "repr": repr(value)[:300]}

    def _write_json(self) -> None:
        path = self.output_dir / "layer_trace.json"
        path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _write_markdown(self) -> None:
        lines = [
            "# SUBHANY Layer-by-Layer Execution Trace",
            "",
            "Values below are captured from actual pipeline execution.",
            "",
        ]

        for index, item in enumerate(self.records, start=1):
            lines += [
                f"## {index}. {item['stage']}",
                "",
                f"**Computation:** {item['computation']}",
                "",
                "**Input sample:**",
                "```json",
                json.dumps(item["input"], indent=2, ensure_ascii=False),
                "```",
                "",
                "**Output sample:**",
                "```json",
                json.dumps(item["output"], indent=2, ensure_ascii=False),
                "```",
                "",
                "**Metrics:**",
                "```json",
                json.dumps(item["metrics"], indent=2, ensure_ascii=False),
                "```",
                "",
                f"**Interpretation:** {item['interpretation']}",
                "",
                "---",
                "",
            ]

        (self.output_dir / "layer_trace.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
