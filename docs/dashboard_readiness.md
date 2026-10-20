# Dashboard readiness (not built)

No dashboard exists. The processing is kept separate from presentation so one can be added without touching it.

## Integration point
```python
from src.service.config import PipelineConfig
from src.service.pipeline import PipelineService
result = PipelineService(PipelineConfig(input_path=...), on_stage=callback).run()   # RunResult
data = result.to_dict()          # JSON-safe; result.artifacts holds in-memory objects (graph, embeddings) for drill-down
```
`on_stage(StageResult)` fires after each stage (for progress bars). `PipelineConfig` is a plain dataclass a UI can fill from widgets.
Nothing in `src/service` imports `src/cli`.

## Intended sections and their data
| Section | Source in `RunResult` |
|---|---|
| Run overview, status, timings, warnings/errors | `metadata`, `stages[]`, `warnings`, `errors`, `ok` |
| Input and parsing quality | `parsing`, `normalization` |
| Graph metrics | `graph` |
| GAE checkpoint status | `gae` |
| Clusters and noise | `clustering`, `units[]` (scope = cluster), `noise.events/units/stats` |
| Novelty explorer | `novelty`, `artifacts.novelty_rows` |
| ATT&CK candidates (with weak/no-match outcomes) | `mapping`, `units[].mapping` |
| Evidence and graph paths | `units[].retrieval` (`paths`, `entities`, `related_units`, `graph_contribution`) |
| Report | `report`, `units[].report.sections` (nine sections) |
| Ontology status | `ontology` |
| Export | `src.service.export.export_run`, `src.ontology.export.export_findings` |

Display rules to carry over: show similarity as "similarity", never as probability or confidence; show `status` per candidate;
label noise units as unclustered event-level evidence; show the stand-in-embedder and MOCK flags prominently.
`tests/test_cli.py::test_run_result_keys_are_stable_for_a_future_dashboard` pins the top-level keys.
