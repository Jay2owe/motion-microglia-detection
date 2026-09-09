"""Result-dependent analysis pipelines; configuration never imports renderers.

Parsing records the request. Resolving it against measured tables checks the
scientific inputs; execution and result-dependent expansion belong to the runner.
"""

from __future__ import annotations


def parse(entries: object, groups: dict | None = None) -> list:
    """Read optional pipeline requests without running analysis or plotting."""
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("pipelines must be a list of request objects")
    from .rhythm_discovery import RhythmDiscoveryRequest

    requests = []
    seen = set()
    for index, entry in enumerate(entries):
        where = f"pipelines[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a request object")
        if entry.get("pipeline") != "rhythm-discovery":
            raise ValueError(f"{where}: unknown pipeline {entry.get('pipeline')!r}; "
                             "available: rhythm-discovery")
        request = RhythmDiscoveryRequest.from_dict(entry, groups or {}, where=where)
        if request.name in seen:
            raise ValueError(f"{where}: duplicate pipeline request name {request.name!r}")
        seen.add(request.name)
        requests.append(request)
    return requests
