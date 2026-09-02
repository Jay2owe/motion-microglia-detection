"""Analysis modules.

Importing this package registers every module. A new readout becomes available
to the command line, the manifest and the report by adding one import here.
"""

from analysis.modules import (channels, contacts, coupling, history,  # noqa: F401
                             intensity, morphology, motility, neighbours,
                             object_geometry, objects, presence, provenance,
                             recurrence, regimes, rhythms, sequence_distance,
                             sholl, surveillance, territory, territory_shape,
                             walk)

__all__ = [
    "morphology", "intensity", "channels", "motility", "surveillance", "presence",
    "rhythms", "regimes", "territory", "territory_shape", "sholl", "coupling",
    "contacts", "neighbours", "objects", "object_geometry", "walk", "provenance",
    "history", "recurrence", "sequence_distance",
]
