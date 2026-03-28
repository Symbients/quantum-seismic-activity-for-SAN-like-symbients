"""Quantum Seismic — ambient environmental sensing for AI agents.

Continuous sensor sampling + temporal aggregation → structured context
that enriches every agent turn with physical environmental awareness.
"""

__version__ = "0.2.0"

from quantum_seismic.daemon import EnvironmentDaemon
from quantum_seismic.snapshot import EnvironmentSnapshot
from quantum_seismic.agent import agent_hook, enrich_prompt, system_prompt

__all__ = [
    "EnvironmentDaemon",
    "EnvironmentSnapshot",
    "agent_hook",
    "enrich_prompt",
    "system_prompt",
]
