"""Claude Agent SDK integration — prompt enrichment + system prompt helpers.

Plugs the EnvironmentDaemon into any claude_agent_sdk.query() call
so the agent receives fresh environmental context at every turn.

Usage:
    from quantum_seismic import EnvironmentDaemon, enrich_prompt, system_prompt
    from claude_agent_sdk import query, ClaudeAgentOptions

    daemon = EnvironmentDaemon()
    daemon.start()

    async for message in query(
        prompt=enrich_prompt(daemon, user_input),
        options=ClaudeAgentOptions(
            system_prompt=system_prompt(),
        ),
    ):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_seismic.daemon import EnvironmentDaemon


def system_prompt(extra: str = "") -> str:
    """Generate a system prompt that establishes environmental awareness.

    Args:
        extra: Additional system prompt text to append.
    """
    base = (
        "You are an agent with environmental awareness. An <environment> block "
        "accompanies each message containing real-time sensor data from your "
        "physical surroundings — accelerometer vibrations, ambient sound levels, "
        "GPS location, webcam observations, and regional seismic activity.\n\n"
        "Use this context naturally: acknowledge the environment when relevant "
        "(e.g. if it's noisy, if there was seismic activity nearby, if the user "
        "appears to have moved locations). Don't force it — only reference the "
        "environment when it genuinely enriches your response.\n\n"
        "The sensor data includes temporal aggregates (1min, 1hr, 24hr) so you "
        "can understand trends, not just instantaneous state."
    )
    if extra:
        base += f"\n\n{extra}"
    return base


def enrich_prompt(daemon: EnvironmentDaemon, user_input: str) -> str:
    """Append the current environment snapshot to the user's message.

    This is the primary integration point. Call this before passing
    the prompt to claude_agent_sdk.query():

        prompt=enrich_prompt(daemon, user_input)

    The agent sees the user's text followed by an <environment> XML block.
    """
    snapshot = daemon.snapshot()
    return f"{user_input}\n\n{snapshot.to_context_block()}"


def dynamic_system_prompt(daemon: EnvironmentDaemon, extra: str = "") -> str:
    """Generate a system prompt that includes the current environment snapshot.

    Alternative to enrich_prompt() — injects context into the system prompt
    instead of the user message. Useful for single-turn queries where you
    want the context always present.
    """
    base = system_prompt(extra)
    snapshot = daemon.snapshot()
    return f"{base}\n\nCurrent environment state:\n{snapshot.to_context_block()}"


# Keep backward compat
def agent_hook(daemon: EnvironmentDaemon):
    """Create a hook callback (for experimentation with SDK hook system).

    Note: UserPromptSubmit hooks in the Agent SDK are for validation/logging,
    not prompt mutation. Use enrich_prompt() instead for reliable context
    injection.
    """

    async def inject_environment(input_data, tool_use_id=None, context=None):
        snapshot = daemon.snapshot()
        context_block = snapshot.to_context_block()
        if isinstance(input_data, dict):
            current_prompt = input_data.get("prompt", "")
            if current_prompt:
                input_data["prompt"] = f"{current_prompt}\n\n{context_block}"
            else:
                input_data["prompt"] = context_block
        return input_data

    return inject_environment
