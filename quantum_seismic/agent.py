"""Claude Agent SDK integration — hook + system prompt helpers.

Plugs the EnvironmentDaemon into any claude_agent_sdk.query() call
so the agent receives fresh environmental context at every turn.

Usage:
    from quantum_seismic import EnvironmentDaemon, agent_hook, system_prompt
    from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher

    daemon = EnvironmentDaemon()
    daemon.start()

    async for message in query(
        prompt=user_input,
        options=ClaudeAgentOptions(
            system_prompt=system_prompt(),
            hooks={
                "UserPromptSubmit": [
                    HookMatcher(matcher=".*", hooks=[agent_hook(daemon)])
                ]
            },
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


def agent_hook(daemon: EnvironmentDaemon):
    """Create a UserPromptSubmit hook that injects environment context.

    Returns an async callback compatible with claude_agent_sdk HookMatcher.

    The hook appends the current environment snapshot as an <environment>
    XML block to the user's message, so the agent sees it alongside
    whatever the user typed.
    """

    async def inject_environment(input_data, tool_use_id=None, context=None):
        """Hook callback — called before each user message is processed."""
        snapshot = daemon.snapshot()
        context_block = snapshot.to_context_block()

        # Append environment context to the user's prompt
        if isinstance(input_data, dict):
            current_prompt = input_data.get("prompt", "")
            if current_prompt:
                input_data["prompt"] = f"{current_prompt}\n\n{context_block}"
            else:
                input_data["prompt"] = context_block

        return input_data

    return inject_environment


def snapshot_as_message(daemon: EnvironmentDaemon) -> str:
    """Get the current environment snapshot formatted as a context block.

    Useful for manual injection into prompts without using hooks:

        prompt = f"{user_input}\\n\\n{snapshot_as_message(daemon)}"
    """
    return daemon.snapshot().to_context_block()
