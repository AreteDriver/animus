"""Agent execution for Forge workflows."""

from animus.cognitive import CognitiveLayer
from animus.forge.models import AgentConfig, StepResult
from animus.logging import get_logger
from animus.tools import ToolRegistry

logger = get_logger("forge.agent")

# Built-in archetype system prompts
ARCHETYPE_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a research analyst. Investigate the topic thoroughly and "
        "produce a structured brief with key findings, sources, and "
        "recommendations."
    ),
    "writer": (
        "You are a skilled writer. Produce clear, engaging content based on "
        "the provided inputs. Follow any style or format requirements."
    ),
    "reviewer": (
        "You are a quality reviewer. Evaluate the work and provide:\n"
        '1. A JSON object with a "score" field (0.0-1.0)\n'
        "2. Specific feedback and suggestions\n"
        "3. A pass/fail recommendation"
    ),
    "producer": (
        "You are a production specialist. Process the inputs into the "
        "requested output format. Focus on accuracy and completeness."
    ),
    "editor": (
        "You are an editor. Revise the provided content for clarity, "
        "consistency, and correctness. Preserve the author's voice."
    ),
    "analyst": (
        "You are a data analyst. Examine the provided data and extract "
        "key insights, patterns, and actionable findings."
    ),
}


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (~4 chars per token)."""
    return max(1, len(text) // 4)


class ForgeAgent:
    """Executes a single agent step in a Forge workflow."""

    def __init__(
        self,
        config: AgentConfig,
        cognitive: CognitiveLayer,
        tools: ToolRegistry | None = None,
    ):
        self.config = config
        self.cognitive = cognitive
        self.tools = tools

    def run(self, inputs: dict[str, str]) -> StepResult:
        """Execute the agent with resolved inputs.

        Args:
            inputs: Dict of input_name -> content from prior steps.

        Returns:
            StepResult with outputs, token usage, and success status.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(inputs)

        logger.debug(
            f"Agent {self.config.name!r} ({self.config.archetype}) "
            f"running with {len(inputs)} inputs"
        )

        try:
            if self.config.tools and self.tools:
                response = self.cognitive.think_with_tools(
                    prompt=user_prompt,
                    context=None,
                    tools=self.tools,
                )
            else:
                response = self.cognitive.think(
                    prompt=user_prompt,
                    context=system_prompt,
                )

            tokens = _estimate_tokens(user_prompt + response)
            outputs = self._parse_outputs(response)

            logger.debug(
                f"Agent {self.config.name!r} completed: {len(outputs)} outputs, ~{tokens} tokens"
            )

            return StepResult(
                agent_name=self.config.name,
                outputs=outputs,
                tokens_used=tokens,
                success=True,
            )

        except Exception as exc:
            logger.error(f"Agent {self.config.name!r} failed: {exc}")
            return StepResult(
                agent_name=self.config.name,
                outputs={},
                tokens_used=0,
                success=False,
                error=str(exc),
            )

    def _build_system_prompt(self) -> str:
        """Build the system prompt from archetype + custom prompt."""
        base = ARCHETYPE_PROMPTS.get(self.config.archetype, "")
        if self.config.system_prompt:
            if base:
                return f"{base}\n\n{self.config.system_prompt}"
            return self.config.system_prompt
        return base or f"You are a {self.config.archetype} agent."

    def _build_user_prompt(self, inputs: dict[str, str]) -> str:
        """Build the user prompt from inputs and output expectations."""
        parts = []

        if inputs:
            parts.append("## Inputs\n")
            for name, content in inputs.items():
                parts.append(f"### {name}\n{content}\n")

        if self.config.outputs:
            parts.append("## Expected Outputs\n")
            parts.append(
                "Please produce the following outputs, each under a "
                "## heading matching the output name:\n"
            )
            for out in self.config.outputs:
                parts.append(f"- **{out}**")

        return "\n".join(parts)

    def _parse_outputs(self, response: str) -> dict[str, str]:
        """Extract declared outputs from the response.

        Looks for ## heading sections matching output names.
        Falls back to assigning the entire response to the first output.
        """
        if not self.config.outputs:
            return {"response": response}

        outputs: dict[str, str] = {}

        for out_name in self.config.outputs:
            # Look for ## out_name header
            pattern = rf"##\s*{out_name}\s*\n(.*?)(?=\n##\s|\Z)"
            import re

            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                outputs[out_name] = match.group(1).strip()

        # Fallback: if no structured outputs found, use whole response
        if not outputs and self.config.outputs:
            outputs[self.config.outputs[0]] = response

        return outputs
