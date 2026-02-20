"""
Animus Register Translation

Adapts communication style between formal, casual, and technical registers.
Phase 2 feature for style-aware response generation.
"""

from enum import Enum

from animus.logging import get_logger

logger = get_logger("register")


class Register(Enum):
    """Communication registers (styles)."""

    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    NEUTRAL = "neutral"  # Default, no strong style


# Keywords and phrases that indicate a register
REGISTER_INDICATORS: dict[Register, list[str]] = {
    Register.FORMAL: [
        "dear",
        "sincerely",
        "furthermore",
        "pursuant",
        "kindly",
        "herein",
        "accordingly",
        "respectfully",
        "please be advised",
        "i would like to inquire",
    ],
    Register.CASUAL: [
        "hey",
        "lol",
        "gonna",
        "wanna",
        "kinda",
        "btw",
        "yep",
        "nope",
        "cool",
        "awesome",
        "thanks!",
        "sup",
        "np",
    ],
    Register.TECHNICAL: [
        "api",
        "endpoint",
        "function",
        "class",
        "method",
        "algorithm",
        "implementation",
        "latency",
        "throughput",
        "refactor",
        "deploy",
        "pipeline",
        "schema",
        "microservice",
    ],
}

# System prompt modifiers per register
REGISTER_PROMPTS: dict[Register, str] = {
    Register.FORMAL: (
        "Respond in a formal, professional tone. Use complete sentences, "
        "proper grammar, and polished language. Avoid slang, contractions, "
        "and casual phrasing. Be courteous and precise."
    ),
    Register.CASUAL: (
        "Respond in a friendly, casual tone. Use natural language, "
        "contractions, and a conversational style. Keep things relaxed "
        "and approachable while still being helpful."
    ),
    Register.TECHNICAL: (
        "Respond with technical precision. Use domain-specific terminology, "
        "concrete examples, and structured explanations. Prioritize accuracy "
        "and detail. Include code snippets or technical references where relevant."
    ),
    Register.NEUTRAL: "",
}


def detect_register(text: str) -> Register:
    """
    Detect the register of a piece of text.

    Scores text against indicator lists for each register.

    Args:
        text: Input text to analyze

    Returns:
        Detected Register
    """
    text_lower = text.lower()
    scores: dict[Register, float] = {r: 0.0 for r in Register}

    for register, indicators in REGISTER_INDICATORS.items():
        for indicator in indicators:
            if indicator in text_lower:
                scores[register] += 1.0

    # Normalize by indicator count to avoid bias toward registers with more indicators
    for register in scores:
        indicator_count = len(REGISTER_INDICATORS.get(register, []))
        if indicator_count > 0:
            scores[register] /= indicator_count

    best = max(scores, key=lambda r: scores[r])
    if scores[best] > 0:
        logger.debug(f"Detected register: {best.value} (score={scores[best]:.2f})")
        return best

    return Register.NEUTRAL


class RegisterTranslator:
    """
    Translates and adapts text between communication registers.

    Used by the CognitiveLayer to match the user's communication style
    and to produce register-appropriate system prompts.
    """

    def __init__(self, default_register: Register = Register.NEUTRAL):
        self.default_register = default_register
        self._user_register: Register | None = None
        self._override_register: Register | None = None
        logger.debug(f"RegisterTranslator initialized, default={default_register.value}")

    @property
    def active_register(self) -> Register:
        """Get the currently active register."""
        if self._override_register:
            return self._override_register
        if self._user_register:
            return self._user_register
        return self.default_register

    def set_override(self, register: Register | None) -> None:
        """
        Set a manual register override.

        Args:
            register: Register to force, or None to clear override
        """
        self._override_register = register
        if register:
            logger.info(f"Register override set: {register.value}")
        else:
            logger.info("Register override cleared")

    def detect_and_set(self, user_text: str) -> Register:
        """
        Detect the user's register from their text and update tracking.

        Args:
            user_text: Text from the user

        Returns:
            Detected register
        """
        detected = detect_register(user_text)
        if detected != Register.NEUTRAL:
            self._user_register = detected
        return detected

    def get_system_prompt_modifier(self) -> str:
        """
        Get a system prompt modifier for the active register.

        Returns:
            String to append to the system prompt, empty if neutral
        """
        return REGISTER_PROMPTS.get(self.active_register, "")

    def adapt_prompt(self, base_prompt: str) -> str:
        """
        Adapt a system prompt to include register-appropriate instructions.

        Args:
            base_prompt: The base system prompt

        Returns:
            Modified system prompt with register instructions
        """
        modifier = self.get_system_prompt_modifier()
        if not modifier:
            return base_prompt
        return f"{base_prompt}\n\nCommunication style: {modifier}"

    def translate_response(
        self,
        text: str,
        cognitive,
        target: Register | None = None,
    ) -> str:
        """
        Translate a response into the target register using the LLM.

        Skips translation when:
        - Target is NEUTRAL (no specific style needed)
        - Text is too short (< 20 chars) to meaningfully translate
        - Text appears to be code (starts with common code patterns)

        Args:
            text: Response text to translate
            cognitive: CognitiveLayer instance for LLM translation
            target: Target register. Defaults to active_register.

        Returns:
            Translated text, or original if no translation needed
        """
        target_reg = target or self.active_register

        if target_reg == Register.NEUTRAL:
            return text

        if len(text) < 20:
            return text

        # Skip code-like text
        code_prefixes = ("```", "def ", "class ", "import ", "from ", "{", "[")
        stripped = text.lstrip()
        if any(stripped.startswith(p) for p in code_prefixes):
            return text

        prompt = (
            f"Rewrite the following text in a {target_reg.value} register. "
            f"Keep the same meaning and information. "
            f"Return only the rewritten text, nothing else.\n\n{text}"
        )

        try:
            result = cognitive.think(prompt)
            if result and result.strip():
                return result.strip()
        except Exception:
            logger.debug("Register translation failed, returning original")

        return text

    def get_register_context(self) -> dict:
        """
        Get register information as a context dict.

        Returns:
            Dict with register information for prompt assembly
        """
        return {
            "register": self.active_register.value,
            "is_override": self._override_register is not None,
            "detected_user_register": (self._user_register.value if self._user_register else None),
        }
