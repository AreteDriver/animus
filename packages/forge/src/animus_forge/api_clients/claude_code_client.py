"""Claude Code API client wrapper with API and CLI modes."""

import asyncio
import json
import logging
import subprocess
from typing import Any

from animus_forge.api_clients.resilience import resilient_call, resilient_call_async
from animus_forge.config import get_settings
from animus_forge.utils.retry import async_with_retry, with_retry

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:
    anthropic = None  # Optional import: anthropic package not installed


# Default role prompts - can be overridden via config/agent_prompts.json
DEFAULT_ROLE_PROMPTS = {
    "planner": """You are a strategic planning agent. Break down the requested feature into:
1. Clear, actionable implementation steps
2. Required files and their purposes
3. Dependencies and prerequisites
4. Success criteria

Output a structured markdown plan.""",
    "builder": """You are a code implementation agent. Using the provided plan:
1. Write clean, production-ready code
2. Follow best practices and patterns
3. Include inline documentation
4. Ensure code is testable

Focus on implementation quality and maintainability.""",
    "tester": """You are a testing specialist agent. For the implemented code:
1. Write comprehensive unit tests
2. Include edge cases and error conditions
3. Ensure good test coverage
4. Write clear test descriptions

Use appropriate testing frameworks.""",
    "reviewer": """You are a code review agent. Analyze the implementation for:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Provide constructive, actionable feedback.""",
    "data_analyst": """You are a data analysis specialist agent. You help with:
1. Writing SQL queries for PostgreSQL, MySQL, BigQuery, Snowflake, and other databases
2. Creating pandas data pipelines for data transformation and analysis
3. Building visualizations with matplotlib, seaborn, plotly, and other libraries
4. Performing statistical analysis and generating insights
5. Data cleaning, validation, and quality assessment
6. Creating reports and dashboards

When analyzing data:
- Always validate data types and handle missing values appropriately
- Use efficient query patterns (avoid SELECT *, use appropriate indexes)
- Include comments explaining the analysis methodology
- Provide both the code and interpretation of results
- Consider data privacy and security implications
- Suggest follow-up analyses when relevant""",
    "devops": """You are a DevOps and infrastructure specialist agent. You help with:
1. Docker containerization and Docker Compose configurations
2. Kubernetes manifests, Helm charts, and cluster configuration
3. CI/CD pipelines for GitHub Actions, GitLab CI, Jenkins, and CircleCI
4. Infrastructure as Code with Terraform, Pulumi, and CloudFormation
5. Cloud platform configurations for AWS, GCP, and Azure
6. Monitoring, logging, and observability setup

When creating infrastructure:
- Follow security best practices (least privilege, secrets management)
- Include health checks and proper resource limits
- Design for high availability and fault tolerance when appropriate
- Document environment variables and configuration options
- Provide cost-conscious alternatives where applicable
- Include rollback strategies and disaster recovery considerations""",
    "security_auditor": """You are a security audit specialist agent. You help with:
1. Code security review for common vulnerabilities (OWASP Top 10)
2. Dependency vulnerability scanning and remediation
3. Infrastructure security assessment
4. Compliance checking (PCI-DSS, HIPAA, SOC2, GDPR)
5. Secure coding recommendations
6. Penetration testing guidance and threat modeling

When performing audits:
- Prioritize findings by severity and exploitability
- Provide specific remediation steps for each vulnerability
- Reference CVEs and security advisories where applicable
- Consider both immediate fixes and long-term security improvements
- Explain the potential impact of each vulnerability
- Suggest security testing approaches for ongoing protection""",
    "migrator": """You are a code migration and refactoring specialist agent. You help with:
1. Framework upgrades (React 17→18, Django 3→4, Rails 6→7, etc.)
2. Language migrations (JavaScript→TypeScript, Python 2→3, etc.)
3. API migrations (REST→GraphQL, SDK version upgrades)
4. Database migrations and schema changes
5. Large-scale refactoring with behavior preservation
6. Dependency updates and compatibility fixes

When planning migrations:
- Analyze breaking changes and their impact
- Create incremental migration phases when possible
- Preserve existing behavior and tests
- Document all changes with before/after examples
- Identify areas requiring manual intervention
- Provide rollback strategies for each phase
- Consider backwards compatibility during transition periods""",
    "model_builder": """You are a 3D modeling and game development specialist agent. You help with:
1. Creating scripts, shaders, and materials for Unity, Blender, Unreal, Godot, and Three.js
2. Generating procedural geometry and mesh manipulation code
3. Setting up scenes, lighting, and camera configurations
4. Writing animation controllers and state machines
5. Optimizing 3D assets for performance (LOD, batching, draw call reduction)
6. Creating prefabs, blueprints, and reusable components

When generating code:
- Use platform-specific best practices (C# for Unity, Python for Blender, C++/Blueprints for Unreal)
- Include comments explaining 3D-specific concepts
- Consider performance implications (polygon count, texture memory, shader complexity)
- Provide step-by-step instructions for tasks requiring manual work in 3D tools

For non-code tasks, provide detailed instructions that can be followed in the target 3D application.""",
}

# Default mapping of Gorgon roles to skill agent categories
DEFAULT_ROLE_SKILL_AGENTS: dict[str, list[str]] = {
    "builder": ["system", "browser"],
    "devops": ["system"],
    "security_auditor": ["system"],
    "planner": ["system", "browser"],
    "reviewer": ["system"],
}


class ClaudeCodeClient:
    """Wrapper for Claude Code with both API and CLI modes."""

    def __init__(self):
        settings = get_settings()
        self.mode = settings.claude_mode
        self.api_key = settings.anthropic_api_key
        self.cli_path = settings.claude_cli_path
        self.client = None
        self._enforcer = None
        self._enforcer_init_attempted = False
        self._voter = None
        self._voter_init_attempted = False
        self._library = None
        self._library_init_attempted = False
        self.role_prompts = self._load_role_prompts(settings)
        self._inject_skill_context(settings)

        if self.mode == "api" and self.api_key and anthropic:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        else:
            self.async_client = None

    def _load_role_prompts(self, settings) -> dict[str, str]:
        """Load role prompts from config file or use defaults."""
        prompts_file = settings.base_dir / "config" / "agent_prompts.json"
        if prompts_file.exists():
            try:
                with open(prompts_file) as f:
                    data = json.load(f)
                return {
                    role: info.get("system_prompt", DEFAULT_ROLE_PROMPTS.get(role, ""))
                    for role, info in data.items()
                }
            except Exception:
                pass  # Non-critical fallback: config file parse failure, use default prompts
        return DEFAULT_ROLE_PROMPTS.copy()

    @staticmethod
    def _load_role_skill_mapping(settings) -> dict[str, list[str]]:
        """Load role-to-skill-agent mapping from config or use defaults."""
        mapping_file = settings.base_dir / "config" / "role_skill_mapping.json"
        if mapping_file.exists():
            try:
                with open(mapping_file) as f:
                    return json.load(f)
            except Exception:
                logger.warning("Failed to load role_skill_mapping.json, using defaults")
        return DEFAULT_ROLE_SKILL_AGENTS.copy()

    def _inject_skill_context(self, settings) -> None:
        """Append skill context to role prompts for mapped roles."""
        try:
            from animus_forge.skills import SkillLibrary
        except ImportError:
            logger.debug("Skills module not available, skipping skill injection")
            return

        mapping = self._load_role_skill_mapping(settings)

        try:
            library = SkillLibrary()
        except Exception:
            logger.debug("Failed to initialize SkillLibrary, skipping skill injection")
            return

        for role, agents in mapping.items():
            if role not in self.role_prompts:
                continue
            context_parts: list[str] = []
            for agent in agents:
                ctx = library.build_skill_context(agent)
                if ctx:
                    context_parts.append(ctx)
            if context_parts:
                self.role_prompts[role] += "\n\n---\n\n" + "\n\n".join(context_parts)

    @property
    def enforcer(self):
        """Lazy-init SkillEnforcer, returning None on failure."""
        if not self._enforcer_init_attempted:
            self._enforcer_init_attempted = True
            try:
                from animus_forge.skills import SkillEnforcer, SkillLibrary

                self._enforcer = SkillEnforcer(
                    SkillLibrary(),
                    role_skill_agents=DEFAULT_ROLE_SKILL_AGENTS,
                )
            except Exception:
                logger.debug("Failed to initialize SkillEnforcer")
        return self._enforcer

    @property
    def library(self):
        """Lazy-init SkillLibrary, returning None on failure."""
        if not self._library_init_attempted:
            self._library_init_attempted = True
            try:
                from animus_forge.skills import SkillLibrary

                self._library = SkillLibrary()
            except Exception:
                logger.debug("Failed to initialize SkillLibrary")
        return self._library

    @property
    def voter(self):
        """Lazy-init ConsensusVoter, returning None on failure."""
        if not self._voter_init_attempted:
            self._voter_init_attempted = True
            try:
                from animus_forge.skills import ConsensusVoter

                self._voter = ConsensusVoter(self)
            except Exception:
                logger.debug("Failed to initialize ConsensusVoter")
        return self._voter

    def _resolve_consensus_level(self, role: str, task: str) -> str | None:
        """Determine consensus level for a role+task.

        Tries capability-level matching first (scanning task text for capability
        names), falling back to the highest consensus across all role capabilities.
        """
        lib = self.library
        if lib is None:
            return None

        agents = DEFAULT_ROLE_SKILL_AGENTS.get(role)
        if not agents:
            return None

        # Try capability-level match: scan task for capability names
        task_lower = task.lower()
        matched_level: str | None = None
        matched_order = -1

        from animus_forge.skills.consensus import consensus_level_order

        for agent in agents:
            for skill in lib.get_skills_for_agent(agent):
                for cap in skill.capabilities:
                    cap_name_parts = cap.name.lower().replace("_", " ")
                    if cap_name_parts in task_lower or cap.name.lower() in task_lower:
                        order = consensus_level_order(cap.consensus_required)
                        if order > matched_order:
                            matched_order = order
                            matched_level = cap.consensus_required

        if matched_level is not None:
            return matched_level

        # Fallback: highest across all capabilities for this role
        return lib.get_highest_consensus_for_role(role, DEFAULT_ROLE_SKILL_AGENTS)

    def _check_consensus(self, role: str, task: str, enforcement: dict) -> dict | None:
        """Run consensus vote, returning None to skip or a verdict dict."""
        try:
            if not enforcement.get("passed", True):
                return None
            level = self._resolve_consensus_level(role, task)
            if level is None or level == "any":
                return None
            v = self.voter
            if v is None:
                return None
            verdict = v.vote(task, level, role=role)
            d = verdict.to_dict()
            if d.get("requires_user_confirmation"):
                d["pending_user_confirmation"] = True
            return d
        except Exception:
            logger.warning("Consensus check failed, skipping", exc_info=True)
            return None

    async def _check_consensus_async(self, role: str, task: str, enforcement: dict) -> dict | None:
        """Async version of _check_consensus."""
        try:
            if not enforcement.get("passed", True):
                return None
            level = self._resolve_consensus_level(role, task)
            if level is None or level == "any":
                return None
            v = self.voter
            if v is None:
                return None
            verdict = await v.vote_async(task, level, role=role)
            d = verdict.to_dict()
            if d.get("requires_user_confirmation"):
                d["pending_user_confirmation"] = True
            return d
        except Exception:
            logger.warning("Consensus check failed, skipping", exc_info=True)
            return None

    def _check_enforcement(self, role: str, output: str) -> dict:
        """Run enforcement check, failing open on errors."""
        try:
            enf = self.enforcer
            if enf is None:
                return {"action": "allow", "passed": True, "violations": []}
            result = enf.check_output(role, output)
            return result.to_dict()
        except Exception:
            logger.warning("Enforcement check failed, allowing output", exc_info=True)
            return {"action": "allow", "passed": True, "violations": []}

    def is_configured(self) -> bool:
        """Check if Claude Code client is configured."""
        if self.mode == "api":
            return self.client is not None
        else:
            # CLI mode - check if claude command exists
            try:
                result = subprocess.run(
                    [self.cli_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False

    def execute_agent(
        self,
        role: str,
        task: str,
        context: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Execute a specialized agent with role-specific prompt.

        Args:
            role: Agent role (planner, builder, tester, reviewer)
            task: The task description
            context: Optional context from previous steps
            model: Claude model to use
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Claude Code client not configured"}

        system_prompt = self.role_prompts.get(role, "")
        if not system_prompt:
            return {"success": False, "error": f"Unknown role: {role}"}

        user_prompt = f"Task: {task}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"

        try:
            if self.mode == "api":
                output = self._execute_via_api(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                output = self._execute_via_cli(prompt=full_prompt)

            result = {"success": True, "output": output, "role": role}
            result["enforcement"] = self._check_enforcement(role, output)

            consensus = self._check_consensus(role, task, result["enforcement"])
            if consensus is not None:
                result["consensus"] = consensus
                if not consensus["approved"]:
                    result["success"] = False
                    result["error"] = "Consensus vote rejected the operation"
                elif consensus.get("pending_user_confirmation"):
                    result["pending_user_confirmation"] = True

            return result
        except Exception as e:
            return {"success": False, "error": str(e), "role": role}

    def generate_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a completion without role-specific prompts.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Claude Code client not configured"}

        try:
            if self.mode == "api":
                output = self._execute_via_api(
                    system_prompt=system_prompt or "You are a helpful assistant.",
                    user_prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                output = self._execute_via_cli(prompt=full_prompt)

            return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_cli_command(
        self,
        prompt: str,
        working_dir: str | None = None,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Execute Claude CLI with a custom prompt.

        Args:
            prompt: The prompt to send to Claude CLI
            working_dir: Optional working directory for the CLI
            timeout: Command timeout in seconds

        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        try:
            output = self._execute_via_cli(
                prompt=prompt,
                working_dir=working_dir,
                timeout=timeout,
            )
            return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_via_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> str:
        """Execute via Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        return self._call_anthropic_api(system_prompt, user_prompt, model, max_tokens)

    @resilient_call("anthropic")
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _call_anthropic_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Make the actual Anthropic API call with retry logic."""
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _execute_via_cli(
        self,
        prompt: str,
        working_dir: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Execute via Claude CLI subprocess."""
        cmd = [self.cli_path, "-p", prompt, "--no-input"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr}")

        return result.stdout

    async def execute_agent_async(
        self,
        role: str,
        task: str,
        context: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Async version of execute_agent().

        Execute a specialized agent with role-specific prompt asynchronously.

        Args:
            role: Agent role (planner, builder, tester, reviewer)
            task: The task description
            context: Optional context from previous steps
            model: Claude model to use
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Claude Code client not configured"}

        system_prompt = self.role_prompts.get(role, "")
        if not system_prompt:
            return {"success": False, "error": f"Unknown role: {role}"}

        user_prompt = f"Task: {task}"
        if context:
            user_prompt += f"\n\nContext:\n{context}"

        try:
            if self.mode == "api":
                output = await self._execute_via_api_async(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                output = await self._execute_via_cli_async(prompt=full_prompt)

            result = {"success": True, "output": output, "role": role}
            result["enforcement"] = self._check_enforcement(role, output)

            consensus = await self._check_consensus_async(role, task, result["enforcement"])
            if consensus is not None:
                result["consensus"] = consensus
                if not consensus["approved"]:
                    result["success"] = False
                    result["error"] = "Consensus vote rejected the operation"
                elif consensus.get("pending_user_confirmation"):
                    result["pending_user_confirmation"] = True

            return result
        except Exception as e:
            return {"success": False, "error": str(e), "role": role}

    async def generate_completion_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Async version of generate_completion().

        Generate a completion without role-specific prompts asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Claude model to use
            max_tokens: Maximum tokens in response

        Returns:
            Dict with 'success', 'output', and optionally 'error'
        """
        if not self.is_configured():
            return {"success": False, "error": "Claude Code client not configured"}

        try:
            if self.mode == "api":
                output = await self._execute_via_api_async(
                    system_prompt=system_prompt or "You are a helpful assistant.",
                    user_prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
            else:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                output = await self._execute_via_cli_async(prompt=full_prompt)

            return {"success": True, "output": output}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _execute_via_api_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> str:
        """Execute via Anthropic API asynchronously."""
        if not self.async_client:
            raise RuntimeError("Anthropic async client not initialized")

        return await self._call_anthropic_api_async(system_prompt, user_prompt, model, max_tokens)

    @resilient_call_async("anthropic")
    @async_with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def _call_anthropic_api_async(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int,
    ) -> str:
        """Make the actual Anthropic API call asynchronously with retry logic."""
        response = await self.async_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    async def _execute_via_cli_async(
        self,
        prompt: str,
        working_dir: str | None = None,
        timeout: int = 300,
    ) -> str:
        """Execute via Claude CLI subprocess asynchronously."""
        cmd = [self.cli_path, "-p", prompt, "--no-input"]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(f"Claude CLI timed out after {timeout} seconds")

        if process.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {stderr.decode()}")

        return stdout.decode()
