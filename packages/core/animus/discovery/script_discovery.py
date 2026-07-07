"""ScriptDiscovery: discovers local executable scripts as Animus tools.

Scans directories for scripts with annotated headers (docstrings, shebangs)
and converts them into Tool schemas. Supports Python, Bash, and Node scripts.

Annotation format:
    #!/usr/bin/env python3
    \"\"\"Animus Tool: my-tool-name

    Description of what this tool does.

    Args:
        param1: Description of param1 (type: str)
        param2: Description of param2 (type: int)
    \"\"\"
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("discovery.scripts")


@dataclass
class ScriptSpec:
    """A discovered local script ready for Tool registration."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    script_path: Path
    interpreter: str  # python3, bash, node


class ScriptDiscovery:
    """Discovers annotated local scripts as potential tools.

    Usage:
        discovery = ScriptDiscovery()
        specs = discovery.scan_directory("~/scripts")
        for spec in specs:
            registry.register(Tool(name=spec.name, ...))
    """

    # Supported extensions and interpreters
    INTERPRETERS = {
        ".py": "python3",
        ".sh": "bash",
        ".js": "node",
    }

    def __init__(self):
        self._discovered: list[ScriptSpec] = []

    def scan_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[ScriptSpec]:
        """Scan a directory for annotated scripts.

        Args:
            directory: Directory to scan.
            recursive: If True, scan subdirectories.

        Returns:
            List of discovered script specs.
        """
        directory = Path(directory).expanduser()
        if not directory.exists():
            logger.warning(f"Script directory not found: {directory}")
            return []

        specs: list[ScriptSpec] = []
        pattern = directory.rglob("*") if recursive else directory.iterdir()

        for path in pattern:
            if not path.is_file():
                continue
            if path.suffix not in self.INTERPRETERS:
                continue

            spec = self._parse_script(path)
            if spec:
                specs.append(spec)
                logger.debug(f"Discovered script tool: {spec.name} at {path}")

        self._discovered.extend(specs)
        logger.info(f"Script scan complete: {len(specs)} tools from {directory}")
        return specs

    def _parse_script(self, path: Path) -> ScriptSpec | None:
        """Parse a single script file for Animus annotations."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        # Check for shebang
        lines = content.splitlines()
        if not lines:
            return None

        # Determine interpreter from extension or shebang
        interpreter = self.INTERPRETERS.get(path.suffix, "bash")
        if lines[0].startswith("#!/"):
            shebang = lines[0].lower()
            if "python" in shebang:
                interpreter = "python3"
            elif "bash" in shebang or "sh" in shebang:
                interpreter = "bash"
            elif "node" in shebang:
                interpreter = "node"

        # Extract docstring / annotation
        name, description, params = self._extract_annotation(content, path)
        if not name:
            return None

        return ScriptSpec(
            name=name,
            description=description,
            parameters=params,
            script_path=path,
            interpreter=interpreter,
        )

    def _extract_annotation(
        self, content: str, path: Path
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Extract tool name, description, and parameters from script content."""
        name: str | None = None
        description = ""
        params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        # Try Python docstring extraction
        if path.suffix == ".py":
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == "main" and ast.get_docstring(node):
                            docstring = ast.get_docstring(node) or ""
                            parsed = self._parse_docstring(docstring)
                            if parsed["name"]:
                                name = parsed["name"]
                                description = parsed["description"]
                                params = parsed["parameters"]
                            break
                    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                        # Module-level docstring
                        docstring = node.value.value if isinstance(node.value.value, str) else ""
                        if docstring and "Animus Tool" in docstring:
                            parsed = self._parse_docstring(docstring)
                            if parsed["name"]:
                                name = parsed["name"]
                                description = parsed["description"]
                                params = parsed["parameters"]
                            break
            except SyntaxError:
                pass

        # Fallback: regex-based extraction for any script type
        if not name:
            match = re.search(
                r'Animus\s+Tool[:\s]+([\w-]+)\s*\n(.*?)(?:\n(?:Args|Parameters):|$)',
                content,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                name = match.group(1).strip()
                description = match.group(2).strip()

                # Extract args section if present
                args_match = re.search(
                    r'(?:Args|Parameters):\s*(.*?)(?:\n\n|$)',
                    content[match.end():],
                    re.DOTALL,
                )
                if args_match:
                    params = self._parse_args_block(args_match.group(1))

        return name, description, params

    def _parse_docstring(self, docstring: str) -> dict[str, Any]:
        """Parse a docstring for tool metadata."""
        result: dict[str, Any] = {
            "name": None,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }

        # Look for "Animus Tool: name" header
        lines = docstring.strip().splitlines()
        if not lines:
            return result

        # First non-empty line might be the tool name
        first_line = lines[0].strip()
        if "Animus Tool" in first_line:
            match = re.search(r'Animus\s+Tool[:\s]+([\w-]+)', first_line, re.IGNORECASE)
            if match:
                result["name"] = match.group(1)
            # Description is remaining text until Args
            desc_lines = []
            for line in lines[1:]:
                if line.strip().lower() in ("args:", "parameters:"):
                    break
                desc_lines.append(line)
            result["description"] = "\n".join(desc_lines).strip()
        else:
            result["description"] = docstring.strip()

        # Parse Args section
        args_match = re.search(r'(?:Args|Parameters):\s*(.*?)$', docstring, re.DOTALL | re.IGNORECASE)
        if args_match:
            result["parameters"] = self._parse_args_block(args_match.group(1))

        return result

    def _parse_args_block(self, block: str) -> dict[str, Any]:
        """Parse an Args block into JSON Schema."""
        params: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("-"):
                continue

            # Match patterns like:
            #   param_name: description (type: str, required)
            #   param_name (type: int): description
            match = re.match(
                r'(\w+)[\s:]*(.+?)(?:\s*\(([^)]+)\))?$',
                line,
            )
            if match:
                pname = match.group(1)
                pdesc = match.group(2).strip()
                ptype = "string"
                required = False

                # Include parenthetical content (group 3) for parsing type/required
                parens = match.group(3) or ""
                combined = f"{pdesc} {parens}"

                # Parse type annotation
                type_match = re.search(r'type[:\s]+(\w+)', combined, re.IGNORECASE)
                if type_match:
                    ptype = self._map_type(type_match.group(1))
                    pdesc = re.sub(r'\s*type[:\s]+\w+', '', pdesc, flags=re.IGNORECASE).strip()
                    parens = re.sub(r'\s*type[:\s]+\w+', '', parens, flags=re.IGNORECASE).strip()

                if "required" in combined.lower():
                    required = True
                    pdesc = pdesc.replace("(required)", "").replace("required", "").strip()
                    parens = parens.replace("(required)", "").replace("required", "").strip()

                params["properties"][pname] = {
                    "type": ptype,
                    "description": pdesc,
                }
                if required:
                    params["required"].append(pname)

        return params

    @staticmethod
    def _map_type(type_str: str) -> str:
        """Map Python/common types to JSON Schema types."""
        mapping = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
        }
        return mapping.get(type_str.lower(), "string")

    def get_all_discovered(self) -> list[ScriptSpec]:
        """Get all discovered script specs."""
        return list(self._discovered)
