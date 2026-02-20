"""Constants and templates for the visual workflow builder."""

# Node type configurations
NODE_TYPE_CONFIG = {
    "claude_code": {
        "label": "Claude Agent",
        "icon": "\U0001f916",
        "color": "#7c3aed",
        "description": "AI agent using Claude for code generation, analysis, or planning",
        "params": ["role", "prompt", "estimated_tokens"],
    },
    "openai": {
        "label": "OpenAI Agent",
        "icon": "\U0001f9e0",
        "color": "#10b981",
        "description": "AI agent using OpenAI models",
        "params": ["role", "prompt", "model", "temperature"],
    },
    "shell": {
        "label": "Shell Command",
        "icon": "\U0001f4bb",
        "color": "#f59e0b",
        "description": "Execute shell commands",
        "params": ["command", "allow_failure"],
    },
    "parallel": {
        "label": "Parallel Group",
        "icon": "\U0001f500",
        "color": "#3b82f6",
        "description": "Execute multiple steps in parallel",
        "params": ["steps"],
    },
    "checkpoint": {
        "label": "Checkpoint",
        "icon": "\U0001f3c1",
        "color": "#6b7280",
        "description": "Resume point for workflow continuation",
        "params": ["message"],
    },
    "fan_out": {
        "label": "Fan Out",
        "icon": "\U0001f4e4",
        "color": "#ec4899",
        "description": "Distribute work across multiple parallel executions",
        "params": ["items_from", "step_template"],
    },
    "fan_in": {
        "label": "Fan In",
        "icon": "\U0001f4e5",
        "color": "#8b5cf6",
        "description": "Aggregate results from fan-out operations",
        "params": ["aggregate_as"],
    },
    "map_reduce": {
        "label": "Map Reduce",
        "icon": "\U0001f5fa\ufe0f",
        "color": "#14b8a6",
        "description": "Map-reduce pattern for data processing",
        "params": ["items", "map_step", "reduce_step"],
    },
    "branch": {
        "label": "Branch",
        "icon": "\U0001f500",
        "color": "#f97316",
        "description": "Conditional branching based on context",
        "params": ["condition", "true_step", "false_step"],
    },
    "loop": {
        "label": "Loop",
        "icon": "\U0001f504",
        "color": "#0ea5e9",
        "description": "Iterate over items or until condition met",
        "params": ["items", "step_template", "max_iterations"],
    },
    "mcp_tool": {
        "label": "MCP Tool",
        "icon": "\U0001f527",
        "color": "#6366f1",
        "description": "Call a tool on a registered MCP server",
        "params": ["server", "tool", "arguments"],
    },
}

# Agent roles available for claude_code/openai steps
AGENT_ROLES = [
    "planner",
    "builder",
    "tester",
    "reviewer",
    "architect",
    "documenter",
    "analyst",
    "visualizer",
    "reporter",
    "data_engineer",
]

# Pre-built workflow templates
WORKFLOW_TEMPLATES = {
    "feature_development": {
        "name": "Feature Development",
        "icon": "\U0001f680",
        "description": "Plan, build, test, and review a new feature",
        "workflow": {
            "name": "Feature Development",
            "version": "1.0",
            "description": "End-to-end feature development workflow with planning, implementation, testing, and code review.",
            "token_budget": 150000,
            "timeout_seconds": 7200,
            "inputs": {
                "feature_request": {
                    "type": "string",
                    "required": True,
                    "description": "Description of the feature to implement",
                },
                "codebase_context": {
                    "type": "string",
                    "required": False,
                    "description": "Relevant codebase information",
                },
            },
            "outputs": ["plan", "code", "tests", "review"],
            "steps": [
                {
                    "id": "plan",
                    "type": "claude_code",
                    "params": {
                        "role": "planner",
                        "prompt": "Create a detailed implementation plan for: {{feature_request}}",
                    },
                    "outputs": ["plan"],
                },
                {
                    "id": "build",
                    "type": "claude_code",
                    "params": {
                        "role": "builder",
                        "prompt": "Implement the feature based on this plan: {{plan}}",
                    },
                    "depends_on": "plan",
                    "outputs": ["code"],
                },
                {
                    "id": "test",
                    "type": "claude_code",
                    "params": {
                        "role": "tester",
                        "prompt": "Write comprehensive tests for: {{code}}",
                    },
                    "depends_on": "build",
                    "outputs": ["tests"],
                },
                {
                    "id": "review",
                    "type": "claude_code",
                    "params": {
                        "role": "reviewer",
                        "prompt": "Review the code and tests for quality and security: {{code}} {{tests}}",
                    },
                    "depends_on": "test",
                    "outputs": ["review"],
                },
            ],
        },
    },
    "code_review": {
        "name": "Code Review",
        "icon": "\U0001f50d",
        "description": "Analyze code for bugs, security issues, and improvements",
        "workflow": {
            "name": "Code Review",
            "version": "1.0",
            "description": "Comprehensive code review with security analysis and improvement suggestions.",
            "token_budget": 80000,
            "timeout_seconds": 3600,
            "inputs": {
                "code": {
                    "type": "string",
                    "required": True,
                    "description": "Code to review",
                },
                "focus_areas": {
                    "type": "string",
                    "required": False,
                    "description": "Specific areas to focus on",
                },
            },
            "outputs": ["analysis", "security_report", "suggestions"],
            "steps": [
                {
                    "id": "analyze",
                    "type": "claude_code",
                    "params": {
                        "role": "analyst",
                        "prompt": "Analyze this code for correctness, patterns, and architecture: {{code}}",
                    },
                    "outputs": ["analysis"],
                },
                {
                    "id": "security",
                    "type": "claude_code",
                    "params": {
                        "role": "reviewer",
                        "prompt": "Review for security vulnerabilities (OWASP Top 10, injection, etc.): {{code}}",
                    },
                    "outputs": ["security_report"],
                },
                {
                    "id": "suggest",
                    "type": "claude_code",
                    "params": {
                        "role": "architect",
                        "prompt": "Based on analysis: {{analysis}} and security review: {{security_report}}, suggest improvements.",
                    },
                    "depends_on": ["analyze", "security"],
                    "outputs": ["suggestions"],
                },
            ],
        },
    },
    "documentation": {
        "name": "Documentation Generator",
        "icon": "\U0001f4da",
        "description": "Generate comprehensive documentation from code",
        "workflow": {
            "name": "Documentation Generator",
            "version": "1.0",
            "description": "Automatically generate API docs, usage guides, and architecture documentation.",
            "token_budget": 100000,
            "timeout_seconds": 5400,
            "inputs": {
                "source_code": {
                    "type": "string",
                    "required": True,
                    "description": "Source code to document",
                },
                "project_name": {
                    "type": "string",
                    "required": True,
                    "description": "Name of the project",
                },
            },
            "outputs": ["api_docs", "usage_guide", "architecture_doc"],
            "steps": [
                {
                    "id": "analyze_structure",
                    "type": "claude_code",
                    "params": {
                        "role": "architect",
                        "prompt": "Analyze the structure and architecture of: {{source_code}}",
                    },
                    "outputs": ["structure_analysis"],
                },
                {
                    "id": "generate_api_docs",
                    "type": "claude_code",
                    "params": {
                        "role": "documenter",
                        "prompt": "Generate API documentation for {{project_name}}: {{source_code}}",
                    },
                    "depends_on": "analyze_structure",
                    "outputs": ["api_docs"],
                },
                {
                    "id": "generate_usage_guide",
                    "type": "claude_code",
                    "params": {
                        "role": "documenter",
                        "prompt": "Write a usage guide for {{project_name}} based on: {{structure_analysis}}",
                    },
                    "depends_on": "analyze_structure",
                    "outputs": ["usage_guide"],
                },
                {
                    "id": "generate_architecture_doc",
                    "type": "claude_code",
                    "params": {
                        "role": "architect",
                        "prompt": "Document the architecture of {{project_name}}: {{structure_analysis}}",
                    },
                    "depends_on": "analyze_structure",
                    "outputs": ["architecture_doc"],
                },
            ],
        },
    },
    "data_analysis": {
        "name": "Data Analysis Pipeline",
        "icon": "\U0001f4ca",
        "description": "Analyze data and generate visualizations with report",
        "workflow": {
            "name": "Data Analysis Pipeline",
            "version": "1.0",
            "description": "Load, analyze, visualize data and generate an executive summary report.",
            "token_budget": 120000,
            "timeout_seconds": 5400,
            "inputs": {
                "data_source": {
                    "type": "string",
                    "required": True,
                    "description": "Path or description of data source",
                },
                "analysis_goals": {
                    "type": "string",
                    "required": True,
                    "description": "What insights are you looking for?",
                },
            },
            "outputs": ["analysis", "visualizations", "report"],
            "steps": [
                {
                    "id": "load_and_explore",
                    "type": "claude_code",
                    "params": {
                        "role": "data_engineer",
                        "prompt": "Load and explore data from: {{data_source}}. Goals: {{analysis_goals}}",
                    },
                    "outputs": ["data_summary"],
                },
                {
                    "id": "analyze",
                    "type": "claude_code",
                    "params": {
                        "role": "analyst",
                        "prompt": "Perform statistical analysis on: {{data_summary}} to answer: {{analysis_goals}}",
                    },
                    "depends_on": "load_and_explore",
                    "outputs": ["analysis"],
                },
                {
                    "id": "visualize",
                    "type": "claude_code",
                    "params": {
                        "role": "visualizer",
                        "prompt": "Create visualizations for: {{analysis}}",
                    },
                    "depends_on": "analyze",
                    "outputs": ["visualizations"],
                },
                {
                    "id": "report",
                    "type": "claude_code",
                    "params": {
                        "role": "reporter",
                        "prompt": "Generate executive summary from: {{analysis}} and {{visualizations}}",
                    },
                    "depends_on": ["analyze", "visualize"],
                    "outputs": ["report"],
                },
            ],
        },
    },
    "bug_fix": {
        "name": "Bug Fix Workflow",
        "icon": "\U0001f41b",
        "description": "Diagnose, fix, and verify bug resolution",
        "workflow": {
            "name": "Bug Fix Workflow",
            "version": "1.0",
            "description": "Systematic bug diagnosis, fix implementation, and verification.",
            "token_budget": 80000,
            "timeout_seconds": 3600,
            "inputs": {
                "bug_report": {
                    "type": "string",
                    "required": True,
                    "description": "Description of the bug",
                },
                "relevant_code": {
                    "type": "string",
                    "required": False,
                    "description": "Code where bug might be located",
                },
            },
            "outputs": ["diagnosis", "fix", "verification"],
            "steps": [
                {
                    "id": "diagnose",
                    "type": "claude_code",
                    "params": {
                        "role": "analyst",
                        "prompt": "Diagnose the root cause of: {{bug_report}}. Context: {{relevant_code}}",
                    },
                    "outputs": ["diagnosis"],
                },
                {
                    "id": "fix",
                    "type": "claude_code",
                    "params": {
                        "role": "builder",
                        "prompt": "Implement a fix based on diagnosis: {{diagnosis}}",
                    },
                    "depends_on": "diagnose",
                    "outputs": ["fix"],
                },
                {
                    "id": "test_fix",
                    "type": "claude_code",
                    "params": {
                        "role": "tester",
                        "prompt": "Write tests to verify the fix works: {{fix}}",
                    },
                    "depends_on": "fix",
                    "outputs": ["tests"],
                },
                {
                    "id": "verify",
                    "type": "claude_code",
                    "params": {
                        "role": "reviewer",
                        "prompt": "Verify fix is complete and doesn't introduce regressions: {{fix}} {{tests}}",
                    },
                    "depends_on": "test_fix",
                    "outputs": ["verification"],
                },
            ],
        },
    },
    "shell_pipeline": {
        "name": "Shell Command Pipeline",
        "icon": "\U0001f4bb",
        "description": "Execute shell commands with AI-assisted analysis",
        "workflow": {
            "name": "Shell Command Pipeline",
            "version": "1.0",
            "description": "Run shell commands and analyze the output with AI.",
            "token_budget": 50000,
            "timeout_seconds": 1800,
            "inputs": {
                "command": {
                    "type": "string",
                    "required": True,
                    "description": "Shell command to execute",
                },
                "analysis_prompt": {
                    "type": "string",
                    "required": False,
                    "description": "What to analyze in the output",
                },
            },
            "outputs": ["command_output", "analysis"],
            "steps": [
                {
                    "id": "run_command",
                    "type": "shell",
                    "params": {"command": "{{command}}", "allow_failure": False},
                    "outputs": ["command_output"],
                },
                {
                    "id": "analyze_output",
                    "type": "claude_code",
                    "params": {
                        "role": "analyst",
                        "prompt": "Analyze this command output: {{command_output}}. {{analysis_prompt}}",
                    },
                    "depends_on": "run_command",
                    "outputs": ["analysis"],
                },
            ],
        },
    },
}


def _get_workflow_templates() -> dict:
    """Get all available workflow templates."""
    return WORKFLOW_TEMPLATES
