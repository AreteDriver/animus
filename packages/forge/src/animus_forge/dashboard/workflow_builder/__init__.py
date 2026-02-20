"""Visual Workflow Builder for Streamlit dashboard.

This module is a backward-compatibility shim. All code has been refactored
into focused submodules:

- constants.py: NODE_TYPE_CONFIG, AGENT_ROLES, WORKFLOW_TEMPLATES
- state.py: Session state management functions
- yaml_ops.py: YAML conversion operations
- persistence.py: File persistence operations
- renderers.py: UI rendering functions
- builder.py: Main entry point (render_workflow_builder)
"""

# get_settings must be importable from this package for test patch compatibility:
#   monkeypatch.setattr("animus_forge.dashboard.workflow_builder.get_settings", ...)
from animus_forge.config import get_settings  # noqa: F401
from animus_forge.workflow.loader import (  # noqa: F401
    VALID_ON_FAILURE,
    VALID_OPERATORS,
    validate_workflow,
)

from .builder import render_workflow_builder  # noqa: F401
from .constants import (  # noqa: F401
    AGENT_ROLES,
    NODE_TYPE_CONFIG,
    WORKFLOW_TEMPLATES,
    _get_workflow_templates,
)
from .persistence import (  # noqa: F401
    _delete_workflow,
    _get_builder_state_path,
    _get_workflows_dir,
    _list_saved_workflows,
    _load_builder_state,
    _load_workflow_from_file,
    _save_builder_state,
    _save_workflow_yaml,
)
from .renderers import (  # noqa: F401
    _get_node_execution_status,
    _render_canvas,
    _render_execute_preview,
    _render_import_section,
    _render_node_card,
    _render_node_config,
    _render_node_palette,
    _render_saved_workflows,
    _render_templates_section,
    _render_visual_graph,
    _render_workflow_settings,
    _render_yaml_preview,
)
from .state import (  # noqa: F401
    _add_edge,
    _add_node,
    _delete_edge,
    _delete_node,
    _generate_node_id,
    _init_session_state,
    _mark_dirty,
    _new_workflow,
)
from .yaml_ops import _build_yaml_from_state, _load_yaml_to_state  # noqa: F401
