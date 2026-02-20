"""Workflow I/O rendering — YAML preview, import, saved workflows, templates."""

from __future__ import annotations

import streamlit as st
import yaml

from animus_forge.workflow.loader import validate_workflow

from ..constants import _get_workflow_templates
from ..persistence import (
    _delete_workflow,
    _get_workflows_dir,
    _list_saved_workflows,
    _load_workflow_from_file,
    _save_workflow_yaml,
)
from ..state import _mark_dirty
from ..yaml_ops import _build_yaml_from_state, _load_yaml_to_state


def _render_yaml_preview() -> None:
    """Render YAML preview and export options."""
    st.markdown("### YAML Preview")

    workflow = _build_yaml_from_state()
    yaml_str = yaml.dump(workflow, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Validate
    errors = validate_workflow(workflow)
    if errors:
        st.error("Validation Errors:")
        for error in errors:
            st.markdown(f"- {error}")
    else:
        st.success("Workflow is valid!")

    # Show YAML
    st.code(yaml_str, language="yaml")

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download YAML",
            yaml_str,
            file_name=f"{workflow['name'].lower().replace(' ', '-')}.yaml",
            mime="text/yaml",
        )

    with col2:
        # Save to workflows directory
        if st.button("Save to Workflows", disabled=bool(errors)):
            filepath = _save_workflow_yaml()
            if filepath:
                st.success(f"Saved to {filepath}")
            else:
                st.error("Failed to save workflow")


def _render_import_section() -> None:
    """Render YAML import section."""
    st.markdown("### Import Workflow")

    tab1, tab2 = st.tabs(["Upload File", "Paste YAML"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload YAML workflow",
            type=["yaml", "yml"],
            key="yaml_upload",
        )

        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                data = yaml.safe_load(content)

                if st.button("Import Workflow", key="import_upload"):
                    _load_yaml_to_state(data)
                    st.success("Workflow imported!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse YAML: {e}")

    with tab2:
        yaml_input = st.text_area(
            "Paste YAML here",
            height=200,
            key="yaml_paste",
        )

        if yaml_input:
            if st.button("Import Workflow", key="import_paste"):
                try:
                    data = yaml.safe_load(yaml_input)
                    _load_yaml_to_state(data)
                    st.success("Workflow imported!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to parse YAML: {e}")

    # Load from existing workflows
    st.markdown("#### Or Load Existing Workflow")

    workflows_dir = _get_workflows_dir()
    if workflows_dir.exists():
        yaml_files = list(workflows_dir.glob("*.yaml"))
        if yaml_files:
            selected_file = st.selectbox(
                "Select workflow",
                yaml_files,
                format_func=lambda x: x.stem,
                key="existing_workflow_select",
            )

            if st.button("Load Workflow", key="load_existing"):
                if _load_workflow_from_file(selected_file):
                    st.success(f"Loaded {selected_file.name}")
                    st.rerun()
                else:
                    st.error("Failed to load workflow")


def _render_saved_workflows() -> None:
    """Render saved workflows management section."""
    st.markdown("### Saved Workflows")

    workflows = _list_saved_workflows()

    if not workflows:
        st.info("No saved workflows yet. Create one and save it!")
        return

    for wf in workflows:
        with st.expander(f"**{wf['name']}** v{wf['version']}", expanded=False):
            st.caption(wf["description"] or "No description")
            st.markdown(f"Steps: {wf['steps']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key=f"load_{wf['path']}", use_container_width=True):
                    if _load_workflow_from_file(wf["path"]):
                        st.success(f"Loaded {wf['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to load")
            with col2:
                if st.button("Delete", key=f"del_{wf['path']}", use_container_width=True):
                    if _delete_workflow(wf["path"]):
                        st.success(f"Deleted {wf['name']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete")


def _render_templates_section() -> None:
    """Render workflow templates selection section."""
    st.markdown("### Templates")
    st.caption("Start with a pre-built workflow pattern")

    templates = _get_workflow_templates()

    for template_id, template in templates.items():
        with st.expander(f"{template['icon']} **{template['name']}**", expanded=False):
            st.markdown(template["description"])

            workflow = template["workflow"]
            st.markdown(f"**Steps:** {len(workflow['steps'])}")

            # Show step preview
            step_names = [s["id"] for s in workflow["steps"]]
            st.caption(" \u2192 ".join(step_names))

            # Show required inputs
            inputs = workflow.get("inputs", {})
            required = [k for k, v in inputs.items() if v.get("required")]
            if required:
                st.caption(f"**Required inputs:** {', '.join(required)}")

            if st.button(
                "Use Template",
                key=f"template_{template_id}",
                use_container_width=True,
                type="primary",
            ):
                _load_yaml_to_state(workflow)
                _mark_dirty()
                st.success(f"Loaded template: {template['name']}")
                st.rerun()
