"""Plugin Marketplace UI for Streamlit dashboard.

A visual marketplace for browsing, installing, and managing plugins
in the Gorgon workflow orchestrator.
"""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from animus_forge.plugins.models import (
    PluginCategory,
)

logger = logging.getLogger(__name__)

# Category icons and colors
CATEGORY_CONFIG = {
    PluginCategory.INTEGRATION: {
        "icon": "🔌",
        "color": "#3b82f6",
        "label": "Integration",
    },
    PluginCategory.DATA_TRANSFORM: {
        "icon": "🔄",
        "color": "#10b981",
        "label": "Data Transform",
    },
    PluginCategory.MONITORING: {
        "icon": "📊",
        "color": "#f59e0b",
        "label": "Monitoring",
    },
    PluginCategory.SECURITY: {"icon": "🔒", "color": "#ef4444", "label": "Security"},
    PluginCategory.WORKFLOW: {"icon": "⚙️", "color": "#8b5cf6", "label": "Workflow"},
    PluginCategory.AI_PROVIDER: {
        "icon": "🤖",
        "color": "#ec4899",
        "label": "AI Provider",
    },
    PluginCategory.STORAGE: {"icon": "💾", "color": "#6366f1", "label": "Storage"},
    PluginCategory.NOTIFICATION: {
        "icon": "🔔",
        "color": "#14b8a6",
        "label": "Notification",
    },
    PluginCategory.ANALYTICS: {"icon": "📈", "color": "#f97316", "label": "Analytics"},
    PluginCategory.OTHER: {"icon": "📦", "color": "#6b7280", "label": "Other"},
}

# Sample plugins for demonstration (would come from marketplace in production)
SAMPLE_PLUGINS: list[dict] = [
    {
        "id": "github-integration",
        "name": "github-integration",
        "display_name": "GitHub Integration",
        "description": "Create issues, PRs, and manage repositories directly from workflows",
        "author": "Gorgon Team",
        "category": PluginCategory.INTEGRATION,
        "tags": ["github", "git", "vcs", "issues", "pull-requests"],
        "downloads": 15420,
        "rating": 4.8,
        "review_count": 234,
        "latest_version": "2.1.0",
        "verified": True,
        "featured": True,
    },
    {
        "id": "slack-notifications",
        "name": "slack-notifications",
        "display_name": "Slack Notifications",
        "description": "Send workflow notifications and alerts to Slack channels",
        "author": "Gorgon Team",
        "category": PluginCategory.NOTIFICATION,
        "tags": ["slack", "notifications", "alerts", "messaging"],
        "downloads": 12350,
        "rating": 4.7,
        "review_count": 189,
        "latest_version": "1.5.2",
        "verified": True,
        "featured": True,
    },
    {
        "id": "openai-advanced",
        "name": "openai-advanced",
        "display_name": "OpenAI Advanced",
        "description": "Advanced OpenAI features including function calling and assistants API",
        "author": "AI Plugins Inc",
        "category": PluginCategory.AI_PROVIDER,
        "tags": ["openai", "gpt-4", "ai", "llm", "function-calling"],
        "downloads": 8920,
        "rating": 4.6,
        "review_count": 156,
        "latest_version": "3.0.1",
        "verified": True,
        "featured": False,
    },
    {
        "id": "prometheus-exporter",
        "name": "prometheus-exporter",
        "display_name": "Prometheus Exporter",
        "description": "Export workflow metrics to Prometheus for monitoring",
        "author": "Metrics Lab",
        "category": PluginCategory.MONITORING,
        "tags": ["prometheus", "metrics", "monitoring", "observability"],
        "downloads": 6780,
        "rating": 4.5,
        "review_count": 98,
        "latest_version": "1.2.0",
        "verified": True,
        "featured": False,
    },
    {
        "id": "s3-storage",
        "name": "s3-storage",
        "display_name": "AWS S3 Storage",
        "description": "Store workflow artifacts and checkpoints in Amazon S3",
        "author": "Cloud Plugins",
        "category": PluginCategory.STORAGE,
        "tags": ["aws", "s3", "storage", "cloud", "artifacts"],
        "downloads": 5430,
        "rating": 4.4,
        "review_count": 87,
        "latest_version": "2.0.0",
        "verified": False,
        "featured": False,
    },
    {
        "id": "data-validator",
        "name": "data-validator",
        "display_name": "Data Validator",
        "description": "Validate and transform data between workflow steps",
        "author": "DataFlow",
        "category": PluginCategory.DATA_TRANSFORM,
        "tags": ["validation", "transform", "schema", "data-quality"],
        "downloads": 4210,
        "rating": 4.3,
        "review_count": 65,
        "latest_version": "1.1.0",
        "verified": False,
        "featured": False,
    },
    {
        "id": "security-scanner",
        "name": "security-scanner",
        "display_name": "Security Scanner",
        "description": "Scan code and dependencies for security vulnerabilities",
        "author": "SecureCode",
        "category": PluginCategory.SECURITY,
        "tags": ["security", "vulnerabilities", "scanning", "sast"],
        "downloads": 3890,
        "rating": 4.6,
        "review_count": 72,
        "latest_version": "1.3.0",
        "verified": True,
        "featured": False,
    },
    {
        "id": "workflow-analytics",
        "name": "workflow-analytics",
        "display_name": "Workflow Analytics",
        "description": "Advanced analytics and insights for workflow execution",
        "author": "Analytics Pro",
        "category": PluginCategory.ANALYTICS,
        "tags": ["analytics", "insights", "reporting", "dashboards"],
        "downloads": 2980,
        "rating": 4.2,
        "review_count": 45,
        "latest_version": "0.9.0",
        "verified": False,
        "featured": False,
    },
]

# Sample installed plugins
SAMPLE_INSTALLED: list[dict] = [
    {
        "id": "inst-1",
        "plugin_name": "github-integration",
        "version": "2.1.0",
        "installed_at": datetime(2024, 1, 15, 10, 30),
        "enabled": True,
        "source": "marketplace",
    },
    {
        "id": "inst-2",
        "plugin_name": "slack-notifications",
        "version": "1.5.0",
        "installed_at": datetime(2024, 2, 1, 14, 0),
        "enabled": True,
        "source": "marketplace",
    },
]


def _init_marketplace_state():
    """Initialize session state for marketplace."""
    if "marketplace_search" not in st.session_state:
        st.session_state.marketplace_search = ""
    if "marketplace_category" not in st.session_state:
        st.session_state.marketplace_category = "all"
    if "marketplace_selected_plugin" not in st.session_state:
        st.session_state.marketplace_selected_plugin = None
    if "installed_plugins" not in st.session_state:
        st.session_state.installed_plugins = {p["plugin_name"]: p for p in SAMPLE_INSTALLED}


def _get_filtered_plugins() -> list[dict]:
    """Get plugins filtered by search and category."""
    plugins = SAMPLE_PLUGINS
    search = st.session_state.marketplace_search.lower()
    category = st.session_state.marketplace_category

    filtered = []
    for plugin in plugins:
        # Category filter
        if category != "all" and plugin["category"].value != category:
            continue

        # Search filter
        if search:
            searchable = f"{plugin['name']} {plugin['display_name']} {plugin['description']} {' '.join(plugin['tags'])}".lower()
            if search not in searchable:
                continue

        filtered.append(plugin)

    return filtered


def _is_installed(plugin_name: str) -> bool:
    """Check if a plugin is installed."""
    return plugin_name in st.session_state.installed_plugins


def _get_installed_version(plugin_name: str) -> str | None:
    """Get installed version of a plugin."""
    installed = st.session_state.installed_plugins.get(plugin_name)
    return installed["version"] if installed else None


def _render_rating_stars(rating: float) -> str:
    """Render rating as stars HTML."""
    full_stars = int(rating)
    half_star = rating - full_stars >= 0.5
    empty_stars = 5 - full_stars - (1 if half_star else 0)

    stars = "★" * full_stars
    if half_star:
        stars += "½"
    stars += "☆" * empty_stars

    return f'<span style="color: #f59e0b;">{stars}</span> <span style="color: #6b7280;">({rating:.1f})</span>'


def _render_plugin_card(plugin: dict) -> None:
    """Render a plugin card."""
    config = CATEGORY_CONFIG.get(plugin["category"], CATEGORY_CONFIG[PluginCategory.OTHER])
    is_installed = _is_installed(plugin["name"])
    installed_version = _get_installed_version(plugin["name"])

    # Status badges
    badges_html = ""
    if plugin.get("verified"):
        badges_html += '<span style="background: #dbeafe; color: #1d4ed8; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-right: 4px;">✓ Verified</span>'
    if plugin.get("featured"):
        badges_html += '<span style="background: #fef3c7; color: #b45309; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-right: 4px;">⭐ Featured</span>'
    if is_installed:
        badges_html += f'<span style="background: #d1fae5; color: #047857; padding: 2px 8px; border-radius: 10px; font-size: 10px;">✓ Installed v{installed_version}</span>'

    # Update available badge
    if is_installed and installed_version != plugin["latest_version"]:
        badges_html += '<span style="background: #fee2e2; color: #b91c1c; padding: 2px 8px; border-radius: 10px; font-size: 10px; margin-left: 4px;">↑ Update Available</span>'

    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 20px;
            margin: 12px 0;
            background: linear-gradient(145deg, white, {config["color"]}05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.2s ease;
        ">
            <div style="display: flex; align-items: flex-start; gap: 16px;">
                <div style="
                    font-size: 36px;
                    background: {config["color"]}15;
                    padding: 12px;
                    border-radius: 14px;
                    line-height: 1;
                ">{config["icon"]}</div>
                <div style="flex: 1; min-width: 0;">
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;">
                        <span style="font-weight: 700; font-size: 17px; color: #1f2937;">{plugin["display_name"]}</span>
                        <span style="
                            background: {config["color"]}20;
                            color: {config["color"]};
                            padding: 2px 10px;
                            border-radius: 10px;
                            font-size: 11px;
                            font-weight: 600;
                        ">{config["label"]}</span>
                    </div>
                    <div style="font-size: 13px; color: #6b7280; margin-bottom: 8px;">
                        by <strong>{plugin["author"]}</strong> • v{plugin["latest_version"]}
                    </div>
                    <div style="font-size: 14px; color: #4b5563; margin-bottom: 12px;">
                        {plugin["description"]}
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                        <span style="font-size: 13px;">{_render_rating_stars(plugin["rating"])}</span>
                        <span style="font-size: 12px; color: #6b7280;">💬 {plugin["review_count"]} reviews</span>
                        <span style="font-size: 12px; color: #6b7280;">📥 {plugin["downloads"]:,} downloads</span>
                    </div>
                    <div style="margin-top: 8px;">
                        {badges_html}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("📖 Details", key=f"details_{plugin['name']}", use_container_width=True):
            st.session_state.marketplace_selected_plugin = plugin["name"]
            st.rerun()

    with col2:
        if is_installed:
            installed = st.session_state.installed_plugins[plugin["name"]]
            if installed.get("enabled", True):
                if st.button(
                    "⏸️ Disable",
                    key=f"disable_{plugin['name']}",
                    use_container_width=True,
                ):
                    st.session_state.installed_plugins[plugin["name"]]["enabled"] = False
                    st.toast(f"Disabled {plugin['display_name']}")
                    st.rerun()
            else:
                if st.button("▶️ Enable", key=f"enable_{plugin['name']}", use_container_width=True):
                    st.session_state.installed_plugins[plugin["name"]]["enabled"] = True
                    st.toast(f"Enabled {plugin['display_name']}")
                    st.rerun()
        else:
            if st.button(
                "📦 Install",
                key=f"install_{plugin['name']}",
                use_container_width=True,
                type="primary",
            ):
                # Simulate installation
                st.session_state.installed_plugins[plugin["name"]] = {
                    "id": f"inst-{plugin['name']}",
                    "plugin_name": plugin["name"],
                    "version": plugin["latest_version"],
                    "installed_at": datetime.now(),
                    "enabled": True,
                    "source": "marketplace",
                }
                st.toast(f"Installed {plugin['display_name']} v{plugin['latest_version']}")
                st.rerun()


def _render_plugin_details(plugin_name: str) -> None:
    """Render detailed view of a plugin."""
    plugin = next((p for p in SAMPLE_PLUGINS if p["name"] == plugin_name), None)
    if not plugin:
        st.error("Plugin not found")
        return

    config = CATEGORY_CONFIG.get(plugin["category"], CATEGORY_CONFIG[PluginCategory.OTHER])
    is_installed = _is_installed(plugin["name"])
    installed_version = _get_installed_version(plugin["name"])

    # Back button
    if st.button("← Back to Marketplace"):
        st.session_state.marketplace_selected_plugin = None
        st.rerun()

    st.markdown("---")

    # Header
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="
                    font-size: 56px;
                    background: {config["color"]}15;
                    padding: 20px;
                    border-radius: 20px;
                    line-height: 1;
                ">{config["icon"]}</div>
                <div>
                    <h1 style="margin: 0; font-size: 32px;">{plugin["display_name"]}</h1>
                    <p style="margin: 4px 0; color: #6b7280; font-size: 16px;">
                        by <strong>{plugin["author"]}</strong>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if is_installed:
            st.success(f"✓ Installed v{installed_version}")
            if installed_version != plugin["latest_version"]:
                if st.button("⬆️ Update", use_container_width=True, type="primary"):
                    st.session_state.installed_plugins[plugin["name"]]["version"] = plugin[
                        "latest_version"
                    ]
                    st.toast(f"Updated to v{plugin['latest_version']}")
                    st.rerun()
            if st.button("🗑️ Uninstall", use_container_width=True):
                del st.session_state.installed_plugins[plugin["name"]]
                st.toast(f"Uninstalled {plugin['display_name']}")
                st.rerun()
        else:
            if st.button("📦 Install Plugin", use_container_width=True, type="primary"):
                st.session_state.installed_plugins[plugin["name"]] = {
                    "id": f"inst-{plugin['name']}",
                    "plugin_name": plugin["name"],
                    "version": plugin["latest_version"],
                    "installed_at": datetime.now(),
                    "enabled": True,
                    "source": "marketplace",
                }
                st.toast(f"Installed {plugin['display_name']}")
                st.rerun()

    # Stats row
    st.markdown(
        f"""
        <div style="
            display: flex;
            gap: 32px;
            padding: 20px;
            background: #f9fafb;
            border-radius: 12px;
            margin: 20px 0;
        ">
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #1f2937;">{plugin["downloads"]:,}</div>
                <div style="font-size: 12px; color: #6b7280;">Downloads</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #f59e0b;">★ {plugin["rating"]}</div>
                <div style="font-size: 12px; color: #6b7280;">{plugin["review_count"]} Reviews</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: #1f2937;">v{plugin["latest_version"]}</div>
                <div style="font-size: 12px; color: #6b7280;">Latest Version</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px;">{config["icon"]}</div>
                <div style="font-size: 12px; color: #6b7280;">{config["label"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Description
    st.markdown("### About")
    st.markdown(plugin["description"])

    # Tags
    st.markdown("### Tags")
    tags_html = " ".join(
        [
            f'<span style="background: #e5e7eb; padding: 4px 12px; border-radius: 16px; font-size: 13px; margin-right: 8px;">{tag}</span>'
            for tag in plugin["tags"]
        ]
    )
    st.markdown(tags_html, unsafe_allow_html=True)

    # Badges
    st.markdown("### Badges")
    badges = []
    if plugin.get("verified"):
        badges.append(("✓ Verified", "Verified by Gorgon team", "#dbeafe", "#1d4ed8"))
    if plugin.get("featured"):
        badges.append(("⭐ Featured", "Featured plugin", "#fef3c7", "#b45309"))

    if badges:
        for label, desc, bg, color in badges:
            st.markdown(
                f"""
                <div style="
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    background: {bg};
                    color: {color};
                    padding: 8px 16px;
                    border-radius: 8px;
                    margin-right: 8px;
                    margin-bottom: 8px;
                ">
                    <strong>{label}</strong>
                    <span style="font-size: 12px;">— {desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No special badges")


def _render_installed_plugins() -> None:
    """Render installed plugins list."""
    st.markdown("### 📦 Installed Plugins")

    installed = st.session_state.installed_plugins
    if not installed:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 48px 24px;
                background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
                border-radius: 20px;
                border: 2px dashed #7dd3fc;
            ">
                <div style="font-size: 48px; margin-bottom: 16px;">📭</div>
                <div style="font-size: 18px; font-weight: 600; color: #0369a1;">No Plugins Installed</div>
                <div style="font-size: 14px; color: #0284c7; margin-top: 8px;">
                    Browse the marketplace to find useful plugins
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        return

    for plugin_name, installation in installed.items():
        plugin = next((p for p in SAMPLE_PLUGINS if p["name"] == plugin_name), None)
        if not plugin:
            continue

        config = CATEGORY_CONFIG.get(plugin["category"], CATEGORY_CONFIG[PluginCategory.OTHER])
        is_enabled = installation.get("enabled", True)
        has_update = installation["version"] != plugin["latest_version"]

        st.markdown(
            f"""
            <div style="
                border: 1px solid {"#d1d5db" if is_enabled else "#fca5a5"};
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
                background: {"white" if is_enabled else "#fef2f2"};
                opacity: {1 if is_enabled else 0.7};
            ">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 28px;">{config["icon"]}</span>
                    <div style="flex: 1;">
                        <div style="font-weight: 600;">{plugin["display_name"]}</div>
                        <div style="font-size: 12px; color: #6b7280;">
                            v{installation["version"]}
                            {' • <span style="color: #b91c1c;">Update available (v' + plugin["latest_version"] + ")</span>" if has_update else ""}
                            {' • <span style="color: #b91c1c;">Disabled</span>' if not is_enabled else ""}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if is_enabled:
                if st.button(
                    "⏸️ Disable",
                    key=f"inst_disable_{plugin_name}",
                    use_container_width=True,
                ):
                    st.session_state.installed_plugins[plugin_name]["enabled"] = False
                    st.rerun()
            else:
                if st.button(
                    "▶️ Enable",
                    key=f"inst_enable_{plugin_name}",
                    use_container_width=True,
                ):
                    st.session_state.installed_plugins[plugin_name]["enabled"] = True
                    st.rerun()
        with col2:
            if has_update:
                if st.button(
                    "⬆️ Update",
                    key=f"inst_update_{plugin_name}",
                    use_container_width=True,
                ):
                    st.session_state.installed_plugins[plugin_name]["version"] = plugin[
                        "latest_version"
                    ]
                    st.toast(f"Updated to v{plugin['latest_version']}")
                    st.rerun()
        with col3:
            if st.button(
                "🗑️ Uninstall",
                key=f"inst_uninstall_{plugin_name}",
                use_container_width=True,
            ):
                del st.session_state.installed_plugins[plugin_name]
                st.toast(f"Uninstalled {plugin['display_name']}")
                st.rerun()


def _render_category_sidebar() -> None:
    """Render category filter sidebar."""
    st.markdown("### Categories")

    # All category
    all_count = len(SAMPLE_PLUGINS)
    if st.button(
        f"📋 All ({all_count})",
        key="cat_all",
        use_container_width=True,
        type="primary" if st.session_state.marketplace_category == "all" else "secondary",
    ):
        st.session_state.marketplace_category = "all"
        st.rerun()

    # Individual categories
    category_counts = {}
    for plugin in SAMPLE_PLUGINS:
        cat = plugin["category"].value
        category_counts[cat] = category_counts.get(cat, 0) + 1

    for category in PluginCategory:
        count = category_counts.get(category.value, 0)
        if count == 0:
            continue
        config = CATEGORY_CONFIG[category]
        is_selected = st.session_state.marketplace_category == category.value
        if st.button(
            f"{config['icon']} {config['label']} ({count})",
            key=f"cat_{category.value}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state.marketplace_category = category.value
            st.rerun()


def render_plugin_marketplace():
    """Main entry point for the plugin marketplace UI."""
    st.title("🏪 Plugin Marketplace")

    _init_marketplace_state()

    # Check if viewing plugin details
    if st.session_state.marketplace_selected_plugin:
        _render_plugin_details(st.session_state.marketplace_selected_plugin)
        return

    # Main layout
    sidebar, main = st.columns([1, 3])

    with sidebar:
        # Search
        st.markdown("### 🔍 Search")
        search = st.text_input(
            "Search plugins",
            value=st.session_state.marketplace_search,
            placeholder="Search by name, description...",
            label_visibility="collapsed",
        )
        if search != st.session_state.marketplace_search:
            st.session_state.marketplace_search = search
            st.rerun()

        st.divider()

        # Categories
        _render_category_sidebar()

        st.divider()

        # Installed
        _render_installed_plugins()

    with main:
        # Stats bar
        filtered = _get_filtered_plugins()
        installed_count = len(st.session_state.installed_plugins)
        enabled_count = sum(
            1 for p in st.session_state.installed_plugins.values() if p.get("enabled", True)
        )

        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: space-between;
                padding: 12px 20px;
                background: #f9fafb;
                border-radius: 12px;
                margin-bottom: 16px;
            ">
                <span style="font-size: 14px; color: #6b7280;">
                    📊 <strong>{len(filtered)}</strong> plugins found
                </span>
                <span style="font-size: 14px; color: #6b7280;">
                    📦 <strong>{installed_count}</strong> installed
                    • ▶️ <strong>{enabled_count}</strong> enabled
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Featured section (only when no filters)
        if (
            not st.session_state.marketplace_search
            and st.session_state.marketplace_category == "all"
        ):
            featured = [p for p in SAMPLE_PLUGINS if p.get("featured")]
            if featured:
                st.markdown("### ⭐ Featured Plugins")
                for plugin in featured[:3]:
                    _render_plugin_card(plugin)
                st.divider()

        # All plugins
        st.markdown("### 📋 All Plugins")

        if not filtered:
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 48px 24px;
                    background: #f9fafb;
                    border-radius: 16px;
                ">
                    <div style="font-size: 48px; margin-bottom: 16px;">🔍</div>
                    <div style="font-size: 18px; font-weight: 600; color: #6b7280;">No Plugins Found</div>
                    <div style="font-size: 14px; color: #9ca3af; margin-top: 8px;">
                        Try adjusting your search or filters
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            for plugin in filtered:
                _render_plugin_card(plugin)
