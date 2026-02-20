"""Metrics and Observability.

Provides instrumentation for workflow execution with support for
Prometheus metrics, push gateway, and Grafana dashboards.
"""

from .audit_checks import register_default_checks
from .collector import (
    MetricsCollector,
    StepMetrics,
    WorkflowMetrics,
    get_collector,
)
from .debt_monitor import (
    AuditCheck,
    AuditFrequency,
    AuditResult,
    AuditStatus,
    DebtSeverity,
    DebtSource,
    DebtStatus,
    SystemAuditor,
    SystemBaseline,
    TechnicalDebt,
    TechnicalDebtRegistry,
    capture_baseline,
    load_active_baseline,
    save_baseline,
)
from .exporters import (
    FileExporter,
    JsonExporter,
    LogExporter,
    MetricsExporter,
    PrometheusExporter,
    create_exporter,
)
from .prometheus_server import (
    MetricsPusher,
    PrometheusMetricsServer,
    PrometheusPushGateway,
    get_grafana_dashboard,
)

__all__ = [
    # Collector
    "MetricsCollector",
    "WorkflowMetrics",
    "StepMetrics",
    "get_collector",
    # Exporters
    "MetricsExporter",
    "PrometheusExporter",
    "JsonExporter",
    "LogExporter",
    "FileExporter",
    "create_exporter",
    # Prometheus
    "PrometheusMetricsServer",
    "PrometheusPushGateway",
    "MetricsPusher",
    "get_grafana_dashboard",
    # Debt Monitoring
    "AuditCheck",
    "AuditFrequency",
    "AuditResult",
    "AuditStatus",
    "DebtSeverity",
    "DebtSource",
    "DebtStatus",
    "SystemAuditor",
    "SystemBaseline",
    "TechnicalDebt",
    "TechnicalDebtRegistry",
    "capture_baseline",
    "load_active_baseline",
    "save_baseline",
    "register_default_checks",
]
