"""Initial schema — object registry and event ledger

Revision ID: 001
Revises: 
Create Date: 2026-06-27 07:30:00.000000+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[Sequence[str], None] = None
depends_on: Union[Sequence[str], None] = None


def upgrade() -> None:
    # --- Object Registry --------------------------------------------------
    # Stores canonical objects (actions, events, assessments, etc.)
    # with bitemporal validity tracking.
    op.create_table(
        'object_registry',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('object_id', sa.String(128), nullable=False),
        sa.Column('object_version', sa.Integer, nullable=False, default=1),
        sa.Column('schema_id', sa.String(256), nullable=False),
        sa.Column('schema_version', sa.String(32), nullable=False),
        sa.Column('owner_id', sa.String(128), nullable=False),
        sa.Column('workspace_id', sa.String(128), nullable=False),
        sa.Column('subject_domain', sa.String(32), nullable=False),
        sa.Column('artifact_type', sa.String(64), nullable=False),
        sa.Column('cognitive_role', sa.String(32), nullable=False),
        sa.Column('workflow_status', sa.String(32), nullable=False),
        sa.Column('epistemic_status', sa.String(32), nullable=False),
        sa.Column('lifecycle_status', sa.String(32), nullable=False),
        sa.Column('storage_tier', sa.String(16), nullable=False),
        sa.Column('presentation', sa.String(32), nullable=False),
        sa.Column('security_class', sa.String(32), nullable=False),
        # Bitemporal: valid time (world time)
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=True),
        sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
        # Bitemporal: transaction time (system time)
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
        # Provenance
        sa.Column('created_by', sa.String(256), nullable=False),
        sa.Column('trace_id', sa.String(256), nullable=True),
        sa.Column('content_sha256', sa.String(64), nullable=False),
        # Payload
        sa.Column('payload', sa.JSON, nullable=False),
        # Indexes
        sa.Index('idx_object_id_version', 'object_id', 'object_version', unique=True),
        sa.Index('idx_artifact_type', 'artifact_type'),
        sa.Index('idx_subject_domain', 'subject_domain'),
        sa.Index('idx_valid_from', 'valid_from'),
        sa.Index('idx_recorded_at', 'recorded_at'),
        sa.Index('idx_trace_id', 'trace_id'),
    )

    # --- Event Ledger -----------------------------------------------------
    # Append-only log of all significant system events.
    # Every mutation to object_registry produces an event here.
    op.create_table(
        'event_ledger',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('event_kind', sa.String(128), nullable=False),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('actor_refs', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('object_refs', sa.JSON, nullable=False, server_default='[]'),
        sa.Column('event_data', sa.JSON, nullable=False, server_default='{}'),
        sa.Column('idempotency_key', sa.String(256), nullable=True),
        # Bitemporal: when the event was valid in world time
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=True),
        sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
        # Transaction time: when the event was recorded
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Index('idx_event_kind', 'event_kind'),
        sa.Index('idx_occurred_at', 'occurred_at'),
        sa.Index('idx_idempotency_key', 'idempotency_key', unique=True),
    )

    # --- Traceability -----------------------------------------------------
    # Links requirements to tests to evidence bundles.
    op.create_table(
        'traceability',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('requirement_id', sa.String(128), nullable=False),
        sa.Column('test_path', sa.String(512), nullable=False),
        sa.Column('test_status', sa.String(32), nullable=False),
        sa.Column('evidence_bundle', sa.String(256), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Index('idx_req_id', 'requirement_id'),
    )


def downgrade() -> None:
    op.drop_table('traceability')
    op.drop_table('event_ledger')
    op.drop_table('object_registry')
