import { useState, useEffect, useCallback } from "react";
import {
  getCitizensSummary,
  getCitizenProposals,
  approveProposal,
  rejectProposal,
  commissionProposal,
  type CitizenSummary,
  type CitizenProposal,
} from "../api";
import "./Citizens.css";

const REFRESH_INTERVAL_MS = 10_000;

const STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  submitted: "Submitted",
  pending_review: "Pending",
  approved: "Approved",
  rejected: "Rejected",
  complete: "Complete",
  implemented: "Implemented",
};

const CITIZEN_NAMES: Record<string, string> = {
  architect: "Architect",
  conversation_designer: "Conversation Designer",
  knowledge_curator: "Knowledge Curator",
  test_oracle: "Test Oracle",
  session_steward: "Session Steward",
};

export function CitizensView() {
  const [summary, setSummary] = useState<CitizenSummary | null>(null);
  const [proposals, setProposals] = useState<CitizenProposal[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("");

  const refresh = useCallback(() => {
    Promise.all([getCitizensSummary(), getCitizenProposals(filter || undefined)])
      .then(([sum, props]) => {
        setSummary(sum);
        setProposals(props);
        setError(null);
        setLastChecked(new Date());
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Connection failed");
        setLastChecked(new Date());
      });
  }, [filter]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, REFRESH_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const handleApprove = async (id: string) => {
    setBusyId(id);
    try {
      await approveProposal(id);
      refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Approve failed");
    } finally {
      setBusyId(null);
    }
  };

  const handleReject = async (id: string) => {
    setBusyId(id);
    try {
      await rejectProposal(id);
      refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Reject failed");
    } finally {
      setBusyId(null);
    }
  };

  const handleCommission = async (id: string) => {
    setBusyId(id);
    try {
      await commissionProposal(id);
      refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Commission failed");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="citizens">
      <h1 className="citizens-title">Citizens</h1>

      {summary && (
        <div className="citizens-stats">
          <div className="citizens-stat">
            <span className="citizens-stat-value">{summary.citizens_total}</span>
            <span className="citizens-stat-label">Total</span>
          </div>
          <div className="citizens-stat">
            <span className="citizens-stat-value citizens-stat-value--ok">
              {summary.citizens_active}
            </span>
            <span className="citizens-stat-label">Active</span>
          </div>
          <div className="citizens-stat">
            <span className="citizens-stat-value citizens-stat-value--warn">
              {summary.proposals_pending}
            </span>
            <span className="citizens-stat-label">Pending</span>
          </div>
          <div className="citizens-stat">
            <span className="citizens-stat-value">{summary.proposals_completed}</span>
            <span className="citizens-stat-label">Done</span>
          </div>
        </div>
      )}

      {!summary?.core_available && (
        <div className="citizens-warning">
          animus-core is not installed. Citizen data is unavailable.
        </div>
      )}

      <div className="citizens-filter">
        <select
          className="citizens-select"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="">All statuses</option>
          <option value="draft">Draft</option>
          <option value="submitted">Submitted</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
          <option value="complete">Complete</option>
        </select>
      </div>

      <div className="citizens-list">
        {proposals.length === 0 && !error && (
          <div className="citizens-empty">
            <div className="citizens-empty-icon">🔍</div>
            <div className="citizens-empty-text">No proposals match.</div>
          </div>
        )}

        {proposals.map((p) => (
          <div key={p.id} className="citizens-card">
            <div className="citizens-card-header">
              <span className="citizens-card-title">{p.title}</span>
              <span
                className={`citizens-badge citizens-badge--${p.status}`}
              >
                {STATUS_LABELS[p.status] ?? p.status}
              </span>
            </div>
            <div className="citizens-card-meta">
              <span>{CITIZEN_NAMES[p.source_citizen] ?? p.source_citizen}</span>
              <span>•</span>
              <span>{p.estimated_effort_hours}h</span>
              <span>•</span>
              <span
                className={`citizens-confidence citizens-confidence--${
                  p.confidence_score >= 0.75
                    ? "high"
                    : p.confidence_score >= 0.5
                      ? "medium"
                      : "low"
                }`}
              >
                {p.confidence_label || `${Math.round(p.confidence_score * 100)}%`}
              </span>
            </div>
            {p.problem && (
              <p className="citizens-card-problem">{p.problem}</p>
            )}
            <div className="citizens-card-actions">
              {(p.status === "draft" || p.status === "submitted" || p.status === "pending_review") && (
                <>
                  <button
                    className="citizens-btn citizens-btn--approve"
                    onClick={() => handleApprove(p.id)}
                    disabled={busyId === p.id}
                  >
                    {busyId === p.id ? "…" : "Approve"}
                  </button>
                  <button
                    className="citizens-btn citizens-btn--reject"
                    onClick={() => handleReject(p.id)}
                    disabled={busyId === p.id}
                  >
                    {busyId === p.id ? "…" : "Reject"}
                  </button>
                </>
              )}
              {p.status === "approved" && (
                <button
                  className="citizens-btn citizens-btn--commission"
                  onClick={() => handleCommission(p.id)}
                  disabled={busyId === p.id}
                >
                  {busyId === p.id ? "…" : "Commission"}
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {lastChecked && (
        <p className="citizens-last-checked">
          Last updated {lastChecked.toLocaleTimeString()}
        </p>
      )}

      {error && <p className="citizens-error">{error}</p>}
    </div>
  );
}
