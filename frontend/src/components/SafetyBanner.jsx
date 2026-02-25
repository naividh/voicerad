import React from "react";

/**
 * SafetyBanner - Visual clinical safety rail UI
 *
 * Displays triage level, confidence indicator, referral alerts,
 * and human-review flags based on the safety assessment from the backend.
 *
 * Color coding:
 *   CRITICAL = red pulse animation
 *   URGENT   = orange
 *   ROUTINE  = yellow
 *   NORMAL   = green
 *
 * Confidence:
 *   HIGH     = green bar
 *   MODERATE = yellow bar
 *   LOW      = orange bar
 *   VERY_LOW = red bar (blocked)
 */

const TRIAGE_STYLES = {
  CRITICAL: {
    bg: "#fee2e2",
    border: "#ef4444",
    text: "#991b1b",
    icon: "🚨",
    label: "CRITICAL - Immediate Review Required",
    pulse: true,
  },
  URGENT: {
    bg: "#fff7ed",
    border: "#f97316",
    text: "#9a3412",
    icon: "⚠️",
    label: "URGENT - Review Within 1 Hour",
    pulse: false,
  },
  ROUTINE: {
    bg: "#fefce8",
    border: "#eab308",
    text: "#854d0e",
    icon: "📋",
    label: "ROUTINE - Standard Review Queue",
    pulse: false,
  },
  NORMAL: {
    bg: "#f0fdf4",
    border: "#22c55e",
    text: "#166534",
    icon: "✅",
    label: "NORMAL - No Acute Findings",
    pulse: false,
  },
};

const CONFIDENCE_COLORS = {
  HIGH: "#22c55e",
  MODERATE: "#eab308",
  LOW: "#f97316",
  VERY_LOW: "#ef4444",
};

function ConfidenceBar({ score, label }) {
  const pct = Math.round(score * 100);
  const color = CONFIDENCE_COLORS[label] || "#999";

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: 12,
        marginBottom: 2,
      }}>
        <span>AI Confidence</span>
        <span style={{ fontWeight: 600, color }}>
          {pct}% ({label})
        </span>
      </div>
      <div style={{
        height: 6,
        background: "#e5e7eb",
        borderRadius: 3,
        overflow: "hidden",
      }}>
        <div style={{
          width: pct + "%",
          height: "100%",
          background: color,
          borderRadius: 3,
          transition: "width 0.5s ease",
        }} />
      </div>
    </div>
  );
}

function ReferralAlert({ safety }) {
  if (!safety.referral_triggered) return null;

  const isImmediate = safety.referral_type === "IMMEDIATE";
  return (
    <div style={{
      marginTop: 8,
      padding: "8px 12px",
      background: isImmediate ? "#fef2f2" : "#fffbeb",
      border: "1px solid " + (isImmediate ? "#fca5a5" : "#fcd34d"),
      borderRadius: 6,
      fontSize: 13,
    }}>
      <strong>
        {isImmediate ? "🔴 IMMEDIATE REFERRAL" : "🟠 URGENT REFERRAL"}
      </strong>
      <div style={{ marginTop: 4 }}>
        {safety.referral_reason}
      </div>
    </div>
  );
}

function CriticalFindings({ findings }) {
  if (!findings || findings.length === 0) return null;

  return (
    <div style={{
      marginTop: 8,
      padding: "8px 12px",
      background: "#fef2f2",
      border: "1px solid #fca5a5",
      borderRadius: 6,
      fontSize: 13,
    }}>
      <strong>Critical Findings Detected:</strong>
      <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
        {findings.map((f, i) => (
          <li key={i} style={{ color: "#991b1b" }}>{f}</li>
        ))}
      </ul>
    </div>
  );
}

function Warnings({ warnings }) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <div style={{ marginTop: 8, fontSize: 12, color: "#6b7280" }}>
      {warnings.map((w, i) => (
        <div key={i} style={{ marginBottom: 2 }}>
          ℹ️ {w}
        </div>
      ))}
    </div>
  );
}

export default function SafetyBanner({ safety }) {
  if (!safety) return null;

  const triage = safety.triage_level || "ROUTINE";
  const style = TRIAGE_STYLES[triage] || TRIAGE_STYLES.ROUTINE;

  return (
    <div
      className={style.pulse ? "safety-pulse" : ""}
      style={{
        padding: "12px 16px",
        marginBottom: 12,
        background: style.bg,
        border: "2px solid " + style.border,
        borderRadius: 8,
        color: style.text,
      }}
    >
      {/* Triage header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontWeight: 700,
        fontSize: 14,
      }}>
        <span style={{ fontSize: 18 }}>{style.icon}</span>
        <span>{style.label}</span>
      </div>

      {/* Confidence bar */}
      <ConfidenceBar
        score={safety.confidence_score || 0}
        label={safety.confidence_label || "MODERATE"}
      />

      {/* Critical findings */}
      <CriticalFindings findings={safety.critical_findings} />

      {/* Referral alert */}
      <ReferralAlert safety={safety} />

      {/* Blocked message */}
      {safety.is_blocked && (
        <div style={{
          marginTop: 8,
          padding: "10px 12px",
          background: "#fef2f2",
          border: "2px solid #ef4444",
          borderRadius: 6,
          fontWeight: 600,
          color: "#991b1b",
        }}>
          🚫 AI interpretation withheld due to low confidence.
          <div style={{ fontWeight: 400, marginTop: 4 }}>
            {safety.block_reason}
          </div>
        </div>
      )}

      {/* Human review badge - always shown */}
      <div style={{
        marginTop: 8,
        display: "flex",
        alignItems: "center",
        gap: 6,
        fontSize: 12,
        color: "#6b7280",
        background: "#f3f4f6",
        padding: "4px 8px",
        borderRadius: 4,
        width: "fit-content",
      }}>
        👨‍⚕️ <strong>Clinician review required</strong> — AI-assisted only
      </div>

      {/* Warnings */}
      <Warnings warnings={safety.warnings} />
    </div>
  );
}

/* Add this CSS to App.css:
@keyframes safety-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
  50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
}
.safety-pulse {
  animation: safety-pulse 2s infinite;
}
*/
