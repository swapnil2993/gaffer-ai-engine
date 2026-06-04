import React from "react";
import "./ImpactAnalysis.css";

/**
 * ImpactAnalysis Component
 *
 * Displays why a player was recommended and how much career data influenced it.
 * Shows confidence level, impact drivers, and career trajectory.
 */

function ImpactBadges({ drivers }) {
  const getIcon = (source) => {
    if (source.includes("this season")) return "⚽";
    if (source.includes("career")) return "📈";
    if (source.includes("manager") || source.includes("team")) return "👤";
    return "📊";
  };

  return (
    <div className="impact-badges">
      <h4 className="impact-title">🎯 Key Drivers</h4>
      <div className="drivers-list">
        {drivers.map((driver, idx) => (
          <div key={idx} className="driver-tag">
            <span className="driver-icon">{getIcon(driver.source)}</span>
            <span className="driver-metric">{driver.metric}</span>
            <span className="driver-value">{driver.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CareerTrajectoryCard({ careerContext }) {
  if (!careerContext) return null;

  const trendEmoji = {
    improving: "📈",
    declining: "📉",
    stable: "➡️",
  }[careerContext.trend] || "❓";

  const trendColor = {
    improving: "#4CAF50",
    declining: "#FF5252",
    stable: "#2196F3",
  }[careerContext.trend] || "#999";

  const momentumStr =
    careerContext.momentum > 0
      ? `+${(careerContext.momentum * 100).toFixed(0)}%`
      : `${(careerContext.momentum * 100).toFixed(0)}%`;

  return (
    <div className="career-trajectory-card">
      <h4>Career Trajectory</h4>
      <div className="trajectory-content">
        <div className="trend-indicator" style={{ borderColor: trendColor }}>
          <span className="trend-emoji">{trendEmoji}</span>
          <span className="trend-text" style={{ color: trendColor }}>
            {careerContext.trend.toUpperCase()}
          </span>
          <span className="momentum" style={{ color: trendColor }}>
            {momentumStr} YoY
          </span>
        </div>

        <div className="trajectory-meta">
          <div className="meta-item">
            <span className="meta-label">Consistency:</span>
            <span className="meta-value">{careerContext.consistency}</span>
          </div>
          {careerContext.best_season && (
            <div className="meta-item">
              <span className="meta-label">Peak Season:</span>
              <span className="meta-value">{careerContext.best_season}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ConfidenceIndicator({ level, riskFactors }) {
  const levels = {
    high: { color: "#4CAF50", icon: "✓", label: "HIGH" },
    medium: { color: "#FFC107", icon: "➡️", label: "MEDIUM" },
    low: { color: "#FF5252", icon: "⚠️", label: "LOW" },
  };

  const config = levels[level] || levels.medium;

  return (
    <div className="confidence-indicator">
      <div className="confidence-header" style={{ borderColor: config.color }}>
        <span className="confidence-icon">{config.icon}</span>
        <span className="confidence-label">
          Confidence: <strong style={{ color: config.color }}>{config.label}</strong>
        </span>
      </div>

      {riskFactors && riskFactors.length > 0 && (
        <div className="risk-factors">
          {riskFactors.map((factor, idx) => (
            <div key={idx} className="risk-item">
              <span className="risk-icon">⚠️</span>
              <span className="risk-text">{factor}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SeasonComparison({ stats, aggregates }) {
  if (!aggregates || Object.keys(aggregates).length === 0) return null;

  const metricsToShow = ["goals", "assists", "tackles", "interceptions"];

  return (
    <div className="season-comparison">
      <h4>This Season vs Career Average</h4>
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>This Season</th>
            <th>Career Avg</th>
            <th>Diff</th>
          </tr>
        </thead>
        <tbody>
          {metricsToShow.map((metric) => {
            const current = stats[metric] || 0;
            const careerData = aggregates[metric];
            if (!careerData) return null;

            const avg = careerData.avg || 0;
            const diff = avg > 0 ? ((current - avg) / avg) * 100 : 0;
            const diffStr = diff > 0 ? `+${diff.toFixed(0)}%` : `${diff.toFixed(0)}%`;
            const arrow = diff > 5 ? "🔼" : diff < -5 ? "🔽" : "→";

            return (
              <tr key={metric}>
                <td className="metric-name">
                  {metric.charAt(0).toUpperCase() + metric.slice(1)}
                </td>
                <td className="metric-value">{Math.round(current)}</td>
                <td className="metric-value">{avg.toFixed(1)}</td>
                <td className={`metric-diff ${diff > 0 ? "positive" : diff < 0 ? "negative" : "neutral"}`}>
                  {diffStr} {arrow}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ImpactAnalysisPanel({ impactAnalysis, candidate, aggregates }) {
  if (!impactAnalysis || !impactAnalysis.primary_driver) return null;

  const primary = impactAnalysis.primary_driver;

  return (
    <div className="impact-analysis-panel">
      <div className="impact-header">
        <h3>📊 Impact Analysis: Why {candidate.player_name}?</h3>
        <span className={`collection-badge ${impactAnalysis.collection_used}`}>
          {impactAnalysis.collection_used === "player_career_collection"
            ? "3-Year Career Profile"
            : "Current Season"}
        </span>
      </div>

      <div className="impact-grid">
        {/* Left column: Drivers and Confidence */}
        <div className="impact-left">
          {primary.impact_drivers && primary.impact_drivers.length > 0 && (
            <ImpactBadges drivers={primary.impact_drivers} />
          )}

          <ConfidenceIndicator level={primary.confidence_level} riskFactors={primary.risk_factors} />
        </div>

        {/* Right column: Career Context and Comparison */}
        <div className="impact-right">
          {primary.career_context && (
            <CareerTrajectoryCard careerContext={primary.career_context} />
          )}

          {aggregates && (
            <SeasonComparison stats={candidate.stats} aggregates={aggregates} />
          )}
        </div>
      </div>

      {/* Runner-up comparison (optional) */}
      {impactAnalysis.runner_up_driver && (
        <div className="runner-up-section">
          <h4>🏃 Runner-Up Consideration</h4>
          <p>
            Also considered: <strong>{impactAnalysis.runner_up_driver.player_name}</strong>
          </p>
          {impactAnalysis.runner_up_driver.impact_drivers && (
            <ImpactBadges drivers={impactAnalysis.runner_up_driver.impact_drivers.slice(0, 2)} />
          )}
        </div>
      )}
    </div>
  );
}

export default ImpactAnalysisPanel;
