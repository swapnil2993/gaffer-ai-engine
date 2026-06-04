import React, { useState, useEffect } from 'react';
import {
  Search, Shield, Activity, Database, BarChart3, ArrowRight,
  Brain, Layers, Coins, UserCog, AlertCircle, CheckCircle2, Clock, FileText,
} from 'lucide-react';
import axios from 'axios';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

const API = 'http://localhost:8000';

const formatError = (detail) => {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(e => (typeof e === 'object' ? e.msg : e)).join('; ');
  }
  if (typeof detail === 'object' && detail.msg) return detail.msg;
  return 'Unknown error';
};

// ---------- small presentational helpers ----------
const Pill = ({ children, tone = 'gray' }) => {
  const tones = {
    gray: 'bg-white/5 border-white/10 text-gray-300',
    gold: 'bg-scout-gold/15 border-scout-gold/40 text-scout-gold',
    green: 'bg-green-500/10 border-green-500/40 text-green-300',
  };
  return (
    <span className={`px-2.5 py-1 rounded text-xs border ${tones[tone]}`}>{children}</span>
  );
};

const Card = ({ children, className = '' }) => (
  <div className={`bg-pitch-accent rounded-xl border border-white/10 shadow-xl ${className}`}>
    {children}
  </div>
);

// ---------- render text that may contain "- " bullet lines ----------
const BulletText = ({ text, italic = false }) => {
  if (!text) return null;
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);
  const cls = `text-sm text-gray-300 leading-relaxed ${italic ? 'italic' : ''}`;
  const out = [];
  let bullets = [];
  const flush = (key) => {
    if (bullets.length) {
      out.push(
        <ul key={`u${key}`} className="list-disc pl-5 space-y-1 mb-2">
          {bullets.map((b, i) => <li key={i} className={cls}>{b}</li>)}
        </ul>
      );
      bullets = [];
    }
  };
  lines.forEach((line, i) => {
    if (/^[-*•]\s+/.test(line)) {
      bullets.push(line.replace(/^[-*•]\s+/, ''));
    } else {
      flush(i);
      out.push(<p key={`p${i}`} className={`${cls} mb-2`}>{line}</p>);
    }
  });
  flush('end');
  return <div>{out}</div>;
};

// ---------- DeepEval correctness scorecard ----------
const METRIC_LABELS = {
  faithfulness: 'Faithfulness — grounded in the retrieved evidence',
  answer_relevancy: 'Answer relevancy — addresses the query',
  contextual_relevancy: 'Contextual relevancy — retrieved context fits the query',
};
const _scoreColor = (s, passed) =>
  passed ? 'text-green-300' : s != null && s >= 0.5 ? 'text-yellow-300' : 'text-red-300';
const _pct = (s) => (s == null ? '—' : `${Math.round(s * 100)}%`);
const _kindTone = (kind) =>
  kind === 'Retrieval quality' ? 'bg-blue-500/15 text-blue-300 border-blue-500/30'
    : 'bg-scout-gold/15 text-scout-gold border-scout-gold/30';

const EvalScorecard = ({ data }) => (
  <div className="mt-4 bg-black/20 rounded-lg p-4 border border-white/10">
    <div className="flex items-center justify-between mb-3">
      <span className="text-xs uppercase tracking-wider text-gray-400">DeepEval correctness</span>
      <span className="text-[10px] text-gray-500">
        judge: {data.judge} · threshold {Math.round(data.threshold * 100)}%
      </span>
    </div>
    <div className="flex items-baseline gap-3 mb-3">
      <span className={`text-2xl font-bold ${_scoreColor(data.overall, data.overall >= data.threshold)}`}>
        {_pct(data.overall)}
      </span>
      <span className="text-xs text-gray-400">overall (avg of metrics)</span>
    </div>
    <div className="space-y-3">
      {Object.entries(data.metrics).map(([k, m]) => (
        <div key={k} className="text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="text-gray-300 flex items-center gap-2">
              {m.kind && (
                <span className={`px-1.5 py-0.5 rounded text-[9px] border ${_kindTone(m.kind)}`}>{m.kind}</span>
              )}
              {METRIC_LABELS[k] || k}
            </span>
            <span className={`shrink-0 ${_scoreColor(m.score, m.passed)}`}>
              {_pct(m.score)} {m.passed === true ? '✓' : m.passed === false ? '✗' : ''}
            </span>
          </div>
          {m.reason && <p className="text-gray-500 mt-0.5 leading-snug">{m.reason}</p>}
        </div>
      ))}
    </div>
  </div>
);

// ---------- system / indexing status ----------
const SystemBar = ({ sys }) => {
  if (!sys) return null;
  const collections = sys.collections || {};
  const playerStats = collections.player_stats?.rows ?? '—';
  const career = collections.player_career?.rows ?? '—';
  const tacticalRef = sys.tactical_reference || {};

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1.5"><Database className="w-4 h-4" /> {sys.vector_store}</span>
        <span className="flex items-center gap-1.5"><Layers className="w-4 h-4" /> {sys.embedding_model} · {sys.vector_dim}d</span>
        <span className={`flex items-center gap-1.5 ${sys.llm?.ready ? 'text-green-400' : 'text-red-400'}`}>
          <Brain className="w-4 h-4" /> {sys.llm?.ready ? sys.llm.model : 'LLM not configured'}
        </span>
      </div>

      {/* Collections Grid */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        {/* Player Stats Collection */}
        <div className="bg-blue-500/10 border border-blue-500/20 rounded p-2">
          <div className="font-semibold text-blue-300 mb-1">Player Stats</div>
          <div className="text-blue-200 text-sm font-bold mb-1">{playerStats}</div>
          <div className="text-blue-400 text-[10px]">3 seasons</div>
          <div className="text-blue-400 text-[10px]">Season-specific stats</div>
        </div>

        {/* Career Collection */}
        <div className="bg-purple-500/10 border border-purple-500/20 rounded p-2">
          <div className="font-semibold text-purple-300 mb-1">Career Profiles</div>
          <div className="text-purple-200 text-sm font-bold mb-1">{career}</div>
          <div className="text-purple-400 text-[10px]">Multi-season</div>
          <div className="text-purple-400 text-[10px]">Trajectory analysis</div>
        </div>

        {/* Tactical Reference */}
        <div className="bg-green-500/10 border border-green-500/20 rounded p-2">
          <div className="font-semibold text-green-300 mb-1">Tactical Reference</div>
          <div className="text-green-200 text-sm font-bold mb-1">{tacticalRef.count ?? '—'}</div>
          <div className="text-green-400 text-[10px]">Tactical systems</div>
          <div className="text-green-400 text-[10px]">Structured profiles</div>
        </div>
      </div>

      {/* Data Sources */}
      {sys.data_sources && (
        <div className="border-t border-white/10 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Data Sources</div>
          <div className="grid grid-cols-2 gap-1 text-[10px] text-gray-400">
            {Object.entries(sys.data_sources).map(([file, desc]) => (
              <div key={file} className="text-gray-400">
                <span className="text-scout-gold">{file.split('.')[0]}:</span> {desc}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ---------- one pipeline phase: input -> output + why ----------
const PhaseCard = ({ phase, last }) => (
  <div className="relative pl-8">
    {/* timeline rail */}
    <div className="absolute left-2.5 top-1 w-px bg-white/10" style={{ height: last ? '1.25rem' : '100%' }} />
    <div className={`absolute left-0 top-1 w-5 h-5 rounded-full border-2 flex items-center justify-center
      ${phase.skipped ? 'border-gray-600 bg-pitch-accent' : 'border-scout-gold bg-pitch-accent'}`}>
      {phase.skipped
        ? <span className="w-1.5 h-1.5 rounded-full bg-gray-600" />
        : <CheckCircle2 className="w-3 h-3 text-scout-gold" />}
    </div>
    <Card className="p-4 mb-4">
      <div className="flex items-center justify-between mb-2">
        <h4 className="font-semibold text-sm">{phase.name}</h4>
        <span className="flex items-center gap-1 text-[11px] text-gray-500">
          <Clock className="w-3 h-3" /> {phase.ms} ms
        </span>
      </div>
      <div className="grid sm:grid-cols-2 gap-3 mb-3">
        <div className="bg-black/30 rounded p-2.5">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Input</div>
          <div className="text-xs text-gray-300 break-words">{phase.input}</div>
        </div>
        <div className="bg-black/30 rounded p-2.5">
          <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-1">Output</div>
          <div className="text-xs text-gray-300 break-words">{phase.output}</div>
        </div>
      </div>
      <div className="border-l-2 border-scout-gold/50 pl-3">
        <div className="text-[10px] uppercase tracking-wider text-scout-gold mb-1">Why it works this way</div>
        <p className="text-xs text-gray-400 leading-relaxed">{phase.why}</p>
      </div>
    </Card>
  </div>
);

// ---------- candidate player card ----------
const CandidateCard = ({ c }) => {
  // Fallback: extract player name from stats_summary if not directly available
  const playerName = c.player_name || (
    c.stats_summary ? c.stats_summary.split('\n')[0] : 'Unknown'
  );

  return (
  <Card className="p-4">
    <div className="flex justify-between items-start mb-2">
      <div>
        <h4 className="font-bold text-base">{playerName}</h4>
        <div className="text-xs text-gray-400">{c.position} · {c.current_club}</div>
        {c.progression_summary && (
          <div className={`text-xs mt-1 font-semibold ${
            c.progression?.trend === 'improving' ? 'text-green-400' :
            c.progression?.trend === 'declining' ? 'text-red-400' :
            'text-yellow-400'
          }`}>
            {c.progression_summary}
          </div>
        )}
      </div>
      {typeof c.relevance_score === 'number' && (
        <Pill tone="gold">cos {c.relevance_score.toFixed(3)}</Pill>
      )}
    </div>
    <div className="space-y-1.5 text-xs text-gray-300">
      {(c.current_manager || c.manager_playing_style) && (
        <div className="flex items-center gap-2">
          <UserCog className="w-3.5 h-3.5 text-scout-gold shrink-0" />
          <span>{c.current_manager || 'Manager TBD'}{c.manager_playing_style ? ` — ${c.manager_playing_style}` : ''}</span>
        </div>
      )}
      {(c.estimated_cost?.annual_wages || c.estimated_cost?.weekly_wages) && (
        <div className="flex items-center gap-2">
          <Coins className="w-3.5 h-3.5 text-scout-gold shrink-0" />
          <span>{c.estimated_cost?.annual_wages ? `${c.estimated_cost.annual_wages}/yr` : ''}{c.estimated_cost?.annual_wages && c.estimated_cost?.weekly_wages ? ' ' : ''}{c.estimated_cost?.weekly_wages ? `(${c.estimated_cost.weekly_wages}/wk)` : ''}</span>
        </div>
      )}
    </div>
    {(c.manager_tactics?.length > 0 || c.tactical_suitability?.length > 0) && (
      <div className="mt-3 pt-3 border-t border-white/10 space-y-2">
        {c.manager_tactics?.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <span className="text-[9px] text-gray-500 w-full">Manager system:</span>
            {c.manager_tactics.map((t, i) => (
              <Pill key={i} tone="gray">{t.replace(/_/g, ' ')}</Pill>
            ))}
          </div>
        )}
        {c.tactical_suitability?.length > 0 && (
          <div className="flex flex-wrap gap-2">
            <span className="text-[9px] text-gray-500 w-full">Player suited to:</span>
            {c.tactical_suitability.map((t, i) => (
              <Pill key={i} tone="gold">{t.replace(/_/g, ' ')}</Pill>
            ))}
          </div>
        )}
        {c.tactical_alignment && c.progression?.trend !== 'declining' && (
          <div className="text-[9px] text-gray-400 space-y-1">
            {c.tactical_alignment.aligned?.length > 0 && (
              <div className="text-green-400">✓ Aligned: {c.tactical_alignment.aligned.join(', ').replace(/_/g, ' ')}</div>
            )}
            {c.tactical_alignment.conflict?.length > 0 && (
              <div className="text-red-400">✗ Conflict: {c.tactical_alignment.conflict.join(', ').replace(/_/g, ' ')}</div>
            )}
            {c.tactical_alignment.surplus?.length > 0 && !c.tactical_alignment.surplus.includes('versatile') && (
              <div className="text-yellow-400">+ Surplus: {c.tactical_alignment.surplus.join(', ').replace(/_/g, ' ')}</div>
            )}
          </div>
        )}
        {c.progression?.trajectory?.length > 1 && (
          <div className="text-[9px] text-gray-500 mt-2">
            <div className="mb-1">History: {c.progression.trajectory.join(' → ')}</div>
            {c.progression.yoy_changes && Object.keys(c.progression.yoy_changes).slice(0, 3).map(key => {
              const change = c.progression.yoy_changes[key];
              const color = change > 0.1 ? 'text-green-400' : change < -0.1 ? 'text-red-400' : 'text-gray-500';
              return (
                <div key={key} className={color}>
                  {key}: {change > 0 ? '+' : ''}{(change * 100).toFixed(0)}%
                </div>
              );
            })}
          </div>
        )}
      </div>
    )}
    <p className="mt-2 text-[11px] text-gray-500 leading-snug">{c.stats_summary}</p>
  </Card>
  );
};

// ---------- cosine similarity scatter ----------
const TYPE_STYLE = {
  query: { fill: '#d4af37', name: 'Your query' },
  candidate: { fill: '#22c55e', name: 'Recommended' },
  corpus: { fill: '#64748b', name: 'Other players' },
};

const ScatterTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-black/90 border border-white/20 rounded p-2 text-xs">
      <div className="font-semibold text-scout-gold">{p.label}</div>
      {p.type !== 'query' && (
        <>
          <div className="text-gray-300">cosine similarity: {p.similarity?.toFixed(4)}</div>
          <div className="text-gray-500">{p.position} · {p.club} · {p.season}</div>
        </>
      )}
    </div>
  );
};

const SimilarityScatter = ({ plot }) => {
  if (!plot?.points?.length) return null;
  const groups = { query: [], candidate: [], corpus: [] };
  plot.points.forEach((p) => (groups[p.type] || groups.corpus).push(p));
  return (
    <Card className="p-6">
      <h3 className="text-sm font-bold text-scout-gold uppercase mb-1 flex items-center gap-2">
        <BarChart3 className="w-4 h-4" /> Cosine Similarity Space (PCA → 2D)
      </h3>
      <p className="text-xs text-gray-500 mb-4">{plot.explanation}</p>
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
          <CartesianGrid stroke="#ffffff14" />
          <XAxis type="number" dataKey="x" name="PC1" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <YAxis type="number" dataKey="y" name="PC2" tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <ZAxis range={[60, 60]} />
          <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: '3 3' }} />
          <Legend />
          {['corpus', 'candidate', 'query'].map((t) => (
            <Scatter
              key={t}
              name={TYPE_STYLE[t].name}
              data={groups[t]}
              fill={TYPE_STYLE[t].fill}
              shape={t === 'query' ? 'star' : 'circle'}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Card>
  );
};

// ---------- tactical theory card ----------
const TheoryCard = ({ t }) => (
  <div className="bg-black/20 rounded-lg p-4 border border-white/5 mb-3">
    <div className="flex items-center justify-between mb-2">
      <span className="text-[10px] uppercase tracking-wider text-scout-gold font-bold">
        {t.heading || 'General'}
      </span>
      <div className="flex gap-1.5">
        {t.era && <Pill tone="gray">{t.era}</Pill>}
        {t.formation && <Pill tone="gold">{t.formation}</Pill>}
      </div>
    </div>
    <div className="flex gap-2 mb-3">
      {t.style && t.style !== 'Unknown' && (
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-300">
          Style: {t.style}
        </span>
      )}
      {t.line_height && t.line_height !== 'Unknown' && (
        <span className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/10 border border-green-500/20 text-green-300">
          Line: {t.line_height}
        </span>
      )}
    </div>
    <p className="text-xs text-gray-400 leading-relaxed italic border-l-2 border-white/10 pl-3">
      "{t.text.split('--- CORE TEXT ---').pop().trim().substring(0, 350)}..."
    </p>
    {t.weights && Object.keys(t.weights).some(k => t.weights[k] > 0) && (
      <div className="mt-3 pt-3 border-t border-white/5 flex flex-wrap gap-2">
        {Object.entries(t.weights)
          .filter(([_, v]) => v > 0)
          .map(([k, v]) => (
            <span key={k} className="text-[9px] text-gray-500">
              {k}: <span className="text-scout-gold">{(v * 100).toFixed(0)}%</span>
            </span>
          ))}
      </div>
    )}
  </div>
);

// ---------- main app ----------
const App = () => {
  const [query, setQuery] = useState('Creative midfielder who excels in defensive transitions for a high press');
  const [season, setSeason] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const [sys, setSys] = useState(null);
  const [evalState, setEvalState] = useState({ loading: false, data: null, error: null });

  useEffect(() => {
    axios.get(`${API}/system`).then((r) => setSys(r.data)).catch(() => {});
  }, []);

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setEvalState({ loading: false, data: null, error: null });
    try {
      const r = await axios.post(`${API}/query`, null, {
        params: {
          query_str: query,
          season: season || undefined,
        },
      });
      setReport(r.data);
      axios.get(`${API}/system`).then((res) => setSys(res.data)).catch(() => {});
      // Auto-run the DeepEval correctness scoring (only if a brief was produced).
      if (r.data?.candidates?.retrieved_and_enriched?.length > 0) handleEvaluate(r.data);
    } catch (err) {
      const detail = err.response?.data?.detail || 'Failed to fetch. Is the backend running on :8000?';
      setError(formatError(detail));
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluate = async (rep = report) => {
    if (!rep) return;
    setEvalState({ loading: true, data: null, error: null });
    try {
      const context = [
        ...(rep.context_used?.tactical_theory?.map(t => t.text) || []),
        ...(rep.context_used?.player_records?.map(p => typeof p === 'string' ? p : p.stats_summary) || []),
      ];
      const r = await axios.post(`${API}/evaluate`, {
        query: rep.query?.tactical_query || rep.tactical_query,
        scouting_brief: rep.brief?.narrative || rep.scouting_brief,
        context,
      });
      setEvalState({ loading: false, data: r.data, error: null });
    } catch (err) {
      const detail = err.response?.data?.detail || 'Evaluation failed.';
      setEvalState({
        loading: false, data: null,
        error: formatError(detail),
      });
    }
  };

  return (
    <div className="min-h-screen bg-football-green text-gray-100 font-sans p-6">
      <header className="mb-8 border-b border-pitch-accent pb-4">
        <div className="flex items-center gap-3 mb-3">
          <Shield className="text-scout-gold w-8 h-8" />
          <h1 className="text-2xl font-bold tracking-tight">GAFFER<span className="text-scout-gold"> AI ENGINE</span></h1>
          <span className="text-xs text-gray-500 ml-2">An explainable RAG scouting pipeline</span>
        </div>
        <SystemBar sys={sys} />
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Controls */}
        <div className="lg:col-span-4 space-y-6">
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-scout-gold" /> Query Understanding
            </h2>
            <textarea
              className="w-full bg-black/30 border border-white/10 rounded-lg p-3 text-sm focus:ring-2 focus:ring-scout-gold outline-none h-48 mb-4 resize-vertical"
              placeholder="Describe the player profile / tactical need (position, specific player, and club are inferred from query)…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="mb-4">
              <label className="text-xs text-gray-400 block mb-2">Season (optional override)</label>
              <select className="w-full bg-black/30 border border-white/10 rounded p-2 text-sm"
                value={season} onChange={(e) => setSeason(e.target.value)}>
                <option value="">Latest available</option>
                <option value="2023/2024">2023/2024</option>
                <option value="2024/2025">2024/2025</option>
                <option value="2025/2026">2025/2026</option>
              </select>
              <p className="text-[10px] text-gray-500 mt-1.5">Position, player name, and club are automatically inferred from the query during Phase 2.</p>
            </div>
            <button onClick={handleSearch} disabled={loading}
              className="w-full bg-scout-gold hover:bg-yellow-600 text-black font-bold py-3 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-60">
              {loading ? 'Running pipeline…' : <>Run Pipeline <ArrowRight className="w-4 h-4" /></>}
            </button>
          </Card>

          {report?.analysis?.retrieval_strategy && (
            <Card className="p-4 text-xs text-gray-400">
              <div className="uppercase tracking-wider text-gray-500 mb-3">Search Strategy</div>
              <div className="space-y-2">
                {report.analysis.retrieval_strategy.search_route && (
                  <div>
                    <span className="text-gray-500">route:</span>
                    <Pill tone={report.analysis.retrieval_strategy.is_career_query ? 'gold' : 'gray'} className="ml-2">
                      {report.analysis.retrieval_strategy.search_route}
                    </Pill>
                  </div>
                )}
                <div>filter used: <span className="text-gray-200">{report.analysis.retrieval_strategy.filter_used}</span></div>
                <div>candidates: <span className="text-gray-200">{report.analysis.retrieval_strategy.result_count} / {report.analysis.retrieval_strategy.pool_size} pool</span></div>
                {report.analysis.retrieval_strategy.ranked_by?.length > 0 && (
                  <div>ranked by: <span className="text-gray-200">{report.analysis.retrieval_strategy.ranked_by.join(', ')}</span></div>
                )}
                {report.analysis.retrieval_strategy.excluded_club && (
                  <div>excluding own club: <span className="text-scout-gold">{report.analysis.retrieval_strategy.excluded_club}</span></div>
                )}
              </div>
            </Card>
          )}

          {sys?.pipeline && !report && (
            <Card className="p-4">
              <div className="text-xs uppercase tracking-wider text-gray-500 mb-3">Pipeline stages</div>
              <ol className="space-y-2 text-xs text-gray-300">
                {sys.pipeline.map((p, i) => (
                  <li key={i} className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-[10px]">{i + 1}</span>
                    {p}
                  </li>
                ))}
              </ol>
            </Card>
          )}
        </div>

        {/* Results */}
        <div className="lg:col-span-8 space-y-6">
          {error && (
            <div className="bg-red-900/30 border border-red-500/50 p-4 rounded-lg flex items-center gap-3 text-red-200">
              <AlertCircle className="w-5 h-5" /> <p className="text-sm">{error}</p>
            </div>
          )}

          {!report && !error && (
            <div className="h-64 flex flex-col items-center justify-center text-gray-500 bg-pitch-accent/50 rounded-xl border border-dashed border-white/10">
              <BarChart3 className="w-12 h-12 mb-3 opacity-20" />
              <p className="text-sm">Run a query to watch it flow through every phase of the pipeline.</p>
            </div>
          )}

          {report && (
            <>
              {/* Phase trace */}
              <Card className="p-6">
                <h3 className="text-sm font-bold text-scout-gold uppercase mb-5 flex items-center gap-2">
                  <Activity className="w-4 h-4" /> How your query flowed through the system
                </h3>
                <div>
                  {report.phases?.map((p, i) => (
                    <PhaseCard key={i} phase={p} last={i === report.phases.length - 1} />
                  ))}
                </div>
              </Card>

              {/* Similarity scatter */}
              <SimilarityScatter plot={report.analysis?.similarity_plot} />

              {/* Candidates */}
              {report.candidates?.retrieved_and_enriched?.length > 0 && (
                <div>
                  <h3 className="text-sm font-bold text-scout-gold uppercase mb-3 flex items-center gap-2">
                    <Layers className="w-4 h-4" /> Retrieved & Enriched Candidates
                  </h3>
                  <div className="grid sm:grid-cols-2 gap-4">
                    {report.candidates.retrieved_and_enriched.map((c, i) => <CandidateCard key={i} c={c} />)}
                  </div>
                </div>
              )}

              {/* Tactical theory used */}
              {report.context_used?.tactical_theory?.length > 0 && (
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-scout-gold uppercase">Tactical Theory Retrieved (Index A)</h3>
                    <a href={`${API}/theory/document`} target="_blank" rel="noreferrer"
                       className="text-xs text-scout-gold hover:underline flex items-center gap-1">
                      <FileText className="w-3.5 h-3.5" /> View full Docling output ↗
                    </a>
                  </div>
                  <p className="text-[11px] text-gray-500 mb-3">
                    These chunks were retrieved from the Docling-parsed book. The link opens the
                    full structured document (headings &amp; tables) Docling extracted from the EPUB.
                  </p>
                  <div className="space-y-0">
                    {report.context_used.tactical_theory.map((t, i) => (
                      <TheoryCard key={i} t={t} />
                    ))}
                  </div>
                </Card>
              )}

              {/* Reasoning */}
              {report.brief?.reasoning && (
                <Card className="p-6 border-l-4 border-scout-gold">
                  <h3 className="text-sm font-bold text-scout-gold uppercase mb-3 flex items-center gap-2">
                    <Brain className="w-4 h-4" /> Chain-of-Thought Reasoning
                  </h3>
                  <BulletText text={report.brief.reasoning} italic />
                </Card>
              )}

              {/* Final brief */}
              <Card className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    <Shield className="w-5 h-5 text-scout-gold" /> Final Scouting Brief
                  </h3>
                  {report.candidates?.retrieved_and_enriched?.length > 0 && (
                    <button onClick={() => handleEvaluate()} disabled={evalState.loading}
                      className="text-xs border border-scout-gold/50 text-scout-gold rounded px-3 py-1.5 hover:bg-scout-gold/10 disabled:opacity-50 flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      {evalState.loading ? 'Scoring with DeepEval…' : 'Re-run correctness check'}
                    </button>
                  )}
                </div>
                <BulletText text={report.brief?.narrative} />

                {/* DeepEval correctness scorecard (auto-runs after each query) */}
                {evalState.loading && !evalState.data && (
                  <p className="mt-4 text-xs text-gray-400 italic">
                    Scoring correctness with DeepEval (faithfulness, relevancy, retrieval)…
                  </p>
                )}
                {evalState.error && (
                  <p className="mt-4 text-xs text-red-300">{evalState.error}</p>
                )}
                {evalState.data && <EvalScorecard data={evalState.data} />}

                {report.notes && (
                  <div className="mt-5 pt-4 border-t border-white/10 text-[11px] text-gray-500 space-y-1">
                    <p>· {report.notes.manager}</p>
                    <p>· {report.notes.cost}</p>
                  </div>
                )}
              </Card>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;
