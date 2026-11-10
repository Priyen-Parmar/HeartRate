import { useState, useEffect, useRef, useCallback } from 'react';
import {
    LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
    Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts';
import { api } from '../api';

const GROUPS = [
    { id: 'veteran', name: 'Veterans', color: '#10b981', trueRate: 0.90, quota: 0.00, label: 'TOP VETERANS' },
    { id: 'intermediate', name: 'Intermediates', color: '#f59e0b', trueRate: 0.70, quota: 0.15, label: 'MID-TIER' },
    { id: 'newbie', name: 'Newbies', color: '#06b6d4', trueRate: 0.45, quota: 0.25, label: 'COLD START' },
];

const DarkTooltip = ({ active, payload, label, accentColor }) => {
    if (!active || !payload?.length) return null;
    return (
        <div style={{ background: '#0a1628', border: `1px solid ${accentColor || '#10b981'}44`, borderRadius: 6, padding: '8px 12px', fontSize: 11, fontFamily: 'monospace' }}>
            <div style={{ color: accentColor || '#10b981', marginBottom: 4 }}>t = {label}</div>
            {payload.map((p, i) => (
                <div key={i} style={{ color: p.color || '#e2e8f0' }}>{p.name}: <b>{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}%</b></div>
            ))}
        </div>
    );
};

export default function AdminSimulation() {
    const [algoState, setAlgoState] = useState(null);
    const [groups, setGroups] = useState([]);
    const [history, setHistory] = useState([]);
    const [log, setLog] = useState([]);
    const [mode, setMode] = useState('fair');
    const [running, setRunning] = useState(false);
    const [driverStats, setDriverStats] = useState(null);
    const [dashData, setDashData] = useState(null);
    const intervalRef = useRef(null);

    const loadState = useCallback(async () => {
        try {
            const [stateData, histData] = await Promise.all([
                api.simState(),
                api.simHistory(300),
            ]);
            setAlgoState(stateData.state);
            setGroups(stateData.groups);
            setMode(stateData.state.mode);
            setHistory(histData.history.map(h => ({
                t: h.t, sat: h.sat,
                shareA: h.share_veteran, shareB: h.share_intermediate, shareC: h.share_newbie,
            })));
        } catch { }
    }, []);

    const loadDashboard = useCallback(async () => {
        try {
            const [stats, dash] = await Promise.all([
                api.driverStats(),
                api.dashboard(),
            ]);
            setDriverStats(stats);
            setDashData(dash);
        } catch { }
    }, []);

    useEffect(() => { loadState(); loadDashboard(); }, [loadState, loadDashboard]);

    const runBatch = useCallback(async () => {
        try {
            const data = await api.simStep({ count: 10, mode });
            const newHistory = data.steps.map(s => ({
                t: s.t, sat: s.sat,
                shareA: s.share_veteran, shareB: s.share_intermediate, shareC: s.share_newbie,
            }));
            setHistory(h => {
                const combined = [...h, ...newHistory];
                return combined.length > 300 ? combined.slice(-300) : combined;
            });
            setLog(l => [...data.steps.map(s => ({
                t: s.t, group: s.chosenGroup, reward: s.reward, mode: s.mode, reason: s.reason,
            })).reverse(), ...l].slice(0, 14));

            // Refresh state
            const stateData = await api.simState();
            setAlgoState(stateData.state);
            setGroups(stateData.groups);
        } catch { }
    }, [mode]);

    useEffect(() => {
        if (running) {
            intervalRef.current = setInterval(runBatch, 300);
        } else if (intervalRef.current) {
            clearInterval(intervalRef.current);
        }
        return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
    }, [running, runBatch]);

    const resetSim = async () => {
        setRunning(false);
        try {
            await api.simReset();
            setHistory([]);
            setLog([]);
            loadState();
        } catch { }
    };

    const toggleMode = async () => {
        const newMode = mode === 'fair' ? 'greedy' : 'fair';
        setMode(newMode);
        try { await api.simSetMode(newMode); } catch { }
    };

    const total = algoState?.total || 0;
    const overallSat = algoState?.overallSat || 0;

    const barData = GROUPS.map((g, i) => ({
        name: g.name,
        share: groups[i]?.share || 0,
        color: g.color,
    }));

    return (
        <div className="page" style={{ fontFamily: "'Inter',sans-serif" }}>
            {/* ── TOP BAR ── */}
            <div className="animate-in" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20, flexWrap: 'wrap', gap: 10 }}>
                <div>
                    <h2 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span>Fair UCB Simulation</span>
                        {running && (
                            <span className="badge badge-emerald pulse" style={{ fontSize: '0.6rem' }}>● LIVE</span>
                        )}
                    </h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginTop: 4 }}>
                        FAIR-LEARN meta-algorithm · Patil et al. 2020
                    </p>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {/* Mode toggle */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)' }}>
                        <span style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.08em', color: mode === 'greedy' ? 'var(--amber)' : 'var(--text-dim)' }}>GREEDY</span>
                        <div className="toggle" onClick={toggleMode}
                            style={{ background: mode === 'fair' ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)', border: `1px solid ${mode === 'fair' ? 'rgba(16,185,129,0.4)' : 'rgba(245,158,11,0.4)'}` }}>
                            <div className="toggle-knob" style={{
                                left: mode === 'fair' ? 24 : 3,
                                background: mode === 'fair' ? '#10b981' : '#f59e0b',
                                boxShadow: `0 0 8px ${mode === 'fair' ? '#10b981' : '#f59e0b'}`,
                            }} />
                        </div>
                        <span style={{ fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.08em', color: mode === 'fair' ? 'var(--emerald)' : 'var(--text-dim)' }}>FAIR-MAB</span>
                    </div>

                    <button onClick={() => setRunning(r => !r)} className={`btn btn-sm ${running ? 'btn-danger' : 'btn-primary'}`}>
                        {running ? '⏸ Pause' : '▶ Start'}
                    </button>
                    <button onClick={resetSim} className="btn btn-sm btn-secondary">↺ Reset</button>
                </div>
            </div>

            {/* ── KPI CARDS ── */}
            <div className="grid-4 stagger" style={{ marginBottom: 16 }}>
                {[
                    { label: 'TOTAL GIGS DISPATCHED', value: total.toLocaleString(), sub: `time step t = ${total}`, color: 'var(--text-primary)' },
                    { label: 'PLATFORM SATISFACTION', value: total > 0 ? `${overallSat}%` : '—', sub: 'Fair ≈80% · Greedy ≈90%', color: overallSat > 82 ? 'var(--emerald)' : 'var(--amber)' },
                    { label: 'ALGORITHM MODE', value: mode === 'fair' ? 'FAIR-UCB' : 'GREEDY', sub: mode === 'fair' ? 'Quotas enforced · α=0' : 'No fairness constraints', color: mode === 'fair' ? 'var(--emerald)' : 'var(--amber)' },
                    { label: 'REAL DRIVERS', value: driverStats?.total || 0, sub: `${driverStats?.online || 0} online now`, color: 'var(--cyan)' },
                ].map((k, i) => (
                    <div key={i} className="card stat-card">
                        <div className="label stat-label">{k.label}</div>
                        <div className="stat-value" style={{ color: k.color }}>{k.value}</div>
                        <div className="stat-sub">{k.sub}</div>
                    </div>
                ))}
            </div>

            {/* ── WORKER GROUP CARDS ── */}
            <div className="grid-3 stagger" style={{ marginBottom: 16 }}>
                {groups.map((g, i) => (
                    <div key={i} className="card" style={{ padding: '16px 18px', position: 'relative', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg,${GROUPS[i].color},transparent)` }} />

                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                            <div>
                                <span style={{ fontFamily: "'Outfit',sans-serif", fontSize: '1rem', fontWeight: 700, color: GROUPS[i].color }}>
                                    {GROUPS[i].name}
                                </span>
                                <div className="label" style={{ marginTop: 2 }}>{GROUPS[i].label}</div>
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div className="label">Quota</div>
                                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '0.95rem', color: GROUPS[i].color, fontWeight: 700 }}>
                                    {(GROUPS[i].quota * 100).toFixed(0)}%
                                </div>
                            </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6, marginBottom: 10 }}>
                            {[
                                { l: 'Gigs N_i', v: g.N?.toLocaleString() || '0' },
                                { l: 'Share', v: `${g.share}%` },
                                { l: 'Sat. Rate', v: `${g.satRate}%` },
                            ].map((s, j) => (
                                <div key={j} style={{ padding: '6px 8px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <div className="label" style={{ fontSize: '0.55rem' }}>{s.l}</div>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 600, marginTop: 2 }}>{s.v}</div>
                                </div>
                            ))}
                        </div>

                        {mode === 'fair' && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                                <span className="label" style={{ fontSize: '0.6rem' }}>V_i =</span>
                                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: g.violation > 0 ? 'var(--red)' : 'var(--emerald)' }}>
                                    {g.violation > 0 ? `▲ ${g.violation?.toFixed(1)} STARVED` : '✓ FAIR'}
                                </span>
                            </div>
                        )}

                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>μ={GROUPS[i].trueRate * 100}%</span>
                            <div style={{ flex: 1, height: 3, background: 'rgba(255,255,255,0.04)', borderRadius: 2 }}>
                                <div style={{ height: '100%', width: `${GROUPS[i].trueRate * 100}%`, background: GROUPS[i].color, borderRadius: 2, opacity: 0.7 }} />
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* ── CHARTS ROW ── */}
            <div className="grid-2" style={{ marginBottom: 16 }}>
                <div className="card" style={{ padding: 20 }}>
                    <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--emerald)', letterSpacing: '0.15em', fontWeight: 700 }}>EFFICIENCY TRADE-OFF</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 2 }}>Cumulative platform satisfaction % over time</div>
                    </div>
                    {history.length < 2 ? (
                        <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-dim)', fontSize: '0.82rem' }}>
                            {running ? 'Acquiring signal…' : 'Start simulation to render'}
                        </div>
                    ) : (
                        <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={history} margin={{ top: 4, right: 8, bottom: 4, left: -22 }}>
                                <CartesianGrid strokeDasharray="2 5" stroke="rgba(255,255,255,0.03)" />
                                <XAxis dataKey="t" tick={{ fill: '#1e3a5f', fontSize: 9 }} tickLine={false} axisLine={false} />
                                <YAxis domain={[40, 100]} tick={{ fill: '#1e3a5f', fontSize: 9 }} tickLine={false} axisLine={false} />
                                <Tooltip content={<DarkTooltip accentColor="#10b981" />} />
                                <ReferenceLine y={90} stroke="rgba(16,185,129,0.18)" strokeDasharray="3 3" />
                                <ReferenceLine y={70} stroke="rgba(245,158,11,0.18)" strokeDasharray="3 3" />
                                <Line type="monotoneX" dataKey="sat" stroke="#10b981" strokeWidth={2} dot={false} name="Satisfaction" />
                            </LineChart>
                        </ResponsiveContainer>
                    )}
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 8 }}>
                        ↑ Switch to GREEDY to see satisfaction rise — that delta is the <span style={{ color: 'var(--amber)' }}>cost of fairness</span>
                    </div>
                </div>

                <div className="card" style={{ padding: 20 }}>
                    <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--cyan)', letterSpacing: '0.15em', fontWeight: 700 }}>MARKET SHARE & EXPOSURE</div>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 2 }}>Gig allocation % per group vs. fairness quotas</div>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 4, left: -22 }}>
                            <CartesianGrid strokeDasharray="2 5" stroke="rgba(255,255,255,0.03)" />
                            <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} tickLine={false} axisLine={false} />
                            <YAxis domain={[0, 100]} tick={{ fill: '#1e3a5f', fontSize: 9 }} tickLine={false} axisLine={false} />
                            <Tooltip content={<DarkTooltip accentColor="#06b6d4" />} />
                            <ReferenceLine y={15} stroke="rgba(245,158,11,0.5)" strokeDasharray="4 2" />
                            <ReferenceLine y={25} stroke="rgba(6,182,212,0.5)" strokeDasharray="4 2" />
                            <Bar dataKey="share" name="Share %" radius={[4, 4, 0, 0]}>
                                {barData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.75} />)}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 8 }}>
                        Dashed lines = quota floors. In FAIR-MAB, Intermediates & Newbies stay <span style={{ color: 'var(--cyan)' }}>above their lines</span>.
                    </div>
                </div>
            </div>

            {/* ── BOTTOM ROW ── */}
            <div className="grid-2">
                {/* Algorithm State */}
                <div className="card" style={{ padding: 20 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--amber)', letterSpacing: '0.15em', fontWeight: 700, marginBottom: 14 }}>
                        ALGORITHM STATE · FAIR-LEARN (α = 0)
                    </div>
                    <div style={{ display: 'grid', gap: 8, marginBottom: 14 }}>
                        {groups.map((g, i) => (
                            <div key={i} style={{
                                display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px',
                                background: g.isStarved && mode === 'fair' ? `${GROUPS[i].color}12` : 'var(--bg-elevated)',
                                borderRadius: 'var(--radius-sm)',
                                border: `1px solid ${g.isStarved && mode === 'fair' ? GROUPS[i].color + '44' : 'transparent'}`,
                            }}>
                                <div style={{ width: 3, height: 38, background: GROUPS[i].color, borderRadius: 2, opacity: g.isStarved ? 1 : 0.3 }} />
                                <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                                        <span style={{ fontSize: '0.78rem', color: GROUPS[i].color, fontWeight: 700 }}>{GROUPS[i].name}</span>
                                        {g.isStarved && mode === 'fair' && <span style={{ fontSize: '0.65rem', color: 'var(--red)' }}>⚠ FAIRNESS OVERRIDE</span>}
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 6 }}>
                                        {[
                                            { l: 'N_i', v: g.N },
                                            { l: 'V_i', v: g.violation?.toFixed(1), c: g.violation > 0 ? 'var(--red)' : 'var(--emerald)' },
                                            { l: 'UCB', v: g.ucbScore },
                                            { l: 'μ̂_i', v: g.N > 0 ? (g.S / g.N).toFixed(3) : '—' },
                                        ].map((s, j) => (
                                            <div key={j} style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                                                {s.l}: <span style={{ color: s.c || 'var(--text-secondary)' }}>{s.v}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Formula box */}
                    <div style={{ padding: '10px 12px', background: 'rgba(255,255,255,0.015)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '0.6rem', color: '#f59e0b44', letterSpacing: '0.1em', marginBottom: 4 }}>ALGORITHM 1 · FAIR-LEARN</div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', lineHeight: 1.8 }}>
                            <div>V<sub>i</sub> = r<sub>i</sub>·(t−1) − N<sub>i,t−1</sub></div>
                            <div>A(t) = {'{'} i : V<sub>i</sub> {'>'} α {'}'} (starved set, α=0)</div>
                            <div style={{ color: '#ef444488' }}>If A(t)≠∅ → pull argmax<sub>i∈A</sub>(V<sub>i</sub>) [fairness]</div>
                            <div style={{ color: '#10b98188' }}>If A(t)=∅ → pull argmax UCB1(i) [learning]</div>
                        </div>
                    </div>
                </div>

                {/* Dispatch Log */}
                <div className="card" style={{ padding: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', letterSpacing: '0.15em', fontWeight: 700 }}>DISPATCH LOG</div>
                        {running && <div className="pulse" style={{ fontSize: '0.7rem', color: 'var(--emerald)' }}>● STREAMING</div>}
                    </div>

                    <div style={{ display: 'grid', gap: 4, minHeight: 180, overflow: 'hidden', marginBottom: 14 }}>
                        {log.length === 0 ? (
                            <div style={{ color: 'var(--text-dim)', fontSize: '0.82rem', padding: 24, textAlign: 'center' }}>
                                Awaiting dispatch events…
                            </div>
                        ) : log.map((e, i) => {
                            const g = GROUPS.find(x => x.id === e.group) || GROUPS[0];
                            return (
                                <div key={i} className="animate-slide" style={{
                                    display: 'flex', alignItems: 'center', gap: 10, padding: '6px 10px',
                                    background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)',
                                    opacity: Math.max(0.2, 1 - i * 0.06),
                                }}>
                                    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', minWidth: 40, fontFamily: "'JetBrains Mono',monospace" }}>#{e.t}</span>
                                    <div style={{ width: 6, height: 6, borderRadius: 2, background: g.color }} />
                                    <span style={{ fontSize: '0.78rem', color: g.color, fontWeight: 600, minWidth: 90 }}>{g.name}</span>
                                    <span style={{ fontSize: '0.78rem', color: e.reward ? 'var(--emerald)' : 'var(--red)', minWidth: 75 }}>
                                        {e.reward ? '✓ SATISFIED' : '✗ FAILED'}
                                    </span>
                                    <span style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>
                                        {e.mode === 'fair' ? 'FAIR-UCB' : 'GREEDY'}
                                    </span>
                                </div>
                            );
                        })}
                    </div>

                    {/* Cost of fairness */}
                    <div style={{ padding: '12px 14px', background: 'var(--amber-bg)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--amber-border)' }}>
                        <div style={{ fontSize: '0.7rem', color: 'var(--amber)', fontWeight: 700, marginBottom: 4, letterSpacing: '0.08em' }}>COST OF FAIRNESS</div>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', lineHeight: 1.7 }}>
                            Routing gigs to Intermediates & Newbies reduces overall satisfaction vs. pure Greedy (which always picks Veterans).
                            That satisfaction delta is the empirical <em style={{ color: '#f59e0b88' }}>r-Regret</em> —
                            the price of inclusion. Theorem 4 proves it grows as O(T) when quotas are large, but stays O(ln T) when quotas are small.
                        </div>
                    </div>
                </div>
            </div>

            {/* ── RECENT REAL RIDES ── */}
            {dashData?.recentRides?.length > 0 && (
                <div className="card" style={{ padding: 20, marginTop: 16 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--purple)', letterSpacing: '0.15em', fontWeight: 700, marginBottom: 14 }}>
                        REAL RIDE DATA · DATABASE
                    </div>
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                    {['#', 'Rider', 'Driver', 'Group', 'Fare', 'Distance', 'Status', 'Algorithm'].map(h => (
                                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.7rem', letterSpacing: '0.05em' }}>{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {dashData.recentRides.map(r => (
                                    <tr key={r.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                        <td style={{ padding: '8px 10px', fontFamily: "'JetBrains Mono',monospace", color: 'var(--text-dim)' }}>{r.id}</td>
                                        <td style={{ padding: '8px 10px' }}>{r.rider_name}</td>
                                        <td style={{ padding: '8px 10px' }}>{r.driver_name || '—'}</td>
                                        <td style={{ padding: '8px 10px' }}>
                                            <span className={`badge badge-${r.driver_group === 'veteran' ? 'emerald' : r.driver_group === 'intermediate' ? 'amber' : 'cyan'}`}
                                                style={{ fontSize: '0.55rem' }}>
                                                {r.driver_group}
                                            </span>
                                        </td>
                                        <td style={{ padding: '8px 10px', color: 'var(--emerald)', fontWeight: 600 }}>₹{r.fare}</td>
                                        <td style={{ padding: '8px 10px' }}>{r.distance_km} km</td>
                                        <td style={{ padding: '8px 10px' }}>
                                            <span className={`badge ${r.status === 'completed' ? 'badge-emerald' : 'badge-amber'}`} style={{ fontSize: '0.55rem' }}>
                                                {r.status}
                                            </span>
                                        </td>
                                        <td style={{ padding: '8px 10px', color: r.algorithm_mode === 'fair' ? 'var(--emerald)' : 'var(--amber)', fontSize: '0.72rem' }}>
                                            {r.algorithm_mode === 'fair' ? 'Fair UCB' : 'Greedy'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Footer */}
            <div style={{ marginTop: 16, display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-dim)', letterSpacing: '0.08em' }}>
                <span>FAIR-LEARN · Patil, Ghalme, Nair, Narahari · IISc · arXiv:1907.10516v2 (2020)</span>
                <span>Σ r_i = 0.40 {'<'} 1 | α = 0 (strict fairness) | 10 gigs/tick</span>
            </div>
        </div>
    );
}
