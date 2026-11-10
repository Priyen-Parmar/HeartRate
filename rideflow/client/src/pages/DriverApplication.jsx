import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';

export default function DriverApplication() {
    const { refreshProfile } = useAuth();
    const navigate = useNavigate();
    const [form, setForm] = useState({
        name: '', experience_years: '', previous_rides: '', resume_text: '',
        vehicle_model: '', vehicle_plate: '',
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const update = (k, v) => setForm(f => ({ ...f, [k]: v }));

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const data = await api.applyDriver({
                ...form,
                experience_years: parseFloat(form.experience_years) || 0,
                previous_rides: parseInt(form.previous_rides) || 0,
            });
            setResult(data);
            await refreshProfile();
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const groupInfo = {
        veteran: {
            color: 'var(--emerald)', bg: 'var(--emerald-bg)', border: 'var(--emerald-border)',
            badge: 'badge-emerald', icon: '🏆',
            desc: 'Top-tier drivers with extensive experience and high satisfaction rates. Typically receive the most ride assignments under pure UCB1.',
        },
        intermediate: {
            color: 'var(--amber)', bg: 'var(--amber-bg)', border: 'var(--amber-border)',
            badge: 'badge-amber', icon: '🔄',
            desc: 'Mid-level drivers with moderate experience. The Fair UCB algorithm guarantees them at least 15% of ride allocations.',
        },
        newbie: {
            color: 'var(--cyan)', bg: 'var(--cyan-bg)', border: 'var(--cyan-border)',
            badge: 'badge-cyan', icon: '🌱',
            desc: 'New drivers just starting out. The Fair UCB algorithm ensures they get at least 25% of ride allocations, protecting against cold-start exclusion.',
        },
    };

    if (result) {
        const g = groupInfo[result.evaluation.group];
        return (
            <div className="page" style={{ maxWidth: 700, margin: '0 auto' }}>
                <div className="card animate-in" style={{ padding: 32 }}>
                    <div style={{ textAlign: 'center', marginBottom: 24 }}>
                        <div style={{ fontSize: '3rem', marginBottom: 8 }}>{result.evaluation.accepted ? '🎉' : '😔'}</div>
                        <h2 style={{ color: result.evaluation.accepted ? 'var(--emerald)' : 'var(--red)' }}>
                            Application {result.evaluation.accepted ? 'Accepted!' : 'Rejected'}
                        </h2>
                    </div>

                    {/* Classification result */}
                    <div style={{ marginBottom: 24, padding: 20, background: g.bg, border: `1px solid ${g.border}`, borderRadius: 'var(--radius-md)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                            <span style={{ fontSize: '1.8rem' }}>{g.icon}</span>
                            <div>
                                <div className="label" style={{ color: g.color }}>Classified As</div>
                                <div style={{ fontFamily: "'Outfit',sans-serif", fontSize: '1.4rem', fontWeight: 800, color: g.color, textTransform: 'capitalize' }}>
                                    {result.evaluation.group}
                                </div>
                            </div>
                        </div>
                        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{g.desc}</p>
                    </div>

                    {/* Algorithm details */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
                        <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                            <div className="label" style={{ marginBottom: 6 }}>UCB Score</div>
                            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: '1.3rem', fontWeight: 700, color: 'var(--amber)' }}>
                                {result.evaluation.ucbScore}
                            </div>
                        </div>
                        <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                            <div className="label" style={{ marginBottom: 6 }}>Algorithm Mode</div>
                            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--emerald)' }}>FAIR-UCB</div>
                        </div>
                    </div>

                    {/* Reason */}
                    <div style={{ padding: '14px', background: 'var(--emerald-bg)', border: '1px solid var(--emerald-border)', borderRadius: 'var(--radius-sm)', marginBottom: 24 }}>
                        <div className="label" style={{ color: 'var(--emerald)', marginBottom: 6 }}>Algorithm Decision</div>
                        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                            {result.evaluation.reason}
                        </p>
                    </div>

                    {/* Fair UCB explanation */}
                    <div style={{ padding: '14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', marginBottom: 24 }}>
                        <div className="label" style={{ marginBottom: 8 }}>How Fair UCB Works</div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
                            <p>The <strong style={{ color: 'var(--text-secondary)' }}>FAIR-LEARN</strong> algorithm (Patil et al., 2020) classifies drivers into groups and sets minimum allocation quotas:</p>
                            <ul style={{ paddingLeft: 16, marginTop: 8 }}>
                                <li><span style={{ color: 'var(--emerald)' }}>Veterans</span> — No minimum quota (naturally selected by UCB1)</li>
                                <li><span style={{ color: 'var(--amber)' }}>Intermediates</span> — Guaranteed ≥15% of ride allocations</li>
                                <li><span style={{ color: 'var(--cyan)' }}>Newbies</span> — Guaranteed ≥25% of ride allocations</li>
                            </ul>
                            <p style={{ marginTop: 8 }}>This ensures fair opportunity for all drivers regardless of experience level.</p>
                        </div>
                    </div>

                    <button onClick={() => navigate('/driver')} className="btn btn-primary btn-lg" style={{ width: '100%' }}>
                        Go to Driver Dashboard →
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="page" style={{ maxWidth: 600, margin: '0 auto' }}>
            <div className="card animate-in" style={{ padding: 32 }}>
                <div style={{ textAlign: 'center', marginBottom: 24 }}>
                    <div style={{ fontSize: '2.5rem', marginBottom: 8 }}>🚗</div>
                    <h2>Driver Application</h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: 8 }}>
                        Submit your profile — the Fair UCB algorithm will classify you and evaluate your application
                    </p>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {error && (
                        <div style={{ padding: '10px 14px', background: 'var(--red-bg)', border: '1px solid var(--red-border)', borderRadius: 'var(--radius-sm)', color: 'var(--red)', fontSize: '0.82rem' }}>
                            {error}
                        </div>
                    )}

                    <div className="input-group">
                        <label>Full Name</label>
                        <input className="input" placeholder="Your full name" value={form.name} onChange={e => update('name', e.target.value)} required />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                        <div className="input-group">
                            <label>Years of Driving Experience</label>
                            <input className="input" type="number" step="0.5" min="0" placeholder="e.g. 3" value={form.experience_years} onChange={e => update('experience_years', e.target.value)} />
                        </div>
                        <div className="input-group">
                            <label>Previous Rides Completed</label>
                            <input className="input" type="number" min="0" placeholder="e.g. 500" value={form.previous_rides} onChange={e => update('previous_rides', e.target.value)} />
                        </div>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                        <div className="input-group">
                            <label>Vehicle Model</label>
                            <input className="input" placeholder="e.g. Maruti Dzire" value={form.vehicle_model} onChange={e => update('vehicle_model', e.target.value)} />
                        </div>
                        <div className="input-group">
                            <label>Vehicle Plate</label>
                            <input className="input" placeholder="e.g. DL-01-AB-1234" value={form.vehicle_plate} onChange={e => update('vehicle_plate', e.target.value)} />
                        </div>
                    </div>

                    <div className="input-group">
                        <label>About You (Resume / Description)</label>
                        <textarea className="input" rows={4} placeholder="Tell us about your driving experience, skills, and why you want to join RideFlow..."
                            value={form.resume_text} onChange={e => update('resume_text', e.target.value)} />
                    </div>

                    {/* Classification preview */}
                    <div style={{ padding: 14, background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                        <div className="label" style={{ marginBottom: 8 }}>Group Classification Preview</div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                            {(!form.experience_years && !form.previous_rides) ? (
                                <span>Fill in your experience to see your estimated group classification.</span>
                            ) : (
                                <>
                                    Based on your profile, you'll likely be classified as:
                                    <strong style={{
                                        color: (parseFloat(form.experience_years) >= 3 || parseInt(form.previous_rides) >= 500)
                                            ? 'var(--emerald)' : (parseFloat(form.experience_years) >= 1 || parseInt(form.previous_rides) >= 100)
                                                ? 'var(--amber)' : 'var(--cyan)',
                                        marginLeft: 4,
                                    }}>
                                        {(parseFloat(form.experience_years) >= 3 || parseInt(form.previous_rides) >= 500)
                                            ? '🏆 Veteran' : (parseFloat(form.experience_years) >= 1 || parseInt(form.previous_rides) >= 100)
                                                ? '🔄 Intermediate' : '🌱 Newbie'}
                                    </strong>
                                </>
                            )}
                        </div>
                    </div>

                    <button className="btn btn-primary btn-lg" type="submit" disabled={loading} style={{ width: '100%' }}>
                        {loading ? '🤖 Evaluating with Fair UCB...' : 'Submit Application'}
                    </button>
                </form>
            </div>
        </div>
    );
}
