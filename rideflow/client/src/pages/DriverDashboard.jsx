import { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';

export default function DriverDashboard() {
    const { user } = useAuth();
    const [profile, setProfile] = useState(null);
    const [activeRide, setActiveRide] = useState(null);
    const [loading, setLoading] = useState(true);
    const [toast, setToast] = useState(null);

    const showToast = (msg, type = 'success') => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3000);
    };

    const load = async () => {
        try {
            const [prof, ride] = await Promise.all([
                api.driverProfile().catch(() => null),
                api.activeRide().catch(() => ({ ride: null })),
            ]);
            if (prof?.profile) setProfile(prof.profile);
            setActiveRide(ride?.ride);
        } catch { }
        setLoading(false);
    };

    useEffect(() => { load(); }, []);

    const toggleOnline = async () => {
        try {
            const data = await api.toggleOnline();
            setProfile(p => ({ ...p, is_online: data.is_online ? 1 : 0 }));
            showToast(data.is_online ? 'You are now online!' : 'You are now offline');
        } catch (err) { showToast(err.message, 'error'); }
    };

    const updateRide = async (status) => {
        if (!activeRide) return;
        try {
            const data = await api.updateRideStatus(activeRide.id, { status, rating: 5 });
            if (status === 'completed' || status === 'cancelled') {
                setActiveRide(null);
                showToast(status === 'completed' ? 'Ride completed! 🎉' : 'Ride cancelled');
                load();
            } else {
                setActiveRide(data.ride);
            }
        } catch (err) { showToast(err.message, 'error'); }
    };

    if (loading) return <div className="page"><div className="skeleton" style={{ height: 200, marginBottom: 16 }} /><div className="skeleton" style={{ height: 300 }} /></div>;

    if (!profile) {
        return (
            <div className="page">
                <div className="card animate-in" style={{ padding: 40, textAlign: 'center' }}>
                    <div style={{ fontSize: '3rem', marginBottom: 16 }}>🚗</div>
                    <h2>Become a Driver</h2>
                    <p style={{ color: 'var(--text-secondary)', margin: '12px 0 24px', maxWidth: 400, marginInline: 'auto' }}>
                        Submit your driver application to get started. The Fair UCB algorithm will evaluate your profile.
                    </p>
                    <a href="/apply" className="btn btn-primary btn-lg">Apply Now</a>
                </div>
            </div>
        );
    }

    const groupColor = profile.driver_group === 'veteran' ? 'var(--emerald)' : profile.driver_group === 'intermediate' ? 'var(--amber)' : 'var(--cyan)';
    const groupBg = profile.driver_group === 'veteran' ? 'var(--emerald-bg)' : profile.driver_group === 'intermediate' ? 'var(--amber-bg)' : 'var(--cyan-bg)';

    return (
        <div className="page">
            {/* Header bar */}
            <div className="animate-in" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                <div>
                    <h2>Driver Dashboard</h2>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: 4 }}>
                        Welcome, {profile.name}
                    </p>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <span className={`badge badge-${profile.driver_group === 'veteran' ? 'emerald' : profile.driver_group === 'intermediate' ? 'amber' : 'cyan'}`}
                        style={{ fontSize: '0.7rem', padding: '4px 12px' }}>
                        {profile.driver_group.toUpperCase()}
                    </span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontSize: '0.82rem', color: profile.is_online ? 'var(--emerald)' : 'var(--text-muted)', fontWeight: 600 }}>
                            {profile.is_online ? '● Online' : '○ Offline'}
                        </span>
                        <div className="toggle" onClick={toggleOnline}
                            style={{
                                background: profile.is_online ? 'rgba(16,185,129,0.15)' : 'rgba(100,116,139,0.15)',
                                border: `1px solid ${profile.is_online ? 'rgba(16,185,129,0.4)' : 'rgba(100,116,139,0.3)'}`,
                            }}>
                            <div className="toggle-knob" style={{
                                left: profile.is_online ? 24 : 3,
                                background: profile.is_online ? 'var(--emerald)' : '#64748b',
                                boxShadow: `0 0 8px ${profile.is_online ? 'var(--emerald)' : 'transparent'}`,
                            }} />
                        </div>
                    </div>
                </div>
            </div>

            {/* Stats grid */}
            <div className="grid-4 stagger" style={{ marginBottom: 20 }}>
                {[
                    { label: 'Total Rides', value: profile.total_rides.toLocaleString(), color: 'var(--text-primary)', icon: '🚗' },
                    { label: 'Rating', value: `${profile.rating}`, color: 'var(--amber)', icon: '⭐' },
                    { label: 'Experience', value: `${profile.experience_years}yr`, color: 'var(--cyan)', icon: '📅' },
                    { label: 'Satisfaction', value: `${(profile.satisfaction_rate * 100).toFixed(0)}%`, color: 'var(--emerald)', icon: '😊' },
                ].map((s, i) => (
                    <div key={i} className="card stat-card">
                        <div className="label stat-label">{s.icon} {s.label}</div>
                        <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
                    </div>
                ))}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                {/* Active ride */}
                <div className="card animate-in" style={{ padding: 20 }}>
                    <h3 style={{ marginBottom: 16 }}>Current Ride</h3>
                    {activeRide ? (
                        <div>
                            <div style={{ display: 'grid', gap: 10, marginBottom: 16 }}>
                                <div style={{ padding: '12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <div className="label" style={{ marginBottom: 4 }}>Status</div>
                                    <span className={`badge ${activeRide.status === 'in_progress' ? 'badge-amber' : 'badge-emerald'}`}>
                                        {activeRide.status.replace('_', ' ')}
                                    </span>
                                </div>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                                    <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                        <div className="label" style={{ marginBottom: 4 }}>Fare</div>
                                        <div style={{ fontWeight: 700, color: 'var(--emerald)', fontSize: '1.1rem' }}>₹{activeRide.fare}</div>
                                    </div>
                                    <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                        <div className="label" style={{ marginBottom: 4 }}>Distance</div>
                                        <div style={{ fontWeight: 600 }}>{activeRide.distance_km} km</div>
                                    </div>
                                </div>
                            </div>
                            <div style={{ display: 'flex', gap: 8 }}>
                                {activeRide.status === 'matched' && (
                                    <button onClick={() => updateRide('in_progress')} className="btn btn-primary" style={{ flex: 1 }}>
                                        🚗 Start Trip
                                    </button>
                                )}
                                {activeRide.status === 'in_progress' && (
                                    <button onClick={() => updateRide('completed')} className="btn btn-primary" style={{ flex: 1 }}>
                                        ✅ Complete Trip
                                    </button>
                                )}
                                <button onClick={() => updateRide('cancelled')} className="btn btn-danger btn-sm">Cancel</button>
                            </div>
                        </div>
                    ) : (
                        <div style={{ textAlign: 'center', padding: 30, color: 'var(--text-muted)' }}>
                            <div style={{ fontSize: '2rem', marginBottom: 8 }}>🕐</div>
                            <p style={{ fontSize: '0.85rem' }}>{profile.is_online ? 'Waiting for ride requests...' : 'Go online to receive rides'}</p>
                        </div>
                    )}
                </div>

                {/* Profile card */}
                <div className="card animate-in" style={{ padding: 20 }}>
                    <h3 style={{ marginBottom: 16 }}>Driver Profile</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 }}>
                        <div style={{
                            width: 48, height: 48, borderRadius: '50%', background: groupBg, display: 'flex', alignItems: 'center', justifyContent: 'center',
                            border: `2px solid ${groupColor}`, fontSize: '1.2rem', fontWeight: 700, color: groupColor
                        }}>
                            {profile.name?.charAt(0)}
                        </div>
                        <div>
                            <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>{profile.name}</div>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{profile.email}</div>
                        </div>
                    </div>
                    <div style={{ display: 'grid', gap: 8 }}>
                        {[
                            { label: 'Vehicle', value: `${profile.vehicle_model} (${profile.vehicle_plate})` },
                            { label: 'Group', value: profile.driver_group.charAt(0).toUpperCase() + profile.driver_group.slice(1), color: groupColor },
                            { label: 'Phone', value: profile.phone || 'N/A' },
                            { label: 'Member Since', value: new Date(profile.created_at).toLocaleDateString() },
                        ].map((item, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{item.label}</span>
                                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: item.color || 'var(--text-primary)' }}>{item.value}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
        </div>
    );
}
