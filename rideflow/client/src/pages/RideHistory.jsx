import { useState, useEffect } from 'react';
import { api } from '../api';

export default function RideHistory() {
    const [rides, setRides] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.rideHistory().then(d => setRides(d.rides)).catch(() => { }).finally(() => setLoading(false));
    }, []);

    if (loading) return (
        <div className="page">
            <h2 style={{ marginBottom: 20 }}>Ride History</h2>
            {[1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 80, marginBottom: 10 }} />)}
        </div>
    );

    const statusColors = {
        completed: 'badge-emerald', matched: 'badge-amber', in_progress: 'badge-amber',
        requested: 'badge-cyan', cancelled: 'badge-red',
    };

    return (
        <div className="page">
            <div className="animate-in" style={{ marginBottom: 20 }}>
                <h2>Ride History</h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: 4 }}>
                    {rides.length} total rides
                </p>
            </div>

            <div className="stagger">
                {rides.length === 0 ? (
                    <div className="card" style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                        <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🕐</div>
                        <p>No rides yet. Book your first ride!</p>
                    </div>
                ) : rides.map((ride, i) => (
                    <div key={ride.id} className="card ride-card">
                        <div className="ride-header">
                            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono',monospace" }}>
                                    #{ride.id}
                                </span>
                                <span className={`badge ${statusColors[ride.status]}`}>{ride.status}</span>
                                {ride.driver_group && (
                                    <span className={`badge badge-${ride.driver_group === 'veteran' ? 'emerald' : ride.driver_group === 'intermediate' ? 'amber' : 'cyan'}`}
                                        style={{ fontSize: '0.55rem' }}>
                                        {ride.driver_group}
                                    </span>
                                )}
                            </div>
                            <div style={{ textAlign: 'right' }}>
                                <div style={{ fontFamily: "'Outfit',sans-serif", fontWeight: 700, color: 'var(--emerald)', fontSize: '1.1rem' }}>₹{ride.fare}</div>
                            </div>
                        </div>

                        <div className="ride-route">
                            <div className="ride-route-dots">
                                <div className="ride-dot" style={{ background: 'var(--emerald)' }} />
                                <div className="ride-line" />
                                <div className="ride-dot" style={{ background: 'var(--red)' }} />
                            </div>
                            <div className="ride-addrs">
                                <div className="ride-addr">{ride.pickup_addr || `${ride.pickup_lat?.toFixed(4)}, ${ride.pickup_lng?.toFixed(4)}`}</div>
                                <div className="ride-addr">{ride.dropoff_addr || `${ride.dropoff_lat?.toFixed(4)}, ${ride.dropoff_lng?.toFixed(4)}`}</div>
                            </div>
                        </div>

                        <div className="ride-meta">
                            <span>📏 {ride.distance_km} km</span>
                            <span>⏱️ {ride.duration_min} min</span>
                            {ride.driver_name && <span>🚗 {ride.driver_name}</span>}
                            {ride.vehicle_model && <span>🚙 {ride.vehicle_model}</span>}
                            {ride.rating > 0 && <span>⭐ {ride.rating}/5</span>}
                            <span style={{ color: ride.algorithm_mode === 'fair' ? 'var(--emerald)' : 'var(--amber)' }}>
                                🤖 {ride.algorithm_mode === 'fair' ? 'Fair UCB' : 'Greedy'}
                            </span>
                            <span>{new Date(ride.created_at).toLocaleString()}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
