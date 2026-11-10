import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';
import 'leaflet/dist/leaflet.css';

// Fix leaflet default icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const greenIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
});
const redIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
});

const DELHI_CENTER = [28.6139, 77.2090];

const LOCATIONS = [
    { name: 'Connaught Place', lat: 28.6315, lng: 77.2167 },
    { name: 'India Gate', lat: 28.6129, lng: 77.2295 },
    { name: 'Karol Bagh', lat: 28.6514, lng: 77.1907 },
    { name: 'Lajpat Nagar', lat: 28.5677, lng: 77.2433 },
    { name: 'Saket', lat: 28.5244, lng: 77.2066 },
    { name: 'Hauz Khas', lat: 28.5494, lng: 77.2001 },
    { name: 'Nehru Place', lat: 28.5491, lng: 77.2533 },
    { name: 'Chandni Chowk', lat: 28.6506, lng: 77.2303 },
    { name: 'Dwarka', lat: 28.5921, lng: 77.0460 },
    { name: 'Noida Sector 18', lat: 28.5706, lng: 77.3218 },
];

function MapClickHandler({ onPickup, onDropoff, mode }) {
    useMapEvents({
        click(e) {
            if (mode === 'pickup') onPickup({ lat: e.latlng.lat, lng: e.latlng.lng });
            else onDropoff({ lat: e.latlng.lat, lng: e.latlng.lng });
        }
    });
    return null;
}

export default function RiderDashboard() {
    const { user } = useAuth();
    const [pickup, setPickup] = useState(null);
    const [dropoff, setDropoff] = useState(null);
    const [pickupAddr, setPickupAddr] = useState('');
    const [dropoffAddr, setDropoffAddr] = useState('');
    const [mapMode, setMapMode] = useState('pickup');
    const [loading, setLoading] = useState(false);
    const [rideResult, setRideResult] = useState(null);
    const [activeRide, setActiveRide] = useState(null);
    const [drivers, setDrivers] = useState([]);
    const [toast, setToast] = useState(null);

    useEffect(() => {
        api.activeRide().then(d => setActiveRide(d.ride)).catch(() => { });
        api.availableDrivers().then(d => setDrivers(d.drivers)).catch(() => { });
    }, []);

    const showToast = (msg, type = 'success') => {
        setToast({ msg, type });
        setTimeout(() => setToast(null), 3000);
    };

    const selectPreset = (loc, type) => {
        if (type === 'pickup') { setPickup({ lat: loc.lat, lng: loc.lng }); setPickupAddr(loc.name); }
        else { setDropoff({ lat: loc.lat, lng: loc.lng }); setDropoffAddr(loc.name); }
    };

    const requestRide = async () => {
        if (!pickup || !dropoff) return showToast('Set both pickup and dropoff', 'error');
        setLoading(true);
        try {
            const data = await api.requestRide({
                pickup_lat: pickup.lat, pickup_lng: pickup.lng,
                dropoff_lat: dropoff.lat, dropoff_lng: dropoff.lng,
                pickup_addr: pickupAddr, dropoff_addr: dropoffAddr,
            });
            setRideResult(data);
            setActiveRide(data.ride);
            showToast('Ride matched!');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            setLoading(false);
        }
    };

    const updateRideStatus = async (status) => {
        if (!activeRide) return;
        try {
            const data = await api.updateRideStatus(activeRide.id, { status, rating: status === 'completed' ? 5 : undefined });
            setActiveRide(status === 'completed' || status === 'cancelled' ? null : data.ride);
            if (status === 'completed') { setRideResult(null); showToast('Ride completed!'); }
            if (status === 'cancelled') { setRideResult(null); showToast('Ride cancelled', 'error'); }
        } catch (err) {
            showToast(err.message, 'error');
        }
    };

    return (
        <div className="page">
            <div style={{ marginBottom: 20 }} className="animate-in">
                <h2>Welcome back, {user?.name?.split(' ')[0]} 👋</h2>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: 4 }}>Where would you like to go?</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: 20 }}>
                {/* Map */}
                <div className="card animate-in" style={{ overflow: 'hidden', height: 520 }}>
                    <MapContainer center={DELHI_CENTER} zoom={12} style={{ height: '100%', width: '100%' }}>
                        <TileLayer
                            attribution='&copy; <a href="https://osm.org/copyright">OpenStreetMap</a>'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />
                        <MapClickHandler onPickup={setPickup} onDropoff={setDropoff} mode={mapMode} />
                        {pickup && <Marker position={[pickup.lat, pickup.lng]} icon={greenIcon} />}
                        {dropoff && <Marker position={[dropoff.lat, dropoff.lng]} icon={redIcon} />}
                        {drivers.map((d, i) => (
                            <Marker key={i} position={[d.lat, d.lng]}
                                icon={new L.DivIcon({
                                    html: '<div style="font-size:20px">🚗</div>',
                                    iconSize: [24, 24], className: ''
                                })} />
                        ))}
                    </MapContainer>
                </div>

                {/* Right panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {/* Ride request form */}
                    {!activeRide && (
                        <div className="card animate-in" style={{ padding: 20 }}>
                            <h3 style={{ marginBottom: 16 }}>Book a Ride</h3>

                            {/* Map mode toggle */}
                            <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
                                <button onClick={() => setMapMode('pickup')} className={`btn btn-sm ${mapMode === 'pickup' ? 'btn-primary' : 'btn-secondary'}`} style={{ flex: 1 }}>
                                    📍 Set Pickup
                                </button>
                                <button onClick={() => setMapMode('dropoff')} className={`btn btn-sm ${mapMode === 'dropoff' ? 'btn-primary' : 'btn-secondary'}`} style={{ flex: 1 }}>
                                    🏁 Set Dropoff
                                </button>
                            </div>

                            {/* Quick location presets */}
                            <div className="label" style={{ marginBottom: 6 }}>Quick Select Pickup</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 12 }}>
                                {LOCATIONS.slice(0, 5).map(l => (
                                    <button key={l.name} onClick={() => selectPreset(l, 'pickup')} className="btn btn-sm btn-secondary"
                                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}>{l.name}</button>
                                ))}
                            </div>
                            <div className="label" style={{ marginBottom: 6 }}>Quick Select Dropoff</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 16 }}>
                                {LOCATIONS.slice(5).map(l => (
                                    <button key={l.name} onClick={() => selectPreset(l, 'dropoff')} className="btn btn-sm btn-secondary"
                                        style={{ fontSize: '0.7rem', padding: '4px 8px' }}>{l.name}</button>
                                ))}
                            </div>

                            {/* Selected points */}
                            <div style={{ marginBottom: 14 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--emerald)' }} />
                                    <span style={{ fontSize: '0.82rem', color: pickup ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                                        {pickupAddr || (pickup ? `${pickup.lat.toFixed(4)}, ${pickup.lng.toFixed(4)}` : 'Click map or select above')}
                                    </span>
                                </div>
                                <div style={{ width: 2, height: 16, background: 'var(--border-subtle)', marginLeft: 4 }} />
                                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                                    <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--red)' }} />
                                    <span style={{ fontSize: '0.82rem', color: dropoff ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                                        {dropoffAddr || (dropoff ? `${dropoff.lat.toFixed(4)}, ${dropoff.lng.toFixed(4)}` : 'Click map or select above')}
                                    </span>
                                </div>
                            </div>

                            <button onClick={requestRide} className="btn btn-primary btn-lg" disabled={loading || !pickup || !dropoff} style={{ width: '100%' }}>
                                {loading ? '🔄 Finding driver...' : '🚗 Request Ride'}
                            </button>
                        </div>
                    )}

                    {/* Active ride / result */}
                    {rideResult && (
                        <div className="card card-glow animate-in" style={{ padding: 20 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                                <h3>🎉 Ride Matched!</h3>
                                <span className={`badge badge-${rideResult.driver.group === 'veteran' ? 'emerald' : rideResult.driver.group === 'intermediate' ? 'amber' : 'cyan'}`}>
                                    {rideResult.driver.group}
                                </span>
                            </div>

                            <div style={{ display: 'grid', gap: 10, marginBottom: 16 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <div>
                                        <div className="label">Driver</div>
                                        <div style={{ fontWeight: 600, marginTop: 2 }}>{rideResult.driver.name}</div>
                                    </div>
                                    <div style={{ textAlign: 'right' }}>
                                        <div className="label">Rating</div>
                                        <div style={{ fontWeight: 600, marginTop: 2, color: 'var(--amber)' }}>⭐ {rideResult.driver.rating}</div>
                                    </div>
                                </div>

                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8 }}>
                                    <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                                        <div className="label" style={{ marginBottom: 4 }}>Fare</div>
                                        <div style={{ fontFamily: "'Outfit',sans-serif", fontWeight: 700, fontSize: '1.2rem', color: 'var(--emerald)' }}>
                                            ₹{rideResult.ride.fare}
                                        </div>
                                    </div>
                                    <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                                        <div className="label" style={{ marginBottom: 4 }}>Distance</div>
                                        <div style={{ fontWeight: 600 }}>{rideResult.ride.distance_km} km</div>
                                    </div>
                                    <div style={{ padding: '10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                                        <div className="label" style={{ marginBottom: 4 }}>ETA</div>
                                        <div style={{ fontWeight: 600 }}>{rideResult.ride.duration_min} min</div>
                                    </div>
                                </div>

                                <div style={{ padding: '8px 12px', background: 'var(--emerald-bg)', border: '1px solid var(--emerald-border)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', color: 'var(--emerald)' }}>
                                    🤖 {rideResult.algorithm.reason}
                                </div>

                                <div style={{ padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                    🚗 {rideResult.driver.vehicle}
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: 8 }}>
                                <button onClick={() => updateRideStatus('in_progress')} className="btn btn-primary" style={{ flex: 1 }}>
                                    Start Ride
                                </button>
                                <button onClick={() => updateRideStatus('completed')} className="btn btn-outline" style={{ flex: 1 }}>
                                    Complete
                                </button>
                                <button onClick={() => updateRideStatus('cancelled')} className="btn btn-danger btn-sm">
                                    Cancel
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Online drivers */}
                    <div className="card animate-in" style={{ padding: 16 }}>
                        <div className="label" style={{ marginBottom: 10 }}>Nearby Drivers ({drivers.length} online)</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 160, overflowY: 'auto' }}>
                            {drivers.slice(0, 6).map((d, i) => (
                                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                                    <div style={{
                                        width: 28, height: 28, borderRadius: '50%', background: d.driver_group === 'veteran' ? 'var(--emerald-bg)' : d.driver_group === 'intermediate' ? 'var(--amber-bg)' : 'var(--cyan-bg)',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', flexShrink: 0
                                    }}>🚗</div>
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>{d.name}</div>
                                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{d.vehicle_model}</div>
                                    </div>
                                    <span className={`badge badge-${d.driver_group === 'veteran' ? 'emerald' : d.driver_group === 'intermediate' ? 'amber' : 'cyan'}`}
                                        style={{ fontSize: '0.55rem' }}>{d.driver_group}</span>
                                    <span style={{ fontSize: '0.78rem', color: 'var(--amber)' }}>⭐{d.rating}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {toast && <div className={`toast toast-${toast.type}`}>{toast.msg}</div>}
        </div>
    );
}
