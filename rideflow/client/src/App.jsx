import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { api } from './api';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Signup from './pages/Signup';
import RiderDashboard from './pages/RiderDashboard';
import DriverDashboard from './pages/DriverDashboard';
import DriverApplication from './pages/DriverApplication';
import RideHistory from './pages/RideHistory';
import AdminSimulation from './pages/AdminSimulation';

function ProtectedRoute({ children, allowedRoles }) {
    const { user, loading } = useAuth();
    if (loading) return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ textAlign: 'center' }}>
                <div className="pulse" style={{ fontSize: '2rem', marginBottom: 12 }}>🚗</div>
                <div style={{ color: 'var(--text-muted)' }}>Loading...</div>
            </div>
        </div>
    );
    if (!user) return <Navigate to="/login" />;
    if (allowedRoles && !allowedRoles.includes(user.role)) {
        return <Navigate to={user.role === 'driver' ? '/driver' : user.role === 'admin' ? '/simulation' : '/dashboard'} />;
    }
    return children;
}

function GuestRoute({ children }) {
    const { user, loading } = useAuth();
    if (loading) return null;
    if (user) return <Navigate to={user.role === 'driver' ? '/driver' : user.role === 'admin' ? '/simulation' : '/dashboard'} />;
    return children;
}

function AppRoutes() {
    return (
        <Routes>
            <Route path="/login" element={<GuestRoute><Login /></GuestRoute>} />
            <Route path="/signup" element={<GuestRoute><Signup /></GuestRoute>} />
            <Route path="/dashboard" element={<ProtectedRoute allowedRoles={['rider', 'admin']}><Navbar /><RiderDashboard /></ProtectedRoute>} />
            <Route path="/driver" element={<ProtectedRoute allowedRoles={['driver']}><Navbar /><DriverDashboard /></ProtectedRoute>} />
            <Route path="/apply" element={<ProtectedRoute><Navbar /><DriverApplication /></ProtectedRoute>} />
            <Route path="/ride-history" element={<ProtectedRoute><Navbar /><RideHistory /></ProtectedRoute>} />
            <Route path="/simulation" element={<ProtectedRoute allowedRoles={['admin']}><Navbar /><AdminSimulation /></ProtectedRoute>} />
            <Route path="/applications" element={<ProtectedRoute allowedRoles={['admin']}><Navbar /><ApplicationsList /></ProtectedRoute>} />
            <Route path="*" element={<Navigate to="/login" />} />
        </Routes>
    );
}

function ApplicationsList() {
    const [apps, setApps] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        api.driverApplications().then(d => setApps(d.applications)).catch(() => { }).finally(() => setLoading(false));
    }, []);

    if (loading) return <div className="page"><div className="skeleton" style={{ height: 400 }} /></div>;

    return (
        <div className="page">
            <h2 style={{ marginBottom: 20 }}>Driver Applications</h2>
            <div className="stagger">
                {apps.map(app => (
                    <div key={app.id} className="card" style={{ padding: 16, marginBottom: 10 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                            <div style={{ fontWeight: 600 }}>{app.name}</div>
                            <div style={{ display: 'flex', gap: 6 }}>
                                <span className={`badge badge-${app.group_assigned === 'veteran' ? 'emerald' : app.group_assigned === 'intermediate' ? 'amber' : 'cyan'}`}>
                                    {app.group_assigned}
                                </span>
                                <span className={`badge ${app.status === 'accepted' ? 'badge-emerald' : app.status === 'rejected' ? 'badge-red' : 'badge-amber'}`}>
                                    {app.status}
                                </span>
                            </div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, marginBottom: 8 }}>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Exp: {app.experience_years}yr</div>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>Rides: {app.previous_rides}</div>
                            <div style={{ fontSize: '0.78rem', color: 'var(--amber)' }}>UCB: {app.ucb_score}</div>
                        </div>
                        {app.fairness_reason && (
                            <div style={{ padding: '8px 10px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                🤖 {app.fairness_reason}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

export default function App() {
    return (
        <BrowserRouter>
            <AuthProvider>
                <AppRoutes />
            </AuthProvider>
        </BrowserRouter>
    );
}
