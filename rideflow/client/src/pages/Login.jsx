import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const data = await login(email, password);
            const role = data.user.role;
            navigate(role === 'driver' ? '/driver' : role === 'admin' ? '/simulation' : '/dashboard');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const quickLogin = (email) => {
        setEmail(email);
        setPassword('password123');
    };

    return (
        <div className="auth-page">
            <div className="auth-card card animate-in">
                <div style={{ textAlign: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: '2.5rem', marginBottom: 4 }}>🚗</div>
                    <h1 style={{ background: 'linear-gradient(135deg, var(--emerald), var(--cyan))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        RideFlow
                    </h1>
                </div>
                <p className="subtitle">Fair ride-sharing powered by UCB algorithm</p>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {error && (
                        <div style={{ padding: '10px 14px', background: 'var(--red-bg)', border: '1px solid var(--red-border)', borderRadius: 'var(--radius-sm)', color: 'var(--red)', fontSize: '0.82rem' }}>
                            {error}
                        </div>
                    )}
                    <div className="input-group">
                        <label>Email Address</label>
                        <input className="input" type="email" placeholder="you@email.com" value={email} onChange={e => setEmail(e.target.value)} required />
                    </div>
                    <div className="input-group">
                        <label>Password</label>
                        <input className="input" type="password" placeholder="••••••••" value={password} onChange={e => setPassword(e.target.value)} required />
                    </div>
                    <button className="btn btn-primary btn-lg" type="submit" disabled={loading} style={{ width: '100%' }}>
                        {loading ? 'Signing in...' : 'Sign In'}
                    </button>
                </form>

                <div className="auth-footer">
                    Don't have an account? <Link to="/signup">Create one</Link>
                </div>

                {/* Quick login buttons */}
                <div style={{ marginTop: 24, padding: '16px', background: 'var(--bg-elevated)', borderRadius: 'var(--radius-sm)' }}>
                    <div className="label" style={{ marginBottom: 10 }}>Quick Demo Login</div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <button onClick={() => quickLogin('rider@rideflow.com')} className="btn btn-sm btn-outline" style={{ flex: 1 }}>🧑 Rider</button>
                        <button onClick={() => quickLogin('raj@rideflow.com')} className="btn btn-sm btn-outline" style={{ flex: 1 }}>🚗 Driver</button>
                        <button onClick={() => quickLogin('admin@rideflow.com')} className="btn btn-sm btn-outline" style={{ flex: 1 }}>👑 Admin</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
