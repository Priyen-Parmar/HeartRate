import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Signup() {
    const [form, setForm] = useState({ name: '', email: '', password: '', phone: '', role: 'rider' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { signup } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            await signup(form);
            if (form.role === 'driver') {
                navigate('/apply');
            } else {
                navigate('/dashboard');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const update = (key, val) => setForm(f => ({ ...f, [key]: val }));

    return (
        <div className="auth-page">
            <div className="auth-card card animate-in">
                <div style={{ textAlign: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: '2.5rem', marginBottom: 4 }}>🚗</div>
                    <h1 style={{ background: 'linear-gradient(135deg, var(--emerald), var(--cyan))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        Join RideFlow
                    </h1>
                </div>
                <p className="subtitle">Start riding or driving with fair algorithm-powered matching</p>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {error && (
                        <div style={{ padding: '10px 14px', background: 'var(--red-bg)', border: '1px solid var(--red-border)', borderRadius: 'var(--radius-sm)', color: 'var(--red)', fontSize: '0.82rem' }}>
                            {error}
                        </div>
                    )}

                    {/* Role selection */}
                    <div>
                        <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', letterSpacing: '0.05em', marginBottom: 8, display: 'block' }}>
                            I want to
                        </label>
                        <div className="role-toggle">
                            <button type="button" className={`role-btn ${form.role === 'rider' ? 'active' : ''}`} onClick={() => update('role', 'rider')}>
                                <span className="role-icon">🧑</span> Ride
                            </button>
                            <button type="button" className={`role-btn ${form.role === 'driver' ? 'active' : ''}`} onClick={() => update('role', 'driver')}>
                                <span className="role-icon">🚗</span> Drive
                            </button>
                        </div>
                    </div>

                    <div className="input-group">
                        <label>Full Name</label>
                        <input className="input" placeholder="Your full name" value={form.name} onChange={e => update('name', e.target.value)} required />
                    </div>
                    <div className="input-group">
                        <label>Email Address</label>
                        <input className="input" type="email" placeholder="you@email.com" value={form.email} onChange={e => update('email', e.target.value)} required />
                    </div>
                    <div className="input-group">
                        <label>Phone Number</label>
                        <input className="input" type="tel" placeholder="+91-XXXXXXXXXX" value={form.phone} onChange={e => update('phone', e.target.value)} />
                    </div>
                    <div className="input-group">
                        <label>Password</label>
                        <input className="input" type="password" placeholder="Min 6 characters" value={form.password} onChange={e => update('password', e.target.value)} required minLength={6} />
                    </div>

                    <button className="btn btn-primary btn-lg" type="submit" disabled={loading} style={{ width: '100%' }}>
                        {loading ? 'Creating account...' : form.role === 'driver' ? 'Continue to Application →' : 'Create Account'}
                    </button>
                </form>

                <div className="auth-footer">
                    Already have an account? <Link to="/login">Sign in</Link>
                </div>

                {form.role === 'driver' && (
                    <div style={{ marginTop: 16, padding: 14, background: 'var(--cyan-bg)', border: '1px solid var(--cyan-border)', borderRadius: 'var(--radius-sm)', fontSize: '0.78rem', color: 'var(--cyan)' }}>
                        💡 After signing up, you'll submit a driver application. The <strong>Fair UCB algorithm</strong> will classify you into a group (Veteran, Intermediate, or Newbie) and evaluate your application using fairness-aware matching.
                    </div>
                )}
            </div>
        </div>
    );
}
