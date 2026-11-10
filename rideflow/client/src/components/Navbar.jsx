import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
    const { user, logout } = useAuth();
    const location = useLocation();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    if (!user) return null;

    const isActive = (path) => location.pathname === path ? 'nav-link active' : 'nav-link';

    const getLinks = () => {
        switch (user.role) {
            case 'rider':
                return [
                    { to: '/dashboard', label: '🏠 Dashboard' },
                    { to: '/ride-history', label: '📋 History' },
                ];
            case 'driver':
                return [
                    { to: '/driver', label: '🚗 Dashboard' },
                    { to: '/ride-history', label: '📋 History' },
                ];
            case 'admin':
                return [
                    { to: '/dashboard', label: '📊 Dashboard' },
                    { to: '/simulation', label: '🧪 Fair UCB' },
                    { to: '/applications', label: '📝 Applications' },
                ];
            default:
                return [];
        }
    };

    return (
        <nav className="navbar">
            <Link to="/" className="navbar-brand">RideFlow</Link>
            <div className="navbar-links">
                {getLinks().map(l => (
                    <Link key={l.to} to={l.to} className={isActive(l.to)}>{l.label}</Link>
                ))}
            </div>
            <div className="navbar-user">
                <div className="navbar-avatar">{user.name?.charAt(0)?.toUpperCase()}</div>
                <div style={{ fontSize: '0.82rem' }}>
                    <div style={{ fontWeight: 600 }}>{user.name}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>{user.role}</div>
                </div>
                <button onClick={handleLogout} className="btn btn-sm btn-secondary" style={{ marginLeft: 8 }}>
                    Logout
                </button>
            </div>
        </nav>
    );
}
