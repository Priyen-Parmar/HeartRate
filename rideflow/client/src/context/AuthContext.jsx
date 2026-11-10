import { createContext, useContext, useState, useEffect } from 'react';
import { api } from '../api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [driverProfile, setDriverProfile] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('rideflow_token');
        if (token) {
            api.me().then(data => {
                setUser(data.user);
                setDriverProfile(data.driverProfile);
            }).catch(() => {
                localStorage.removeItem('rideflow_token');
            }).finally(() => setLoading(false));
        } else {
            setLoading(false);
        }
    }, []);

    const login = async (email, password) => {
        const data = await api.login({ email, password });
        localStorage.setItem('rideflow_token', data.token);
        setUser(data.user);
        setDriverProfile(data.driverProfile);
        return data;
    };

    const signup = async (fields) => {
        const data = await api.signup(fields);
        localStorage.setItem('rideflow_token', data.token);
        setUser(data.user);
        return data;
    };

    const logout = () => {
        localStorage.removeItem('rideflow_token');
        setUser(null);
        setDriverProfile(null);
    };

    const refreshProfile = async () => {
        try {
            const data = await api.me();
            setUser(data.user);
            setDriverProfile(data.driverProfile);
        } catch { }
    };

    return (
        <AuthContext.Provider value={{ user, driverProfile, loading, login, signup, logout, refreshProfile }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const ctx = useContext(AuthContext);
    if (!ctx) throw new Error('useAuth must be inside AuthProvider');
    return ctx;
}
