const API_BASE = '/api';

async function request(url, options = {}) {
    const token = localStorage.getItem('rideflow_token');
    const headers = { 'Content-Type': 'application/json', ...options.headers };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}${url}`, { ...options, headers });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');
    return data;
}

export const api = {
    // Auth
    signup: (body) => request('/auth/signup', { method: 'POST', body: JSON.stringify(body) }),
    login: (body) => request('/auth/login', { method: 'POST', body: JSON.stringify(body) }),
    me: () => request('/auth/me'),

    // Rides
    requestRide: (body) => request('/rides/request', { method: 'POST', body: JSON.stringify(body) }),
    activeRide: () => request('/rides/active'),
    updateRideStatus: (id, body) => request(`/rides/${id}/status`, { method: 'PATCH', body: JSON.stringify(body) }),
    rideHistory: () => request('/rides/history'),

    // Drivers
    applyDriver: (body) => request('/drivers/apply', { method: 'POST', body: JSON.stringify(body) }),
    driverApplications: () => request('/drivers/applications'),
    availableDrivers: () => request('/drivers/available'),
    driverProfile: () => request('/drivers/profile'),
    toggleOnline: () => request('/drivers/toggle-online', { method: 'PATCH' }),
    driverStats: () => request('/drivers/stats'),

    // Simulation
    simState: () => request('/sim/state'),
    simStep: (body) => request('/sim/step', { method: 'POST', body: JSON.stringify(body) }),
    simReset: () => request('/sim/reset', { method: 'POST' }),
    simHistory: (limit) => request(`/sim/history?limit=${limit || 300}`),
    simSetMode: (mode) => request('/sim/mode', { method: 'PATCH', body: JSON.stringify({ mode }) }),

    // Dashboard
    dashboard: () => request('/dashboard'),
};
