const express = require('express');
const cors = require('cors');
const path = require('path');

// Initialize DB (creates tables)
const db = require('./db');

const { router: authRouter, authMiddleware } = require('./auth');
const ridesRouter = require('./routes/rides');
const driversRouter = require('./routes/drivers');
const simulationRouter = require('./routes/simulation');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({ origin: true, credentials: true }));
app.use(express.json());

// Routes
app.use('/api/auth', authRouter);
app.use('/api/rides', ridesRouter);
app.use('/api/drivers', driversRouter);
app.use('/api/sim', simulationRouter);

// Health check
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
});

// Dashboard stats
app.get('/api/dashboard', authMiddleware, (req, res) => {
    const totalUsers = db.prepare('SELECT COUNT(*) as c FROM users').get().c;
    const totalRides = db.prepare('SELECT COUNT(*) as c FROM rides').get().c;
    const completedRides = db.prepare("SELECT COUNT(*) as c FROM rides WHERE status = 'completed'").get().c;
    const totalDrivers = db.prepare('SELECT COUNT(*) as c FROM driver_profiles WHERE is_approved = 1').get().c;
    const onlineDrivers = db.prepare('SELECT COUNT(*) as c FROM driver_profiles WHERE is_online = 1 AND is_approved = 1').get().c;
    const algoState = db.prepare('SELECT * FROM algorithm_state WHERE id = 1').get();
    const recentRides = db.prepare(`
    SELECT r.*, u1.name as rider_name, u2.name as driver_name, dp.driver_group
    FROM rides r
    JOIN users u1 ON u1.id = r.rider_id
    LEFT JOIN users u2 ON u2.id = r.driver_id
    LEFT JOIN driver_profiles dp ON dp.user_id = r.driver_id
    ORDER BY r.created_at DESC LIMIT 10
  `).all();

    res.json({ totalUsers, totalRides, completedRides, totalDrivers, onlineDrivers, algoState, recentRides });
});

app.listen(PORT, () => {
    console.log(`\n  🚗 RideFlow Server running on http://localhost:${PORT}`);
    console.log(`  📊 Fair UCB Algorithm active\n`);
});
