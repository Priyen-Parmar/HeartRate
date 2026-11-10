const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const db = require('./db');

const router = express.Router();
const JWT_SECRET = 'rideflow_fair_ucb_secret_2024';

// ── Middleware: verify JWT ────────────────────────────────────────────────────
function authMiddleware(req, res, next) {
    const header = req.headers.authorization;
    if (!header || !header.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'No token provided' });
    }
    try {
        const decoded = jwt.verify(header.split(' ')[1], JWT_SECRET);
        req.user = decoded;
        next();
    } catch {
        return res.status(401).json({ error: 'Invalid token' });
    }
}

// ── POST /api/auth/signup ─────────────────────────────────────────────────────
router.post('/signup', async (req, res) => {
    try {
        const { email, password, name, phone, role } = req.body;
        if (!email || !password || !name || !role) {
            return res.status(400).json({ error: 'Missing required fields' });
        }
        if (!['rider', 'driver', 'admin'].includes(role)) {
            return res.status(400).json({ error: 'Invalid role' });
        }
        const exists = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
        if (exists) {
            return res.status(409).json({ error: 'Email already registered' });
        }
        const hash = await bcrypt.hash(password, 10);
        const result = db.prepare(
            'INSERT INTO users (email, password, name, phone, role) VALUES (?, ?, ?, ?, ?)'
        ).run(email, hash, name, phone || '', role);

        const token = jwt.sign({ id: result.lastInsertRowid, email, name, role }, JWT_SECRET, { expiresIn: '7d' });
        res.status(201).json({
            token,
            user: { id: result.lastInsertRowid, email, name, role, phone: phone || '' }
        });
    } catch (err) {
        console.error('Signup error:', err);
        res.status(500).json({ error: 'Server error' });
    }
});

// ── POST /api/auth/login ──────────────────────────────────────────────────────
router.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        if (!email || !password) {
            return res.status(400).json({ error: 'Email and password required' });
        }
        const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email);
        if (!user) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        const valid = await bcrypt.compare(password, user.password);
        if (!valid) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }
        const token = jwt.sign(
            { id: user.id, email: user.email, name: user.name, role: user.role },
            JWT_SECRET,
            { expiresIn: '7d' }
        );
        // If driver, include profile info
        let driverProfile = null;
        if (user.role === 'driver') {
            driverProfile = db.prepare('SELECT * FROM driver_profiles WHERE user_id = ?').get(user.id);
        }
        res.json({
            token,
            user: { id: user.id, email: user.email, name: user.name, role: user.role, phone: user.phone },
            driverProfile
        });
    } catch (err) {
        console.error('Login error:', err);
        res.status(500).json({ error: 'Server error' });
    }
});

// ── GET /api/auth/me ──────────────────────────────────────────────────────────
router.get('/me', authMiddleware, (req, res) => {
    const user = db.prepare('SELECT id, email, name, phone, role, created_at FROM users WHERE id = ?').get(req.user.id);
    if (!user) return res.status(404).json({ error: 'User not found' });
    let driverProfile = null;
    if (user.role === 'driver') {
        driverProfile = db.prepare('SELECT * FROM driver_profiles WHERE user_id = ?').get(user.id);
    }
    res.json({ user, driverProfile });
});

module.exports = { router, authMiddleware, JWT_SECRET };
