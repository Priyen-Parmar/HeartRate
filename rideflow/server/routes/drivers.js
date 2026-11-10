const express = require('express');
const db = require('../db');
const { authMiddleware } = require('../auth');
const { evaluateApplication, classifyDriverGroup } = require('../fairUcb');

const router = express.Router();

// ── POST /api/drivers/apply ───────────────────────────────────────────────────
router.post('/apply', authMiddleware, (req, res) => {
    try {
        const { name, experience_years, previous_rides, resume_text, vehicle_model, vehicle_plate } = req.body;
        if (!name) return res.status(400).json({ error: 'Name is required' });

        // Check if already applied
        const existing = db.prepare('SELECT id FROM driver_applications WHERE user_id = ? AND status = ?')
            .get(req.user.id, 'pending');
        if (existing) return res.status(409).json({ error: 'You already have a pending application' });

        // Get current algorithm state
        const algoState = db.prepare('SELECT * FROM algorithm_state WHERE id = 1').get();

        // Evaluate application using Fair UCB
        const evaluation = evaluateApplication(
            { experience_years: experience_years || 0, previous_rides: previous_rides || 0 },
            algoState,
            algoState.mode
        );

        // Insert application
        const appResult = db.prepare(`
      INSERT INTO driver_applications (user_id, name, experience_years, previous_rides, resume_text,
                                       group_assigned, status, ucb_score, fairness_reason)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
            req.user.id, name,
            experience_years || 0, previous_rides || 0, resume_text || '',
            evaluation.group,
            evaluation.shouldAccept ? 'accepted' : 'rejected',
            evaluation.ucbScore,
            evaluation.fairnessReason
        );

        // If accepted, create driver profile and update user role
        if (evaluation.shouldAccept) {
            db.prepare('UPDATE users SET role = ? WHERE id = ?').run('driver', req.user.id);

            const existingProfile = db.prepare('SELECT id FROM driver_profiles WHERE user_id = ?').get(req.user.id);
            if (!existingProfile) {
                // Assign random location near Delhi for demo
                const baseLat = 28.6139 + (Math.random() - 0.5) * 0.08;
                const baseLng = 77.2090 + (Math.random() - 0.5) * 0.08;

                db.prepare(`
          INSERT INTO driver_profiles (user_id, experience_years, total_rides, satisfaction_rate, rating,
                                       is_online, is_approved, driver_group, vehicle_model, vehicle_plate, lat, lng)
          VALUES (?, ?, ?, 0.5, 4.5, 1, 1, ?, ?, ?, ?, ?)
        `).run(
                    req.user.id,
                    experience_years || 0,
                    previous_rides || 0,
                    evaluation.group,
                    vehicle_model || 'Sedan',
                    vehicle_plate || 'XX-00-XX-0000',
                    baseLat, baseLng
                );
            }
        }

        const application = db.prepare('SELECT * FROM driver_applications WHERE id = ?').get(appResult.lastInsertRowid);
        res.status(201).json({
            application,
            evaluation: {
                group: evaluation.group,
                ucbScore: evaluation.ucbScore,
                accepted: evaluation.shouldAccept,
                reason: evaluation.fairnessReason,
            }
        });
    } catch (err) {
        console.error('Application error:', err);
        res.status(500).json({ error: 'Server error' });
    }
});

// ── GET /api/drivers/applications ─────────────────────────────────────────────
router.get('/applications', authMiddleware, (req, res) => {
    const apps = db.prepare(`
    SELECT da.*, u.email
    FROM driver_applications da
    JOIN users u ON u.id = da.user_id
    ORDER BY da.created_at DESC
  `).all();
    res.json({ applications: apps });
});

// ── GET /api/drivers/available ────────────────────────────────────────────────
router.get('/available', authMiddleware, (req, res) => {
    const drivers = db.prepare(`
    SELECT dp.*, u.name, u.email, u.phone
    FROM driver_profiles dp
    JOIN users u ON u.id = dp.user_id
    WHERE dp.is_online = 1 AND dp.is_approved = 1
    ORDER BY dp.rating DESC
  `).all();
    res.json({ drivers });
});

// ── GET /api/drivers/profile ──────────────────────────────────────────────────
router.get('/profile', authMiddleware, (req, res) => {
    const profile = db.prepare(`
    SELECT dp.*, u.name, u.email, u.phone
    FROM driver_profiles dp
    JOIN users u ON u.id = dp.user_id
    WHERE dp.user_id = ?
  `).get(req.user.id);
    if (!profile) return res.status(404).json({ error: 'Driver profile not found' });
    res.json({ profile });
});

// ── PATCH /api/drivers/toggle-online ──────────────────────────────────────────
router.patch('/toggle-online', authMiddleware, (req, res) => {
    const profile = db.prepare('SELECT * FROM driver_profiles WHERE user_id = ?').get(req.user.id);
    if (!profile) return res.status(404).json({ error: 'Driver profile not found' });
    const newStatus = profile.is_online ? 0 : 1;
    db.prepare('UPDATE driver_profiles SET is_online = ? WHERE user_id = ?').run(newStatus, req.user.id);
    res.json({ is_online: !!newStatus });
});

// ── GET /api/drivers/stats ────────────────────────────────────────────────────
router.get('/stats', authMiddleware, (req, res) => {
    const totalDrivers = db.prepare('SELECT COUNT(*) as count FROM driver_profiles WHERE is_approved = 1').get();
    const byGroup = db.prepare(`
    SELECT driver_group, COUNT(*) as count, AVG(rating) as avg_rating, SUM(total_rides) as total_rides
    FROM driver_profiles WHERE is_approved = 1 GROUP BY driver_group
  `).all();
    const onlineCount = db.prepare('SELECT COUNT(*) as count FROM driver_profiles WHERE is_online = 1 AND is_approved = 1').get();
    res.json({
        total: totalDrivers.count,
        online: onlineCount.count,
        groups: byGroup,
    });
});

module.exports = router;
