const express = require('express');
const db = require('../db');
const { authMiddleware } = require('../auth');
const { fairLearnSelect, simulateReward, classifyDriverGroup, GROUPS, GROUP_INDEX } = require('../fairUcb');

const router = express.Router();

// ── Helper: calculate fare ────────────────────────────────────────────────────
function calculateFare(distanceKm) {
    const baseFare = 50;       // ₹50 base
    const perKm = 12;          // ₹12/km
    const surcharge = 20;      // platform fee
    return Math.round(baseFare + perKm * distanceKm + surcharge);
}

// ── Helper: haversine distance ────────────────────────────────────────────────
function haversine(lat1, lng1, lat2, lng2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── POST /api/rides/request ───────────────────────────────────────────────────
router.post('/request', authMiddleware, (req, res) => {
    try {
        const { pickup_lat, pickup_lng, dropoff_lat, dropoff_lng, pickup_addr, dropoff_addr } = req.body;
        if (!pickup_lat || !pickup_lng || !dropoff_lat || !dropoff_lng) {
            return res.status(400).json({ error: 'Pickup and dropoff coordinates required' });
        }

        const distance = haversine(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng);
        const fare = calculateFare(distance);
        const duration = Math.round(distance * 3 + 5); // rough estimate

        // Get algorithm state
        const algoState = db.prepare('SELECT * FROM algorithm_state WHERE id = 1').get();

        // Fair-LEARN selects which GROUP should get this ride
        const { chosenGroup, reason } = fairLearnSelect(algoState, algoState.mode);

        // Find an available driver from that group
        let driver = db.prepare(`
      SELECT dp.*, u.name as driver_name, u.email as driver_email, u.phone as driver_phone
      FROM driver_profiles dp
      JOIN users u ON u.id = dp.user_id
      WHERE dp.is_online = 1 AND dp.is_approved = 1 AND dp.driver_group = ?
      ORDER BY ABS(dp.lat - ?) + ABS(dp.lng - ?) ASC
      LIMIT 1
    `).get(chosenGroup, pickup_lat, pickup_lng);

        // If no driver in chosen group, try any available driver
        if (!driver) {
            driver = db.prepare(`
        SELECT dp.*, u.name as driver_name, u.email as driver_email, u.phone as driver_phone
        FROM driver_profiles dp
        JOIN users u ON u.id = dp.user_id
        WHERE dp.is_online = 1 AND dp.is_approved = 1
        ORDER BY ABS(dp.lat - ?) + ABS(dp.lng - ?) ASC
        LIMIT 1
      `).get(pickup_lat, pickup_lng);
        }

        if (!driver) {
            return res.status(404).json({ error: 'No drivers available at the moment. Please try again.' });
        }

        // Create ride
        const result = db.prepare(`
      INSERT INTO rides (rider_id, driver_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                         pickup_addr, dropoff_addr, status, fare, distance_km, duration_min,
                         algorithm_mode, matched_group)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'matched', ?, ?, ?, ?, ?)
    `).run(
            req.user.id, driver.user_id,
            pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
            pickup_addr || '', dropoff_addr || '',
            fare, +distance.toFixed(2), duration,
            algoState.mode, driver.driver_group
        );

        // Update algorithm state: increment N for the matched group, simulate reward
        const reward = simulateReward(driver.driver_group);
        const groupCol = driver.driver_group;
        db.prepare(`
      UPDATE algorithm_state SET
        total_rounds = total_rounds + 1,
        n_${groupCol} = n_${groupCol} + 1,
        s_${groupCol} = s_${groupCol} + ?
      WHERE id = 1
    `).run(reward);

        // Update driver total rides
        db.prepare('UPDATE driver_profiles SET total_rides = total_rides + 1 WHERE user_id = ?').run(driver.user_id);

        const ride = db.prepare('SELECT * FROM rides WHERE id = ?').get(result.lastInsertRowid);

        res.status(201).json({
            ride,
            driver: {
                name: driver.driver_name,
                phone: driver.driver_phone,
                vehicle: `${driver.vehicle_model} (${driver.vehicle_plate})`,
                rating: driver.rating,
                group: driver.driver_group,
                lat: driver.lat,
                lng: driver.lng,
            },
            algorithm: { mode: algoState.mode, group: driver.driver_group, reason, reward },
        });
    } catch (err) {
        console.error('Ride request error:', err);
        res.status(500).json({ error: 'Server error' });
    }
});

// ── GET /api/rides/active ─────────────────────────────────────────────────────
router.get('/active', authMiddleware, (req, res) => {
    const ride = db.prepare(`
    SELECT r.*, u.name as driver_name, u.phone as driver_phone,
           dp.vehicle_model, dp.vehicle_plate, dp.rating as driver_rating, dp.driver_group,
           dp.lat as driver_lat, dp.lng as driver_lng
    FROM rides r
    LEFT JOIN users u ON u.id = r.driver_id
    LEFT JOIN driver_profiles dp ON dp.user_id = r.driver_id
    WHERE (r.rider_id = ? OR r.driver_id = ?) AND r.status IN ('requested','matched','in_progress')
    ORDER BY r.created_at DESC LIMIT 1
  `).get(req.user.id, req.user.id);
    res.json({ ride: ride || null });
});

// ── PATCH /api/rides/:id/status ───────────────────────────────────────────────
router.patch('/:id/status', authMiddleware, (req, res) => {
    const { status, rating } = req.body;
    const validTransitions = {
        matched: ['in_progress', 'cancelled'],
        in_progress: ['completed', 'cancelled'],
        requested: ['matched', 'cancelled'],
    };
    const ride = db.prepare('SELECT * FROM rides WHERE id = ?').get(req.params.id);
    if (!ride) return res.status(404).json({ error: 'Ride not found' });
    if (!validTransitions[ride.status]?.includes(status)) {
        return res.status(400).json({ error: `Cannot transition from ${ride.status} to ${status}` });
    }
    const updates = { status };
    if (status === 'completed') {
        updates.completed_at = new Date().toISOString();
        if (rating) {
            updates.rating = rating;
            // Update driver satisfaction
            const dr = db.prepare('SELECT * FROM driver_profiles WHERE user_id = ?').get(ride.driver_id);
            if (dr) {
                const newRate = (dr.satisfaction_rate * dr.total_rides + (rating >= 4 ? 1 : 0)) / (dr.total_rides);
                db.prepare('UPDATE driver_profiles SET satisfaction_rate = ?, rating = ROUND((rating * (total_rides - 1) + ?) / total_rides, 2) WHERE user_id = ?')
                    .run(+newRate.toFixed(3), rating, ride.driver_id);
            }
        }
    }
    db.prepare(`UPDATE rides SET status = ?, completed_at = COALESCE(?, completed_at), rating = COALESCE(?, rating) WHERE id = ?`)
        .run(updates.status, updates.completed_at || null, updates.rating || null, req.params.id);

    const updated = db.prepare('SELECT * FROM rides WHERE id = ?').get(req.params.id);
    res.json({ ride: updated });
});

// ── GET /api/rides/history ────────────────────────────────────────────────────
router.get('/history', authMiddleware, (req, res) => {
    const rides = db.prepare(`
    SELECT r.*, u.name as driver_name, dp.driver_group, dp.vehicle_model
    FROM rides r
    LEFT JOIN users u ON u.id = r.driver_id
    LEFT JOIN driver_profiles dp ON dp.user_id = r.driver_id
    WHERE r.rider_id = ? OR r.driver_id = ?
    ORDER BY r.created_at DESC
    LIMIT 50
  `).all(req.user.id, req.user.id);
    res.json({ rides });
});

module.exports = router;
