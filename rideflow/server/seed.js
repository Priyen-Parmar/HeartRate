const bcrypt = require('bcryptjs');
const db = require('./db');
const { classifyDriverGroup } = require('./fairUcb');

async function seed() {
    console.log('🌱 Seeding RideFlow database...\n');

    // Clear existing data
    db.exec('DELETE FROM sim_history');
    db.exec('DELETE FROM rides');
    db.exec('DELETE FROM driver_applications');
    db.exec('DELETE FROM driver_profiles');
    db.exec('DELETE FROM users');
    db.exec(`UPDATE algorithm_state SET total_rounds=0, n_veteran=0, n_intermediate=0, n_newbie=0,
           s_veteran=0, s_intermediate=0, s_newbie=0, mode='fair' WHERE id=1`);

    const hash = await bcrypt.hash('password123', 10);

    // ── Admin user ──────────────────────────────────────────────────────────────
    db.prepare('INSERT INTO users (email, password, name, phone, role) VALUES (?,?,?,?,?)')
        .run('admin@rideflow.com', hash, 'Admin', '+91-9000000000', 'admin');
    console.log('  ✓ Admin: admin@rideflow.com / password123');

    // ── Rider accounts ──────────────────────────────────────────────────────────
    const riders = [
        { email: 'rider@rideflow.com', name: 'Priya Sharma', phone: '+91-9111111111' },
        { email: 'amit@rideflow.com', name: 'Amit Patel', phone: '+91-9222222222' },
        { email: 'sneha@rideflow.com', name: 'Sneha Verma', phone: '+91-9333333333' },
    ];
    for (const r of riders) {
        db.prepare('INSERT INTO users (email, password, name, phone, role) VALUES (?,?,?,?,?)')
            .run(r.email, hash, r.name, r.phone, 'rider');
    }
    console.log(`  ✓ ${riders.length} Riders created (password: password123)`);

    // ── Driver accounts ─────────────────────────────────────────────────────────
    const drivers = [
        // Veterans
        { email: 'raj@rideflow.com', name: 'Raj Kumar', phone: '+91-9444441111', exp: 5, rides: 2400, sat: 0.92, rating: 4.8, vehicle: 'Toyota Innova', plate: 'DL-01-AB-1234', group: 'veteran' },
        { email: 'meena@rideflow.com', name: 'Meena Devi', phone: '+91-9444442222', exp: 4, rides: 1800, sat: 0.88, rating: 4.7, vehicle: 'Maruti Dzire', plate: 'DL-02-CD-5678', group: 'veteran' },
        { email: 'karan@rideflow.com', name: 'Karan Singh', phone: '+91-9444443333', exp: 6, rides: 3200, sat: 0.95, rating: 4.9, vehicle: 'Honda City', plate: 'DL-03-EF-9012', group: 'veteran' },
        // Intermediates
        { email: 'anita@rideflow.com', name: 'Anita Nair', phone: '+91-9555551111', exp: 2, rides: 350, sat: 0.72, rating: 4.3, vehicle: 'Hyundai i20', plate: 'DL-04-GH-3456', group: 'intermediate' },
        { email: 'vijay@rideflow.com', name: 'Vijay Reddy', phone: '+91-9555552222', exp: 1.5, rides: 200, sat: 0.68, rating: 4.1, vehicle: 'Tata Nexon', plate: 'DL-05-IJ-7890', group: 'intermediate' },
        { email: 'pooja@rideflow.com', name: 'Pooja Gupta', phone: '+91-9555553333', exp: 2.5, rides: 420, sat: 0.75, rating: 4.4, vehicle: 'Maruti Baleno', plate: 'DL-06-KL-1234', group: 'intermediate' },
        // Newbies
        { email: 'arjun@rideflow.com', name: 'Arjun Das', phone: '+91-9666661111', exp: 0.2, rides: 15, sat: 0.40, rating: 3.8, vehicle: 'Maruti WagonR', plate: 'DL-07-MN-5678', group: 'newbie' },
        { email: 'fatima@rideflow.com', name: 'Fatima Khan', phone: '+91-9666662222', exp: 0, rides: 3, sat: 0.33, rating: 3.5, vehicle: 'Tata Tiago', plate: 'DL-08-OP-9012', group: 'newbie' },
        { email: 'deepak@rideflow.com', name: 'Deepak Yadav', phone: '+91-9666663333', exp: 0.5, rides: 45, sat: 0.48, rating: 4.0, vehicle: 'Hyundai Santro', plate: 'DL-09-QR-3456', group: 'newbie' },
    ];

    // Locations around Delhi
    const locations = [
        { lat: 28.6139, lng: 77.2090 }, // Connaught Place
        { lat: 28.6304, lng: 77.2177 }, // Karol Bagh
        { lat: 28.5921, lng: 77.2307 }, // Lodhi Colony
        { lat: 28.6353, lng: 77.2250 }, // Paharganj
        { lat: 28.6273, lng: 77.2190 }, // Rajiv Chowk
        { lat: 28.6448, lng: 77.2167 }, // New Delhi Station
        { lat: 28.6129, lng: 77.2295 }, // India Gate
        { lat: 28.5562, lng: 77.1000 }, // IGI Airport
        { lat: 28.6692, lng: 77.4538 }, // Noida
    ];

    let driverIndex = 0;
    for (const d of drivers) {
        const userResult = db.prepare('INSERT INTO users (email, password, name, phone, role) VALUES (?,?,?,?,?)')
            .run(d.email, hash, d.name, d.phone, 'driver');

        const loc = locations[driverIndex % locations.length];
        const jitterLat = loc.lat + (Math.random() - 0.5) * 0.015;
        const jitterLng = loc.lng + (Math.random() - 0.5) * 0.015;

        db.prepare(`
      INSERT INTO driver_profiles (user_id, experience_years, total_rides, satisfaction_rate, rating,
                                   is_online, is_approved, driver_group, vehicle_model, vehicle_plate, lat, lng)
      VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?)
    `).run(userResult.lastInsertRowid, d.exp, d.rides, d.sat, d.rating, d.group, d.vehicle, d.plate, jitterLat, jitterLng);

        driverIndex++;
    }
    console.log(`  ✓ ${drivers.length} Drivers seeded (3 Veterans, 3 Intermediates, 3 Newbies)`);
    console.log('    All passwords: password123\n');

    // ── Some sample completed rides ─────────────────────────────────────────────
    const riderIds = db.prepare("SELECT id FROM users WHERE role = 'rider'").all().map(r => r.id);
    const driverRows = db.prepare("SELECT dp.user_id, dp.driver_group FROM driver_profiles dp WHERE dp.is_approved = 1").all();

    for (let i = 0; i < 20; i++) {
        const riderId = riderIds[i % riderIds.length];
        const driverRow = driverRows[i % driverRows.length];
        const pickLoc = locations[Math.floor(Math.random() * locations.length)];
        const dropLoc = locations[Math.floor(Math.random() * locations.length)];
        const dist = 2 + Math.random() * 15;
        const fare = Math.round(50 + dist * 12 + 20);

        db.prepare(`
      INSERT INTO rides (rider_id, driver_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng,
                         status, fare, distance_km, duration_min, rating, algorithm_mode, matched_group, completed_at)
      VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, 'fair', ?, datetime('now', '-' || ? || ' hours'))
    `).run(
            riderId, driverRow.user_id,
            pickLoc.lat + (Math.random() - 0.5) * 0.01, pickLoc.lng + (Math.random() - 0.5) * 0.01,
            dropLoc.lat + (Math.random() - 0.5) * 0.01, dropLoc.lng + (Math.random() - 0.5) * 0.01,
            fare, +dist.toFixed(2), Math.round(dist * 3 + 5),
            Math.floor(Math.random() * 2 + 4),
            driverRow.driver_group,
            Math.floor(Math.random() * 48)
        );
    }
    console.log('  ✓ 20 sample completed rides created');
    console.log('\n🎉 Seed complete! Start the server with: npm start\n');
}

seed().catch(console.error);
