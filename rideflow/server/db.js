const Database = require('better-sqlite3');
const path = require('path');

const db = new Database(path.join(__dirname, 'rideflow.db'));

// Enable WAL mode for better concurrency
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// ── Schema ────────────────────────────────────────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT    UNIQUE NOT NULL,
    password    TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    phone       TEXT    DEFAULT '',
    role        TEXT    NOT NULL CHECK(role IN ('rider','driver','admin')),
    avatar_url  TEXT    DEFAULT '',
    created_at  TEXT    DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS driver_profiles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER UNIQUE NOT NULL REFERENCES users(id),
    experience_years  REAL    DEFAULT 0,
    total_rides       INTEGER DEFAULT 0,
    satisfaction_rate REAL    DEFAULT 0,
    rating            REAL    DEFAULT 5.0,
    is_online         INTEGER DEFAULT 0,
    is_approved       INTEGER DEFAULT 0,
    driver_group      TEXT    NOT NULL CHECK(driver_group IN ('veteran','intermediate','newbie')),
    vehicle_model     TEXT    DEFAULT '',
    vehicle_plate     TEXT    DEFAULT '',
    lat               REAL    DEFAULT 28.6139,
    lng               REAL    DEFAULT 77.2090,
    created_at        TEXT    DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS rides (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    rider_id      INTEGER NOT NULL REFERENCES users(id),
    driver_id     INTEGER REFERENCES users(id),
    pickup_lat    REAL    NOT NULL,
    pickup_lng    REAL    NOT NULL,
    dropoff_lat   REAL    NOT NULL,
    dropoff_lng   REAL    NOT NULL,
    pickup_addr   TEXT    DEFAULT '',
    dropoff_addr  TEXT    DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'requested'
                  CHECK(status IN ('requested','matched','in_progress','completed','cancelled')),
    fare          REAL    DEFAULT 0,
    distance_km   REAL    DEFAULT 0,
    duration_min  REAL    DEFAULT 0,
    rating        INTEGER DEFAULT 0,
    algorithm_mode TEXT   DEFAULT 'fair',
    matched_group  TEXT   DEFAULT '',
    created_at    TEXT    DEFAULT (datetime('now')),
    completed_at  TEXT
  );

  CREATE TABLE IF NOT EXISTS driver_applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    name            TEXT    NOT NULL,
    experience_years REAL   DEFAULT 0,
    previous_rides  INTEGER DEFAULT 0,
    resume_text     TEXT    DEFAULT '',
    group_assigned  TEXT    CHECK(group_assigned IN ('veteran','intermediate','newbie')),
    status          TEXT    DEFAULT 'pending' CHECK(status IN ('pending','accepted','rejected')),
    ucb_score       REAL    DEFAULT 0,
    fairness_reason TEXT    DEFAULT '',
    created_at      TEXT    DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS algorithm_state (
    id              INTEGER PRIMARY KEY CHECK(id = 1),
    total_rounds    INTEGER DEFAULT 0,
    n_veteran       INTEGER DEFAULT 0,
    n_intermediate  INTEGER DEFAULT 0,
    n_newbie        INTEGER DEFAULT 0,
    s_veteran       INTEGER DEFAULT 0,
    s_intermediate  INTEGER DEFAULT 0,
    s_newbie        INTEGER DEFAULT 0,
    mode            TEXT    DEFAULT 'fair' CHECK(mode IN ('fair','greedy'))
  );

  CREATE TABLE IF NOT EXISTS sim_history (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    t     INTEGER NOT NULL,
    sat   REAL,
    share_veteran      REAL,
    share_intermediate REAL,
    share_newbie       REAL,
    mode  TEXT
  );
`);

// Initialize algorithm state if not present
const row = db.prepare('SELECT id FROM algorithm_state WHERE id = 1').get();
if (!row) {
    db.prepare(`INSERT INTO algorithm_state (id, total_rounds, n_veteran, n_intermediate, n_newbie, s_veteran, s_intermediate, s_newbie, mode)
              VALUES (1, 0, 0, 0, 0, 0, 0, 0, 'fair')`).run();
}

module.exports = db;
