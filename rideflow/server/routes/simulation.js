const express = require('express');
const db = require('../db');
const { authMiddleware } = require('../auth');
const { runSimulationStep, GROUPS, getUcbScores } = require('../fairUcb');

const router = express.Router();

// ── GET /api/sim/state ────────────────────────────────────────────────────────
router.get('/state', authMiddleware, (req, res) => {
    const state = db.prepare('SELECT * FROM algorithm_state WHERE id = 1').get();
    const N = [state.n_veteran, state.n_intermediate, state.n_newbie];
    const S = [state.s_veteran, state.s_intermediate, state.s_newbie];
    const total = N[0] + N[1] + N[2];
    const totalS = S[0] + S[1] + S[2];
    const t = state.total_rounds + 1;
    const ucbScores = getUcbScores(t, N, S);

    const groups = GROUPS.map((g, i) => ({
        ...g,
        N: N[i],
        S: S[i],
        share: total > 0 ? +(N[i] / total * 100).toFixed(1) : 0,
        satRate: N[i] > 0 ? +(S[i] / N[i] * 100).toFixed(1) : 0,
        violation: g.quota * (state.total_rounds) - N[i],
        ucbScore: ucbScores[i] === Infinity ? '∞' : ucbScores[i].toFixed(3),
        isStarved: g.quota * (state.total_rounds) - N[i] > 0,
    }));

    res.json({
        state: {
            t: state.total_rounds,
            mode: state.mode,
            N, S,
            overallSat: total > 0 ? +(totalS / total * 100).toFixed(1) : 0,
            total,
        },
        groups,
    });
});

// ── POST /api/sim/step ────────────────────────────────────────────────────────
router.post('/step', authMiddleware, (req, res) => {
    const { count = 10, mode } = req.body;
    const steps = Math.min(Math.max(1, count), 100);
    let state = db.prepare('SELECT * FROM algorithm_state WHERE id = 1').get();
    const simMode = mode || state.mode;
    const results = [];

    for (let i = 0; i < steps; i++) {
        const { newState, stats } = runSimulationStep(state, simMode);
        state = { ...state, ...newState };
        results.push(stats);

        // Save history point every step
        db.prepare(`
      INSERT INTO sim_history (t, sat, share_veteran, share_intermediate, share_newbie, mode)
      VALUES (?, ?, ?, ?, ?, ?)
    `).run(stats.t, stats.sat, stats.share_veteran, stats.share_intermediate, stats.share_newbie, simMode);
    }

    // Update algorithm state
    db.prepare(`
    UPDATE algorithm_state SET
      total_rounds = ?, n_veteran = ?, n_intermediate = ?, n_newbie = ?,
      s_veteran = ?, s_intermediate = ?, s_newbie = ?, mode = ?
    WHERE id = 1
  `).run(
        state.total_rounds, state.n_veteran, state.n_intermediate, state.n_newbie,
        state.s_veteran, state.s_intermediate, state.s_newbie, simMode
    );

    res.json({ steps: results, state });
});

// ── POST /api/sim/reset ───────────────────────────────────────────────────────
router.post('/reset', authMiddleware, (req, res) => {
    db.prepare(`
    UPDATE algorithm_state SET
      total_rounds = 0, n_veteran = 0, n_intermediate = 0, n_newbie = 0,
      s_veteran = 0, s_intermediate = 0, s_newbie = 0, mode = 'fair'
    WHERE id = 1
  `).run();
    db.prepare('DELETE FROM sim_history').run();
    res.json({ success: true });
});

// ── GET /api/sim/history ──────────────────────────────────────────────────────
router.get('/history', authMiddleware, (req, res) => {
    const limit = Math.min(parseInt(req.query.limit) || 300, 1000);
    const history = db.prepare('SELECT * FROM sim_history ORDER BY t DESC LIMIT ?').all(limit);
    res.json({ history: history.reverse() });
});

// ── PATCH /api/sim/mode ───────────────────────────────────────────────────────
router.patch('/mode', authMiddleware, (req, res) => {
    const { mode } = req.body;
    if (!['fair', 'greedy'].includes(mode)) return res.status(400).json({ error: 'Invalid mode' });
    db.prepare('UPDATE algorithm_state SET mode = ? WHERE id = 1').run(mode);
    res.json({ mode });
});

module.exports = router;
