// ══════════════════════════════════════════════════════════════════════════════
// FAIR-LEARN Algorithm (Patil, Ghalme, Nair, Narahari — IISc, 2020)
// FAIR-MAB meta-algorithm with UCB1 as the black-box bandit learner.
// Three arms = three driver groups in RideFlow.
// ══════════════════════════════════════════════════════════════════════════════

const GROUPS = [
    { id: 'veteran', index: 0, name: 'Veterans', trueRate: 0.90, quota: 0.00, color: '#10b981' },
    { id: 'intermediate', index: 1, name: 'Intermediates', trueRate: 0.70, quota: 0.15, color: '#f59e0b' },
    { id: 'newbie', index: 2, name: 'Newbies', trueRate: 0.45, quota: 0.25, color: '#06b6d4' },
];

const GROUP_INDEX = { veteran: 0, intermediate: 1, newbie: 2 };

// ── Classify a driver into a group based on their profile ─────────────────────
function classifyDriverGroup(experienceYears, previousRides, satisfactionRate) {
    // Veterans: 3+ years experience OR 500+ rides AND 80%+ satisfaction
    if ((experienceYears >= 3 || previousRides >= 500) && satisfactionRate >= 0.8) {
        return 'veteran';
    }
    // Intermediates: 1-3 years OR 100-500 rides
    if ((experienceYears >= 1 || previousRides >= 100) && satisfactionRate >= 0.5) {
        return 'intermediate';
    }
    // Everyone else is a newbie
    return 'newbie';
}

// ── UCB1 arm selection ────────────────────────────────────────────────────────
function ucb1Pick(t, N, S) {
    // Play any arm with 0 pulls first
    const unplayed = N.findIndex(n => n === 0);
    if (unplayed !== -1) return unplayed;
    // UCB1: argmax_i [ S_i/N_i + sqrt(2·ln(t) / N_i) ]
    const scores = N.map((n, i) => S[i] / n + Math.sqrt((2 * Math.log(t)) / n));
    return scores.indexOf(Math.max(...scores));
}

// ── Get UCB scores for all arms ───────────────────────────────────────────────
function getUcbScores(t, N, S) {
    return N.map((n, i) => {
        if (n === 0) return Infinity;
        return S[i] / n + Math.sqrt((2 * Math.log(Math.max(t, 2))) / n);
    });
}

// ── FAIR-LEARN: select which group should get the next ride ───────────────────
function fairLearnSelect(state, mode = 'fair') {
    const t = state.total_rounds + 1;
    const N = [state.n_veteran, state.n_intermediate, state.n_newbie];
    const S = [state.s_veteran, state.s_intermediate, state.s_newbie];
    let chosen;
    let reason = '';

    if (mode === 'fair') {
        // Violation: V_i = r_i·(t−1) − N_{i,t−1}
        const violations = GROUPS.map((g, i) => ({ i, v: g.quota * (t - 1) - N[i], group: g.id }));
        // Starved set A(t) = { i : V_i > α }, with α = 0
        const starved = violations.filter(x => x.v > 0);
        if (starved.length > 0) {
            // Fairness override: pull arm with maximum violation
            const pick = starved.reduce((a, b) => a.v > b.v ? a : b);
            chosen = pick.i;
            reason = `FAIRNESS OVERRIDE — Group ${GROUPS[chosen].name} starved (V=${pick.v.toFixed(1)})`;
        } else {
            chosen = ucb1Pick(t, N, S);
            reason = `UCB1 selected Group ${GROUPS[chosen].name} (score=${getUcbScores(t, N, S)[chosen].toFixed(3)})`;
        }
    } else {
        // Pure Greedy UCB1
        chosen = ucb1Pick(t, N, S);
        reason = `GREEDY UCB1 selected Group ${GROUPS[chosen].name}`;
    }

    return { chosenIndex: chosen, chosenGroup: GROUPS[chosen].id, reason };
}

// ── Simulate a reward (Bernoulli draw from true satisfaction rate) ─────────────
function simulateReward(groupId) {
    const group = GROUPS.find(g => g.id === groupId);
    return Math.random() < group.trueRate ? 1 : 0;
}

// ── Run a single simulation step ──────────────────────────────────────────────
function runSimulationStep(state, mode) {
    const { chosenIndex, chosenGroup, reason } = fairLearnSelect(state, mode);
    const reward = simulateReward(chosenGroup);

    const N = [state.n_veteran, state.n_intermediate, state.n_newbie];
    const S = [state.s_veteran, state.s_intermediate, state.s_newbie];
    N[chosenIndex]++;
    S[chosenIndex] += reward;

    const newState = {
        total_rounds: state.total_rounds + 1,
        n_veteran: N[0],
        n_intermediate: N[1],
        n_newbie: N[2],
        s_veteran: S[0],
        s_intermediate: S[1],
        s_newbie: S[2],
        mode,
    };

    const total = N[0] + N[1] + N[2];
    const totalS = S[0] + S[1] + S[2];
    const stats = {
        t: newState.total_rounds,
        sat: total > 0 ? +(totalS / total * 100).toFixed(2) : 0,
        share_veteran: total > 0 ? +(N[0] / total * 100).toFixed(1) : 0,
        share_intermediate: total > 0 ? +(N[1] / total * 100).toFixed(1) : 0,
        share_newbie: total > 0 ? +(N[2] / total * 100).toFixed(1) : 0,
        chosenGroup,
        reward,
        reason,
        mode,
    };

    return { newState, stats };
}

// ── Evaluate a driver application using Fair UCB ──────────────────────────────
function evaluateApplication(application, state, mode = 'fair') {
    const group = classifyDriverGroup(
        application.experience_years || 0,
        application.previous_rides || 0,
        0.5 // default satisfaction for new applicants
    );

    const groupIdx = GROUP_INDEX[group];
    const t = state.total_rounds + 1;
    const N = [state.n_veteran, state.n_intermediate, state.n_newbie];
    const S = [state.s_veteran, state.s_intermediate, state.s_newbie];
    const ucbScores = getUcbScores(t, N, S);
    const ucbScore = ucbScores[groupIdx] === Infinity ? 999 : +ucbScores[groupIdx].toFixed(3);

    // In fair mode, check if this group is starved
    let shouldAccept = true;
    let fairnessReason = '';

    if (mode === 'fair') {
        const violation = GROUPS[groupIdx].quota * (t - 1) - N[groupIdx];
        if (violation > 0) {
            shouldAccept = true;
            fairnessReason = `ACCEPTED: Group ${GROUPS[groupIdx].name} is starved (V=${violation.toFixed(1)}). Fairness quota requires more drivers in this group.`;
        } else {
            // Accept if UCB score is reasonable or the group is under-represented
            const totalDrivers = N[0] + N[1] + N[2];
            const groupShare = totalDrivers > 0 ? N[groupIdx] / totalDrivers : 0;
            if (groupShare < GROUPS[groupIdx].quota + 0.1 || ucbScore > 0.5) {
                shouldAccept = true;
                fairnessReason = `ACCEPTED: UCB score ${ucbScore} is favorable. Group share is ${(groupShare * 100).toFixed(1)}%.`;
            } else {
                shouldAccept = true; // We generally accept drivers, just classify them
                fairnessReason = `ACCEPTED: Classified as ${GROUPS[groupIdx].name}. Fair allocation will prioritize underrepresented groups.`;
            }
        }
    } else {
        fairnessReason = `ACCEPTED: Greedy mode — classified as ${GROUPS[groupIdx].name} with UCB score ${ucbScore}.`;
    }

    return {
        group,
        ucbScore,
        shouldAccept,
        fairnessReason,
    };
}

module.exports = {
    GROUPS,
    GROUP_INDEX,
    classifyDriverGroup,
    fairLearnSelect,
    simulateReward,
    runSimulationStep,
    evaluateApplication,
    getUcbScores,
};
