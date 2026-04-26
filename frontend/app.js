/* ═══════════════════════════════════════════════════════════════════════════
   ACRS Frontend — Live SSE streaming + dashboard updates
   ═══════════════════════════════════════════════════════════════════════════ */

const API = '';  // Same origin

let isRunning = false;
let eventSource = null;

// ── Telemetry State ─────────────────────────────────────────────────────────

let latencyChart = null;
let cpuChart = null;
let memChart = null;
const MAX_DATAPOINTS = 20;

let telemetryData = {
    labels: [],
    latency: [],
    cpuApi: [],
    cpuDb: [],
    cpuCache: [],
    memApi: [],
    memDb: [],
    memCache: []
};

function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        plugins: {
            legend: {
                display: true,
                position: 'top',
                labels: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace", size: 10 } }
            },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.9)',
                titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                borderColor: 'rgba(71, 85, 105, 0.4)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                display: false,
                grid: { display: false }
            },
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(71, 85, 105, 0.2)' },
                ticks: { color: '#64748b', font: { family: "'JetBrains Mono', monospace", size: 10 } }
            }
        }
    };

    const ctxLatency = document.getElementById('latencyChart').getContext('2d');
    latencyChart = new Chart(ctxLatency, {
        type: 'line',
        data: {
            labels: telemetryData.labels,
            datasets: [{
                label: 'E2E Latency (ms)',
                data: telemetryData.latency,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                pointRadius: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: commonOptions
    });

    const ctxCpu = document.getElementById('cpuChart').getContext('2d');
    cpuChart = new Chart(ctxCpu, {
        type: 'line',
        data: {
            labels: telemetryData.labels,
            datasets: [
                {
                    label: 'API CPU',
                    data: telemetryData.cpuApi,
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'DB CPU',
                    data: telemetryData.cpuDb,
                    borderColor: '#ef4444',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Cache CPU',
                    data: telemetryData.cpuCache,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, max: 100 }
            }
        }
    });

    const ctxMem = document.getElementById('memChart').getContext('2d');
    memChart = new Chart(ctxMem, {
        type: 'line',
        data: {
            labels: telemetryData.labels,
            datasets: [
                {
                    label: 'API MEM',
                    data: telemetryData.memApi,
                    borderColor: '#0ea5e9', // lighter blue
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'DB MEM',
                    data: telemetryData.memDb,
                    borderColor: '#f43f5e', // lighter red
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Cache MEM',
                    data: telemetryData.memCache,
                    borderColor: '#fbbf24', // lighter yellow
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, max: 100 }
            }
        }
    });
}

function resetTelemetry() {
    telemetryData.labels.length = 0;
    telemetryData.latency.length = 0;
    telemetryData.cpuApi.length = 0;
    telemetryData.cpuDb.length = 0;
    telemetryData.cpuCache.length = 0;
    telemetryData.memApi.length = 0;
    telemetryData.memDb.length = 0;
    telemetryData.memCache.length = 0;
    if (latencyChart) latencyChart.update();
    if (cpuChart) cpuChart.update();
    if (memChart) memChart.update();
}

function updateTelemetry(state) {
    if (!latencyChart || !cpuChart || !memChart) return;

    const timeLabel = new Date().toLocaleTimeString([], { hour12: false });
    telemetryData.labels.push(timeLabel);

    telemetryData.latency.push(state.latency || 0);

    const svcs = state.services || {};
    telemetryData.cpuApi.push(svcs['api-service']?.cpu || 0);
    telemetryData.cpuDb.push(svcs['db-service']?.cpu || 0);
    telemetryData.cpuCache.push(svcs['cache-service']?.cpu || 0);

    telemetryData.memApi.push(svcs['api-service']?.memory || 0);
    telemetryData.memDb.push(svcs['db-service']?.memory || 0);
    telemetryData.memCache.push(svcs['cache-service']?.memory || 0);

    if (telemetryData.labels.length > MAX_DATAPOINTS) {
        telemetryData.labels.shift();
        telemetryData.latency.shift();
        telemetryData.cpuApi.shift();
        telemetryData.cpuDb.shift();
        telemetryData.cpuCache.shift();
        telemetryData.memApi.shift();
        telemetryData.memDb.shift();
        telemetryData.memCache.shift();
    }

    latencyChart.update();
    cpuChart.update();
    memChart.update();
}

// ── Phase Banner ────────────────────────────────────────────────────────────

const PHASE_CONFIG = {
    'NORMAL':    { text: 'SYSTEM NORMAL',           icon: '\u25CF', css: 'phase-normal' },
    'CHAOS':     { text: '\uD83D\uDD25 CHAOS INJECTED',  icon: '\u26A0', css: 'phase-chaos' },
    'DEGRADED':  { text: '\u26A0\uFE0F DEGRADED SYSTEM',  icon: '\u26A0', css: 'phase-degraded' },
    'FAILURE':   { text: '\uD83D\uDEA8 CRITICAL INCIDENT', icon: '\uD83D\uDD34', css: 'phase-failure' },
    'RECOVERY':  { text: '\u2705 SYSTEM RECOVERED',  icon: '\u2714', css: 'phase-recovery' },
};

function updatePhase(phase) {
    const banner = document.getElementById('phase-banner');
    const text = document.getElementById('phase-text');
    const icon = document.getElementById('phase-icon');
    const config = PHASE_CONFIG[phase] || PHASE_CONFIG['NORMAL'];

    banner.className = 'phase-banner ' + config.css;
    text.textContent = config.text;
    icon.textContent = config.icon;
}

// ── Latency Display ─────────────────────────────────────────────────────────

function updateLatency(value) {
    const el = document.getElementById('latency-value');
    el.textContent = value;
    el.className = 'latency-value';
    if (value > 1000) el.classList.add('critical');
    else if (value > 200) el.classList.add('warn');
}

function updateReward(value) {
    const el = document.getElementById('reward-value');
    el.textContent = (value >= 0 ? '+' : '') + value.toFixed(3);
    el.style.color = value >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
}

// ── Service Cards ───────────────────────────────────────────────────────────

function updateServices(services) {
    if (!services) return;

    const mapping = {
        'api-service': { card: 'card-api', status: 'status-api', cpu: 'cpu-api', mem: 'mem-api' },
        'db-service':  { card: 'card-db',  status: 'status-db',  cpu: 'cpu-db',  mem: 'mem-db' },
        'cache-service': { card: 'card-cache', status: 'status-cache', cpu: 'cpu-cache', mem: 'mem-cache' },
    };

    for (const [name, ids] of Object.entries(mapping)) {
        const svc = services[name];
        if (!svc) continue;

        const card = document.getElementById(ids.card);
        const statusEl = document.getElementById(ids.status);
        const cpuEl = document.getElementById(ids.cpu);
        const memEl = document.getElementById(ids.mem);

        const status = (svc.status || 'unknown').toLowerCase();
        statusEl.textContent = status.toUpperCase();

        // Color the status text
        const colors = { running: 'var(--accent-green)', degraded: 'var(--accent-yellow)', overloaded: 'var(--accent-red)', down: 'var(--accent-red)' };
        statusEl.style.color = colors[status] || 'var(--text-muted)';

        // Card border
        card.className = 'service-card status-' + status;

        cpuEl.textContent = svc.cpu !== undefined ? svc.cpu : '--';
        memEl.textContent = svc.memory !== undefined ? svc.memory : '--';
    }
}

// ── Fix Chain ───────────────────────────────────────────────────────────────

function updateFixChain(required, applied) {
    const container = document.getElementById('fix-chain');
    if (!required || required.length === 0) {
        container.innerHTML = '<span style="color: var(--text-muted); font-size: 11px;">No fix chain loaded</span>';
        return;
    }

    container.innerHTML = required.map((fix, i) => {
        const isApplied = applied && applied.includes(fix);
        const icon = isApplied ? '\u2714' : '\u25CB';
        const cls = isApplied ? 'applied' : 'pending';
        return `<div class="fix-item ${cls}">${icon} ${i + 1}. ${fix}</div>`;
    }).join('');
}

// ── Logs ────────────────────────────────────────────────────────────────────

function updateLogs(logs) {
    const container = document.getElementById('log-container');
    if (!logs || logs.length === 0) return;

    // Clear empty state
    const empty = container.querySelector('.log-empty');
    if (empty) empty.remove();

    container.innerHTML = logs.map(log => {
        let cls = 'log-line';
        const lower = log.toLowerCase();
        if (lower.includes('critical') || lower.includes('chaos')) cls += ' log-critical';
        else if (lower.includes('error')) cls += ' log-error';
        else if (lower.includes('warn')) cls += ' log-warn';
        else if (lower.includes('info')) cls += ' log-info';
        return `<div class="${cls}">${escapeHtml(log)}</div>`;
    }).join('');

    container.scrollTop = container.scrollHeight;
}

function appendLog(log) {
    const container = document.getElementById('log-container');
    const empty = container.querySelector('.log-empty');
    if (empty) empty.remove();

    const div = document.createElement('div');
    let cls = 'log-line';
    const lower = log.toLowerCase();
    if (lower.includes('critical') || lower.includes('chaos')) cls += ' log-critical';
    else if (lower.includes('error')) cls += ' log-error';
    else if (lower.includes('warn')) cls += ' log-warn';
    else if (lower.includes('info')) cls += ' log-info';
    div.className = cls;
    div.textContent = log;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ── Agent Trace ─────────────────────────────────────────────────────────────

function addTraceCard(step) {
    const container = document.getElementById('trace-container');
    const empty = container.querySelector('.trace-empty');
    if (empty) empty.remove();

    const isError = step.source === 'LLM_ERROR';
    const rewardCls = step.reward >= 0 ? 'positive' : 'negative';
    const rewardStr = (step.reward >= 0 ? '+' : '') + step.reward.toFixed(3);
    const totalStr = (step.total_reward >= 0 ? '+' : '') + step.total_reward.toFixed(3);

    const paramsStr = step.params && Object.keys(step.params).length > 0
        ? ' ' + JSON.stringify(step.params) : '';

    const conf = step.confidence !== undefined ? step.confidence : 0;
    const confCls = conf >= 0.8 ? 'positive' : conf >= 0.5 ? 'warning' : 'negative';

    const card = document.createElement('div');
    card.className = 'trace-card' + (isError ? ' trace-error' : '');
    card.innerHTML = `
        <div class="trace-header">
            <span class="trace-step">STEP ${step.step}</span>
            <div style="display:flex; gap:12px; align-items:center;">
                <span class="trace-confidence ${confCls}">CONF: ${conf.toFixed(2)}</span>
                <span class="trace-reward ${rewardCls}">${rewardStr} (${totalStr})</span>
            </div>
        </div>
        ${!isError ? `
        <div class="trace-thinking">
            <span class="label">HYPOTHESIS</span>
            ${escapeHtml(step.hypothesis || '')}
        </div>
        <div class="trace-action">\u25B6 ${escapeHtml(step.tool || '')}${escapeHtml(paramsStr)}</div>
        ` : `
        <div class="trace-thinking">
            <span class="label">ERROR</span>
            LLM failed to produce valid action. Penalty applied.
        </div>
        `}
        <div class="trace-result">${escapeHtml(step.result || '')}</div>
    `;
    container.appendChild(card);
    container.scrollTop = container.scrollHeight;
}

// ── Recovery Summary ────────────────────────────────────────────────────────

function showSummary(summary) {
    const el = document.getElementById('recovery-summary');
    el.classList.remove('hidden');

    const isResolved = summary.status === 'RESOLVED';
    el.className = 'recovery-summary' + (isResolved ? '' : ' failed');

    document.getElementById('summary-status').textContent =
        isResolved ? '\u2705 INCIDENT RESOLVED' : '\u274C INCIDENT UNRESOLVED';
    document.getElementById('summary-status').style.color =
        isResolved ? 'var(--accent-green)' : 'var(--accent-red)';

    document.getElementById('summary-scenario').textContent = summary.scenario || '--';
    document.getElementById('summary-signals').textContent = summary.signals_gathered || '0';
    document.getElementById('summary-fixes').textContent =
        (summary.fixes_applied && summary.fixes_applied.length > 0)
            ? summary.fixes_applied.join(' \u2192 ') : 'none';
    document.getElementById('summary-steps').textContent = summary.steps_taken || '0';
    document.getElementById('summary-latency').textContent = (summary.final_latency || 0) + 'ms';
    document.getElementById('summary-reward').textContent =
        (summary.total_reward >= 0 ? '+' : '') + (summary.total_reward || 0).toFixed(3);

    // Failure Reasoning
    let reasonContainer = document.getElementById('summary-failure-reason');
    if (!reasonContainer) {
        reasonContainer = document.createElement('div');
        reasonContainer.id = 'summary-failure-reason';
        reasonContainer.className = 'summary-reasoning hidden';
        document.querySelector('.summary-grid').after(reasonContainer);
    }

    if (!isResolved && summary.failure_reason) {
        reasonContainer.classList.remove('hidden');
        reasonContainer.innerHTML = `
            <div class="reason-block">
                <strong>Reason:</strong> ${escapeHtml(summary.failure_reason)}
            </div>
            <div class="suggestion-block">
                <strong>Suggested Improvement:</strong> ${escapeHtml(summary.suggested_improvement || '')}
            </div>
        `;
    } else {
        reasonContainer.classList.add('hidden');
    }
}

// ── State Update (from SSE state field) ─────────────────────────────────────

function updateFromState(state) {
    if (!state) return;
    updatePhase(state.phase || 'NORMAL');
    updateLatency(state.latency || 0);
    updateReward(state.total_reward || 0);
    updateServices(state.services || {});
    updateFixChain(state.required_fixes, state.applied_fixes);
    updateLogs(state.logs || []);
    updateTelemetry(state);

    const scenario = document.getElementById('scenario-name');
    if (state.scenario) scenario.textContent = state.scenario;
}

// ── Controls ────────────────────────────────────────────────────────────────

function setButtonsEnabled(running) {
    isRunning = running;
    document.getElementById('btn-run').disabled = running;
    document.getElementById('btn-reset').disabled = running;

    if (running) {
        document.getElementById('btn-run').innerHTML = '<span class="spinner"></span> Running...';
    } else {
        document.getElementById('btn-run').innerHTML = '<span class="btn-icon">\u25B6</span> Run Agent';
    }
}

async function resetSystem() {
    // Close any existing SSE
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    try {
        const res = await fetch(API + '/api/agent/reset', { method: 'POST' });
        const data = await res.json();

        // Clear UI
        document.getElementById('trace-container').innerHTML =
            '<div class="trace-empty">Click "Run Agent" to start autonomous recovery</div>';
        document.getElementById('recovery-summary').classList.add('hidden');

        // Update state
        updatePhase(data.phase || 'CHAOS');
        updateLatency(data.observation?.latency || 0);
        updateReward(0);
        updateServices(data.observation?.services || {});
        updateLogs(data.observation?.logs || []);
        document.getElementById('scenario-name').textContent = data.scenario || 'Unknown';

        // Reset fix chain — fetch full state
        const stateRes = await fetch(API + '/api/state');
        const stateData = await stateRes.json();
        updateFixChain(stateData.required_fixes, stateData.applied_fixes);

        resetTelemetry();
        if (data.observation) {
            updateTelemetry({ latency: data.observation.latency, services: data.observation.services });
        }

        setButtonsEnabled(false);
    } catch (e) {
        console.error('Reset failed:', e);
    }
}

async function runAgent() {
    if (isRunning) return;

    setButtonsEnabled(true);

    // Clear trace
    document.getElementById('trace-container').innerHTML = '';
    document.getElementById('recovery-summary').classList.add('hidden');

    // Connect SSE
    eventSource = new EventSource(API + '/api/agent/run');

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'init') {
                // Initial state
                updateFromState(data.state);
                document.getElementById('scenario-name').textContent = data.scenario || 'Unknown';
                return;
            }

            if (data.type === 'summary') {
                // Recovery summary
                showSummary(data);
                if (data.state) updateFromState(data.state);
                return;
            }

            if (data.type === 'done') {
                // Stream ended
                setButtonsEnabled(false);
                if (eventSource) { eventSource.close(); eventSource = null; }
                return;
            }

            // Regular step
            addTraceCard(data);
            if (data.state) updateFromState(data.state);

        } catch (e) {
            console.error('SSE parse error:', e);
        }
    };

    eventSource.onerror = function() {
        setButtonsEnabled(false);
        if (eventSource) { eventSource.close(); eventSource = null; }
    };
}

function toggleManualControls() {
    const isManual = document.querySelector('input[name="mode"][value="manual"]').checked;
    const manualDiv = document.getElementById('manual-controls');
    const runBtn = document.getElementById('btn-run');
    
    if (isManual) {
        manualDiv.classList.remove('hidden');
        runBtn.classList.add('hidden');
    } else {
        manualDiv.classList.add('hidden');
        runBtn.classList.remove('hidden');
    }
}

function toggleServiceSelect() {
    const tool = document.getElementById('manual-tool').value;
    const svcGroup = document.getElementById('service-select-group');
    if (tool === 'restart_service' || tool === 'scale_service') {
        svcGroup.classList.remove('hidden');
    } else {
        svcGroup.classList.add('hidden');
    }
}



async function executeManualStep() {
    const tool = document.getElementById('manual-tool').value;
    let params = {};
    if (tool === 'restart_service' || tool === 'scale_service') {
        params.service = document.getElementById('manual-service').value;
    }

    const actionType = ['get_db_metrics', 'get_network_latency', 'get_error_logs', 'get_cache_status', 'clear_db_connections']
        .includes(tool) ? 'tool_call' : 'system_action';

    try {
        const res = await fetch(API + '/api/agent/step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action_type: actionType, tool, params }),
        });
        const data = await res.json();

        addTraceCard({
            step: data.step,
            hypothesis: 'Manual step',
            reasoning: '',
            tool: tool,
            params: params,
            result: data.result || '',
            reward: data.reward || 0,
            total_reward: data.total_reward || 0,
            source: 'MANUAL',
        });

        if (data.observation) {
            updateServices(data.observation.services);
            updateLatency(data.observation.latency);
            updateLogs(data.observation.logs);
        }
        updateReward(data.total_reward || 0);
        updatePhase(data.phase || 'DEGRADED');

        // Refresh fix chain
        const stateRes = await fetch(API + '/api/state');
        const stateData = await stateRes.json();
        updateFixChain(stateData.required_fixes, stateData.applied_fixes);

        if (data.done) {
            showSummary({
                status: 'RESOLVED',
                scenario: stateData.scenario,
                signals_gathered: stateData.signals,
                fixes_applied: stateData.applied_fixes,
                steps_taken: stateData.step,
                final_latency: data.observation?.latency || 0,
                total_reward: data.total_reward,
            });
        }
    } catch (e) {
        console.error('Step failed:', e);
    }
}

// ── Utilities ───────────────────────────────────────────────────────────────

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Init ────────────────────────────────────────────────────────────────────

window.addEventListener('load', async () => {
    initCharts();
    try {
        const res = await fetch(API + '/api/state');
        const state = await res.json();
        if (state.initialized) {
            updateFromState(state);
        }
    } catch (e) {
        console.log('Server not connected yet');
    }
});
