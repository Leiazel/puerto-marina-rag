// Frontend JavaScript for Club Atlético Puerto Marina RAG

const API_BASE = "";

// State
let currentAlpha = 0.5;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initControls();
    loadStats();
    loadPlayers();
    loadDocuments();
    loadStaff();
    loadLogs();
});

// Navigation
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
            
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            document.getElementById(targetId).classList.add("active");

            if (targetId === "tab-logs") loadLogs();
            if (targetId === "tab-docs") loadDocuments();
            if (targetId === "tab-players") loadPlayers();
        });
    });
}

// Controls
function initControls() {
    const alphaSlider = document.getElementById("alpha-slider");
    const alphaValueLabel = document.getElementById("alpha-val");

    if (alphaSlider) {
        alphaSlider.addEventListener("input", (e) => {
            currentAlpha = parseFloat(e.target.value);
            alphaValueLabel.textContent = `${Math.round(currentAlpha * 100)}% BM25 / ${Math.round((1 - currentAlpha) * 100)}% Vector`;
        });
    }

    const queryInput = document.getElementById("query-input");
    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") executeQuery();
        });
    }

    // Ingestion Form
    const ingestForm = document.getElementById("ingest-form");
    if (ingestForm) {
        ingestForm.addEventListener("submit", handleIngestSubmit);
    }
}

// Stats
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        if (data.stats) {
            document.getElementById("stat-docs").textContent = data.stats.documents_count;
            document.getElementById("stat-chunks").textContent = data.stats.chunks_count;
            document.getElementById("stat-players").textContent = data.stats.players_count;
            document.getElementById("stat-latency").textContent = `${data.stats.average_query_latency_ms} ms`;
        }
    } catch (e) {
        console.warn("Could not load stats:", e);
    }
}

// Query Execution
async function executeQuery(presetQuery = null) {
    const input = document.getElementById("query-input");
    const query = presetQuery || input.value.trim();
    if (!query) return;

    if (presetQuery) {
        input.value = presetQuery;
    }

    const deptFilter = document.getElementById("dept-filter").value || null;
    const topK = parseInt(document.getElementById("top-k-select").value) || 4;
    const submitBtn = document.getElementById("submit-btn");

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Consultando...';

    const resultCard = document.getElementById("result-card");
    resultCard.classList.remove("active");

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                department_filter: deptFilter,
                top_k: topK,
                alpha: currentAlpha
            })
        });

        const data = await res.json();
        renderResult(data);
        loadStats(); // refresh log count & latency
    } catch (err) {
        alert("Error al ejecutar la consulta: " + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>⚡</span> Consultar RAG';
    }
}

// Render Query Result
function renderResult(data) {
    const resultCard = document.getElementById("result-card");
    const answerEl = document.getElementById("answer-content");
    const telemetryEl = document.getElementById("telemetry-badges");
    const chunksEl = document.getElementById("chunks-container");

    // Telemetry
    telemetryEl.innerHTML = `
        <span class="badge badge-info">⏱️ Total: ${data.timings.total_ms} ms</span>
        <span class="badge badge-success">🔍 Búsqueda: ${data.timings.retrieval_ms} ms</span>
        <span class="badge badge-gold">🧠 Síntesis: ${data.timings.generation_ms} ms</span>
        <span class="badge badge-info">🎯 Confianza: ${Math.round(data.confidence * 100)}%</span>
    `;

    // Markdown Answer rendering
    answerEl.innerHTML = formatMarkdown(data.answer);

    // Chunks Accordion
    chunksEl.innerHTML = "";
    data.retrieved_chunks.forEach((ch, idx) => {
        const chunkDiv = document.createElement("div");
        chunkDiv.className = "chunk-card";
        
        const modalities = ch.modalities.map(m => `<span class="badge badge-gold">${m === "lexical" ? "BM25 FTS5" : "Vector Denso"}</span>`).join(" ");

        chunkDiv.innerHTML = `
            <div class="chunk-header" onclick="toggleChunk(${idx})">
                <div>
                    <strong>[#${idx + 1}] ${escapeHtml(ch.title)}</strong>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                        Área: <code>${ch.department_id}</code> | Ref: <code>${ch.metadata.doc_code || 'DOC'}</code>
                    </div>
                </div>
                <div class="chunk-scores">
                    ${modalities}
                    <span class="badge badge-info">RRF: ${ch.rrf_score.toFixed(4)}</span>
                    <span class="badge badge-success">CosSim: ${ch.cosine_sim.toFixed(3)}</span>
                </div>
            </div>
            <div class="chunk-body" id="chunk-body-${idx}" style="display: ${idx === 0 ? 'block' : 'none'};">
                ${ch.match_snippet ? `<p style="margin-bottom: 8px; color: #f59e0b;"><strong>Coincidencia Lexical:</strong> ${ch.match_snippet}</p>` : ''}
                ${escapeHtml(ch.content)}
            </div>
        `;
        chunksEl.appendChild(chunkDiv);
    });

    resultCard.classList.add("active");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function toggleChunk(idx) {
    const el = document.getElementById(`chunk-body-${idx}`);
    if (el) {
        el.style.display = el.style.display === "none" ? "block" : "none";
    }
}

// Markdown formatter helper
function formatMarkdown(mdText) {
    if (!mdText) return "";
    let html = mdText
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em>$1</em>')
        .replace(/`([^`]+)`/gim, '<code>$1</code>')
        .replace(/^\s*-\s+(.*$)/gim, '<li>$1</li>');

    // Wrap list items
    html = html.replace(/(<li>[\s\S]*?<\/li>)/gm, '<ul>$1</ul>');
    
    // Process markdown tables
    html = html.replace(/(\|.+\|\n)+/gm, (match) => {
        const rows = match.trim().split("\n");
        let tableHtml = "<table>";
        rows.forEach((row, i) => {
            if (row.includes("---")) return;
            const cells = row.split("|").filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
            tableHtml += "<tr>";
            cells.forEach(cell => {
                const tag = i === 0 ? "th" : "td";
                tableHtml += `<${tag}>${cell.trim()}</${tag}>`;
            });
            tableHtml += "</tr>";
        });
        tableHtml += "</table>";
        return tableHtml;
    });

    return html.replace(/\n\n/g, '<br><br>');
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Load Players Roster
async function loadPlayers() {
    const container = document.getElementById("players-grid");
    if (!container) return;
    try {
        const res = await fetch(`${API_BASE}/api/players`);
        const players = await res.json();
        container.innerHTML = "";
        players.forEach(p => {
            const card = document.createElement("div");
            card.className = "player-card";
            card.innerHTML = `
                <div class="player-dorsal">#${p.dorsal}</div>
                <div class="player-name">${p.name} ${p.nickname ? `"${p.nickname}"` : ''}</div>
                <div class="player-pos">${p.position}</div>
                <div class="player-meta">
                    <span>${p.age} años</span>
                    <span>•</span>
                    <span>${p.height_m} m</span>
                    <span>•</span>
                    <span>${p.weight_kg} kg</span>
                </div>
                <div class="player-notes">${p.notes || 'Sin observaciones'}</div>
                <button class="player-rag-btn" onclick="queryPlayer('${p.name}')">
                    🔍 Consultar Ficha Médica / GPS
                </button>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading players:", e);
    }
}

function queryPlayer(playerName) {
    document.querySelector('.nav-btn[data-tab="tab-query"]').click();
    document.getElementById("query-input").value = `¿Cuál es el estado físico, médico o informe sobre ${playerName}?`;
    executeQuery();
}

// Load Documents
async function loadDocuments() {
    const tableBody = document.getElementById("docs-table-body");
    if (!tableBody) return;
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        const docs = await res.json();
        tableBody.innerHTML = "";
        docs.forEach(d => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><code>${d.doc_code}</code></td>
                <td><strong>${escapeHtml(d.title)}</strong></td>
                <td><span class="badge badge-info">${d.department_name}</span></td>
                <td>${d.author}</td>
                <td>${d.date}</td>
                <td><span class="badge badge-gold">${d.chunk_count} chunks</span></td>
            `;
            tableBody.appendChild(row);
        });
    } catch (e) {
        console.error("Error loading documents:", e);
    }
}

// Load Staff
async function loadStaff() {
    const container = document.getElementById("staff-grid");
    if (!container) return;
    try {
        const res = await fetch(`${API_BASE}/api/staff`);
        const staff = await res.json();
        container.innerHTML = "";
        staff.forEach(s => {
            const card = document.createElement("div");
            card.className = "player-card";
            card.innerHTML = `
                <div class="player-name">${s.name}</div>
                <div class="player-pos">${s.role}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">
                    Área: <strong>${s.department_name}</strong>
                </div>
                <div style="font-size: 0.78rem; color: #cbd5e1;">
                    📧 ${s.email}<br>
                    📞 ${s.phone}
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading staff:", e);
    }
}

// Load Logs
async function loadLogs() {
    const tableBody = document.getElementById("logs-table-body");
    if (!tableBody) return;
    try {
        const res = await fetch(`${API_BASE}/api/logs`);
        const logs = await res.json();
        tableBody.innerHTML = "";
        logs.forEach(l => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${l.created_at}</td>
                <td><strong>${escapeHtml(l.query_text)}</strong></td>
                <td><span class="badge badge-info">${l.department_filter || 'Todos'}</span></td>
                <td>${l.retrieved_chunks_count}</td>
                <td><span class="badge badge-success">${l.total_ms} ms</span></td>
                <td><code>${(l.top_similarity || 0).toFixed(3)}</code></td>
            `;
            tableBody.appendChild(row);
        });
    } catch (e) {
        console.error("Error loading logs:", e);
    }
}

// Handle Ingest Form Submit
async function handleIngestSubmit(e) {
    e.preventDefault();
    const btn = document.getElementById("ingest-submit-btn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Indexando...';

    const payload = {
        doc_code: document.getElementById("ingest-code").value.trim(),
        title: document.getElementById("ingest-title").value.trim(),
        department_id: document.getElementById("ingest-dept").value,
        author: document.getElementById("ingest-author").value.trim(),
        date: document.getElementById("ingest-date").value.trim(),
        tags: document.getElementById("ingest-tags").value.split(",").map(t => t.trim()).filter(Boolean),
        content: document.getElementById("ingest-content").value.trim()
    };

    try {
        const res = await fetch(`${API_BASE}/api/documents`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        alert(`Éxito: ${data.message} (${data.chunks_created} fragmentos generados con FastEmbed)`);
        document.getElementById("ingest-form").reset();
        loadStats();
        loadDocuments();
    } catch (err) {
        alert("Error al indexar: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "📥 Indexar Documento en Base RAG";
    }
}
