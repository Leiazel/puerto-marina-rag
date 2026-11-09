// Frontend JavaScript for Club Atlético Puerto Marina RAG (v3.0)
// Complete with Multi-Format Upload, Cascading Deletion, and XSS-Hardened Rendering

const API_BASE = "";
let currentAlpha = 0.5;
let uploadQueue = [];

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initControls();
    initUploadZone();
    loadStats();
    loadPlayers();
    loadDocuments();
    loadStaff();
    loadLogs();
});

function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const pane = document.getElementById(targetId);
            if (pane) pane.classList.add("active");

            if (targetId === "tab-logs") loadLogs();
            if (targetId === "tab-docs") loadDocuments();
            if (targetId === "tab-players") loadPlayers();
            if (targetId === "tab-staff") loadStaff();
        });
    });
}

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

    const ingestForm = document.getElementById("ingest-form");
    if (ingestForm) {
        ingestForm.addEventListener("submit", handleIngestSubmit);
    }
}

function initUploadZone() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const uploadForm = document.getElementById("upload-form");

    if (!dropzone || !fileInput || !uploadForm) return;

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            addFilesToQueue(Array.from(dt.files));
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            addFilesToQueue(Array.from(e.target.files));
            fileInput.value = "";
        }
    });

    uploadForm.addEventListener("submit", handleUploadSubmit);
}

function addFilesToQueue(newFiles) {
    const allowedExts = [".pdf", ".docx", ".doc", ".md", ".txt"];
    const maxSizeBytes = 25 * 1024 * 1024;

    newFiles.forEach(f => {
        const ext = "." + f.name.split(".").pop().toLowerCase();
        if (!allowedExts.includes(ext)) {
            alert(`El archivo "${f.name}" tiene una extensión no admitida (${ext}). Formatos soportados: PDF, DOCX, DOC, MD, TXT.`);
            return;
        }
        if (f.size > maxSizeBytes) {
            alert(`El archivo "${f.name}" supera el límite de 25 MB (${(f.size / (1024 * 1024)).toFixed(1)} MB).`);
            return;
        }
        // Avoid duplicate by name and size
        if (!uploadQueue.some(item => item.name === f.name && item.size === f.size)) {
            uploadQueue.push(f);
        }
    });

    renderFileQueue();
}

function renderFileQueue() {
    const queueContainer = document.getElementById("file-queue");
    const submitBtn = document.getElementById("upload-submit-btn");

    if (!queueContainer) return;

    if (uploadQueue.length === 0) {
        queueContainer.style.display = "none";
        queueContainer.innerHTML = "";
        if (submitBtn) submitBtn.disabled = true;
        return;
    }

    queueContainer.style.display = "flex";
    queueContainer.innerHTML = "";

    uploadQueue.forEach((file, idx) => {
        const ext = file.name.split(".").pop().toLowerCase();
        const sizeKb = (file.size / 1024).toFixed(1);
        const itemDiv = document.createElement("div");
        itemDiv.className = "file-queue-item";
        itemDiv.innerHTML = `
            <div class="file-queue-left">
                <span class="badge badge-gold">${escapeHtml(ext.toUpperCase())}</span>
                <span><strong>${escapeHtml(file.name)}</strong> <small style="color: #94a3b8;">(${sizeKb} KB)</small></span>
            </div>
            <button type="button" class="file-remove-btn" onclick="removeFileFromQueue(${idx})" title="Quitar archivo">✖</button>
        `;
        queueContainer.appendChild(itemDiv);
    });

    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `📤 Subir e Indexar ${uploadQueue.length} ${uploadQueue.length === 1 ? 'Archivo' : 'Archivos'} en RAG`;
    }
}

function removeFileFromQueue(idx) {
    uploadQueue.splice(idx, 1);
    renderFileQueue();
}

async function handleUploadSubmit(e) {
    e.preventDefault();
    if (uploadQueue.length === 0) return;

    const submitBtn = document.getElementById("upload-submit-btn");
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Procesando e indexando documentos...';

    const formData = new FormData();
    uploadQueue.forEach(f => formData.append("files", f));
    formData.append("department_id", document.getElementById("upload-dept").value);
    formData.append("author", document.getElementById("upload-author").value.trim() || "Staff Técnico");
    formData.append("tags", document.getElementById("upload-tags").value.trim());

    const resultsContainer = document.getElementById("upload-results");
    resultsContainer.style.display = "none";
    resultsContainer.innerHTML = "";

    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        renderUploadResults(data);

        // Clear queue on completion
        uploadQueue = [];
        renderFileQueue();
        loadStats();
        loadDocuments();
    } catch (err) {
        alert("Error durante la carga: " + err.message);
    } finally {
        submitBtn.disabled = uploadQueue.length === 0;
        submitBtn.innerHTML = "📤 Subir e Indexar Archivos en RAG";
    }
}

function renderUploadResults(data) {
    const container = document.getElementById("upload-results");
    if (!container) return;

    container.style.display = "grid";
    container.innerHTML = "";

    if (!data.details || data.details.length === 0) {
        container.innerHTML = "<p>No se recibieron detalles de procesamiento.</p>";
        return;
    }

    data.details.forEach(item => {
        const card = document.createElement("div");
        card.className = `upload-result-card ${item.status === 'success' ? 'success' : 'error'}`;

        if (item.status === "success") {
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <strong>📄 ${escapeHtml(item.filename)}</strong>
                    <span class="badge badge-success">✓ Indexado</span>
                </div>
                <div style="font-size: 0.8rem; color: #cbd5e1; margin-bottom: 0.5rem;">
                    Código RAG: <code>${escapeHtml(item.doc_code)}</code><br>
                    Fragmentos generados: <strong>${item.chunks_created}</strong><br>
                    ${item.size_bytes ? `Tamaño: ${(item.size_bytes / 1024).toFixed(1)} KB` : ''}
                </div>
                <button class="player-rag-btn" style="margin-top: 0.4rem; padding: 0.4rem 0.6rem; font-size: 0.78rem;" onclick="queryUploadedDoc('${escapeHtml(item.doc_code)}')">
                    🔍 Consultar este documento
                </button>
            `;
        } else {
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <strong>⚠️ ${escapeHtml(item.filename)}</strong>
                    <span class="badge badge-danger" style="background:#ef4444; color:#fff;">Error</span>
                </div>
                <div style="font-size: 0.8rem; color: #f87171;">
                    ${escapeHtml(item.message)}
                </div>
            `;
        }
        container.appendChild(card);
    });
}

function queryUploadedDoc(docCode) {
    document.querySelector('.nav-btn[data-tab="tab-query"]').click();
    document.getElementById("query-input").value = `¿Qué información clave contiene el documento con código ${docCode}?`;
    executeQuery();
}

async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        if (data.stats) {
            document.getElementById("stat-docs").textContent = data.stats.documents_count;
            document.getElementById("stat-chunks").textContent = data.stats.chunks_count;
            document.getElementById("stat-players").textContent = data.stats.players_count;
            document.getElementById("stat-latency").textContent = `${data.stats.average_query_latency_ms} ms`;
            const modeEl = document.getElementById("stat-mode");
            if (modeEl && data.stats.synthesis_mode) {
                modeEl.textContent = data.stats.synthesis_mode;
            }
        }
    } catch (e) {
        console.warn("Could not load stats:", e);
    }
}

async function executeQuery(presetQuery = null) {
    const input = document.getElementById("query-input");
    const query = presetQuery || input.value.trim();
    if (!query) return;

    if (presetQuery) input.value = presetQuery;

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
        loadStats();
    } catch (err) {
        alert("Error al ejecutar la consulta: " + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>⚡</span> Consultar RAG';
    }
}

function renderResult(data) {
    const resultCard = document.getElementById("result-card");
    const answerEl = document.getElementById("answer-content");
    const telemetryEl = document.getElementById("telemetry-badges");
    const chunksEl = document.getElementById("chunks-container");

    telemetryEl.innerHTML = `
        <span class="badge badge-info">⏱️ Total: ${data.timings.total_ms} ms</span>
        <span class="badge badge-success">🔍 Búsqueda: ${data.timings.retrieval_ms} ms</span>
        <span class="badge badge-gold">🧠 Motor: ${escapeHtml(data.model_used)} (${data.timings.generation_ms} ms)</span>
        <span class="badge badge-info">🎯 Confianza: ${Math.round(data.confidence * 100)}%</span>
    `;

    answerEl.innerHTML = formatMarkdown(data.answer);

    chunksEl.innerHTML = "";
    data.retrieved_chunks.forEach((ch, idx) => {
        const chunkDiv = document.createElement("div");
        chunkDiv.className = "chunk-card";

        const modalities = (ch.modalities || [])
            .map(m => `<span class="badge badge-gold">${m === "lexical" ? "BM25 FTS5" : "Vector Denso"}</span>`)
            .join(" ");

        const snippetHtml = formatSnippet(ch.match_snippet);

        chunkDiv.innerHTML = `
            <div class="chunk-header" onclick="toggleChunk(${idx})">
                <div>
                    <strong>[#${idx + 1}] ${escapeHtml(ch.title)}</strong>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                        Área: <code>${escapeHtml(ch.department_id)}</code> | Ref: <code>${escapeHtml((ch.metadata && ch.metadata.doc_code) || 'DOC')}</code>
                    </div>
                </div>
                <div class="chunk-scores">
                    ${modalities}
                    <span class="badge badge-info">RRF: ${ch.rrf_score.toFixed(4)}</span>
                    <span class="badge badge-success">CosSim: ${ch.cosine_sim.toFixed(3)}</span>
                </div>
            </div>
            <div class="chunk-body" id="chunk-body-${idx}" style="display: ${idx === 0 ? 'block' : 'none'};">
                ${snippetHtml ? `<p style="margin-bottom: 8px; color: #f59e0b;"><strong>Coincidencia Léxica:</strong> ${snippetHtml}</p>` : ''}
                <div style="white-space: pre-wrap; font-family: inherit;">${escapeHtml(ch.content)}</div>
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

function formatSnippet(snippet) {
    if (!snippet) return "";
    const escaped = escapeHtml(snippet);
    return escaped.replace(/«([^»]+)»/g, '<mark class="fts-match">$1</mark>');
}

function formatMarkdown(mdText) {
    if (!mdText) return "";
    let html = mdText
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/^---+$/gim, '<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.12); margin: 1rem 0;">')
        .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h3>$1</h3>')
        .replace(/^\s*[-*]\s+(.*$)/gim, '<li>$1</li>')
        .replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
        .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/gim, '<em>$1</em>')
        .replace(/`([^`]+)`/gim, '<code>$1</code>')
        .replace(/\[REF:\s*([^\]]+)\]/gim, '<span class="badge badge-gold" style="font-size:0.7rem;">REF: $1</span>');

    html = html.replace(/(?:<li>.*?<\/li>\s*)+/gm, (match) => `<ul>${match}</ul>`);

    html = html.replace(/(\|.+\|\n?)+/gm, (match) => {
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
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

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
            const safeName = escapeHtml(p.name);
            card.innerHTML = `
                <div class="player-dorsal">#${p.dorsal}</div>
                <div class="player-name">${safeName} ${p.nickname ? `"${escapeHtml(p.nickname)}"` : ''}</div>
                <div class="player-pos">${escapeHtml(p.position)}</div>
                <div class="player-meta">
                    <span>${p.age} años</span>
                    <span>•</span>
                    <span>${p.height_m} m</span>
                    <span>•</span>
                    <span>${p.weight_kg} kg</span>
                </div>
                <div class="player-notes">${escapeHtml(p.notes || 'Sin observaciones')}</div>
                <button class="player-rag-btn" onclick="queryPlayer('${safeName}')">
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
    document.getElementById("query-input").value = `¿Cuál es el estado físico, médico, peso y altura de ${playerName}?`;
    executeQuery();
}

async function loadDocuments() {
    const tableBody = document.getElementById("docs-table-body");
    if (!tableBody) return;
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        const docs = await res.json();
        tableBody.innerHTML = "";
        docs.forEach(d => {
            const row = document.createElement("tr");
            const fmt = (d.source_type || 'md').toUpperCase();
            const safeTitle = escapeHtml(d.title);
            row.innerHTML = `
                <td><code>${escapeHtml(d.doc_code)}</code></td>
                <td><strong>${safeTitle}</strong></td>
                <td><span class="badge badge-info">${escapeHtml(d.department_name)}</span></td>
                <td>${escapeHtml(d.author)}</td>
                <td><span class="badge badge-gold" style="font-size:0.75rem;">${escapeHtml(fmt)}</span></td>
                <td>${escapeHtml(d.date || d.created_at || '')}</td>
                <td><span class="badge badge-info">${d.chunk_count} chunks</span></td>
                <td>
                    <button class="btn-delete" onclick="confirmDeleteDocument(${d.id}, '${safeTitle}')" title="Eliminar documento e índices RAG">
                        🗑️ Eliminar
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });
    } catch (e) {
        console.error("Error loading documents:", e);
    }
}

async function confirmDeleteDocument(docId, docTitle) {
    if (!confirm(`¿Estás seguro de que deseas eliminar permanentemente el documento "${docTitle}" y todos sus fragmentos indexados de la base de datos RAG?`)) {
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/api/documents/${docId}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (res.ok) {
            alert(data.message || "Documento eliminado correctamente.");
            loadStats();
            loadDocuments();
        } else {
            alert("Error al eliminar: " + (data.detail || "Error desconocido."));
        }
    } catch (err) {
        alert("Error de conexión al eliminar: " + err.message);
    }
}

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
                <div class="player-name">${escapeHtml(s.name)}</div>
                <div class="player-pos">${escapeHtml(s.role)}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.5rem;">
                    Área: <strong>${escapeHtml(s.department_name)}</strong>
                </div>
                <div style="font-size: 0.78rem; color: #cbd5e1;">
                    📧 ${escapeHtml(s.email)}<br>
                    📞 ${escapeHtml(s.phone)}
                </div>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading staff:", e);
    }
}

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
                <td>${escapeHtml(l.created_at)}</td>
                <td><strong>${escapeHtml(l.query_text)}</strong></td>
                <td><span class="badge badge-info">${escapeHtml(l.department_filter || 'Todos')}</span></td>
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
        if (res.ok) {
            alert(`Éxito: ${data.message} (${data.chunks_created} fragmentos generados)`);
            document.getElementById("ingest-form").reset();
            loadStats();
            loadDocuments();
        } else {
            alert("Error al indexar: " + (data.detail || "Error desconocido"));
        }
    } catch (err) {
        alert("Error al indexar: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "📥 Indexar Documento en Base RAG";
    }
}
