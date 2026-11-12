// Frontend JavaScript for Club Atlético Puerto Marina RAG (v3.0)

const API_BASE = "";
let currentAlpha = 0.5;
let bannerTimer = null;
let selectedFiles = [];

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initControls();
    initUploadControls();
    setDefaultIngestDate();
    loadStats();
    loadLlmHealth();
    loadPlayers();
    loadDocuments();
    loadStaff();
    loadLogs();
});

function setDefaultIngestDate() {
    const dateInput = document.getElementById("ingest-date");
    if (dateInput && !dateInput.value) {
        dateInput.value = new Date().toISOString().slice(0, 10);
    }
}

function showBanner(message, type = "info") {
    const banner = document.getElementById("app-banner");
    if (!banner) return;
    banner.className = `app-banner ${type}`;
    banner.innerHTML = "";

    const msgSpan = document.createElement("span");
    msgSpan.textContent = message;
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "mini-action-btn";
    closeBtn.textContent = "✕";
    closeBtn.addEventListener("click", () => {
        banner.style.display = "none";
    });

    banner.appendChild(msgSpan);
    banner.appendChild(closeBtn);
    banner.style.display = "flex";

    if (bannerTimer) clearTimeout(bannerTimer);
    if (type !== "error") {
        bannerTimer = setTimeout(() => {
            banner.style.display = "none";
        }, 6000);
    }
}

async function fetchJson(url, options = {}) {
    const res = await fetch(url, options);
    let data = null;
    try {
        data = await res.json();
    } catch {
        data = null;
    }
    if (!res.ok) {
        const detail = data && (data.detail || data.message)
            ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail))
            : `HTTP ${res.status} ${res.statusText}`;
        throw new Error(detail);
    }
    return data;
}

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

function initControls() {
    const alphaSlider = document.getElementById("alpha-slider");
    const alphaValueLabel = document.getElementById("alpha-val");

    if (alphaSlider) {
        alphaSlider.addEventListener("input", (e) => {
            currentAlpha = parseFloat(e.target.value);
            alphaValueLabel.textContent = `${Math.round(currentAlpha * 100)}% BM25 / ${Math.round((1 - currentAlpha) * 100)}% Vector`;
        });
    }

    const submitBtn = document.getElementById("submit-btn");
    if (submitBtn) {
        submitBtn.addEventListener("click", () => executeQuery());
    }

    const queryInput = document.getElementById("query-input");
    if (queryInput) {
        queryInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") executeQuery();
        });
    }

    document.querySelectorAll(".suggestion-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            if (q) executeQuery(q);
        });
    });

    const ingestForm = document.getElementById("ingest-form");
    if (ingestForm) {
        ingestForm.addEventListener("submit", handleIngestSubmit);
    }

    const reindexBtn = document.getElementById("reindex-btn");
    if (reindexBtn) {
        reindexBtn.addEventListener("click", handleReindex);
    }

    const modal = document.getElementById("doc-modal");
    const closeBtn = document.getElementById("modal-close-btn");
    if (closeBtn && modal) {
        closeBtn.addEventListener("click", () => { modal.style.display = "none"; });
        modal.addEventListener("click", (e) => {
            if (e.target === modal) modal.style.display = "none";
        });
    }
}

async function loadStats() {
    try {
        const data = await fetchJson(`${API_BASE}/api/status`);
        if (data.stats) {
            document.getElementById("stat-docs").textContent = data.stats.documents_count;
            document.getElementById("stat-chunks").textContent = data.stats.chunks_count;
            document.getElementById("stat-players").textContent = data.stats.players_count;
            document.getElementById("stat-latency").textContent = `${data.stats.average_query_latency_ms} ms`;

            const deptsSub = document.getElementById("stat-depts-sub");
            if (deptsSub) deptsSub.textContent = `${data.stats.departments_count} departamentos operativos`;
            const playersSub = document.getElementById("stat-players-sub");
            if (playersSub) playersSub.textContent = `${data.stats.players_count} perfiles biomédicos`;

            const modeEl = document.getElementById("stat-mode");
            if (modeEl && data.stats.synthesis_mode) {
                modeEl.textContent = data.stats.synthesis_mode;
            }
        }
    } catch (e) {
        console.warn("Could not load stats:", e);
    }
}

async function loadLlmHealth() {
    const badge = document.getElementById("llm-health-badge");
    if (!badge) return;
    try {
        const health = await fetchJson(`${API_BASE}/api/llm/health`);
        if (!health.enabled) {
            badge.className = "badge badge-info";
            badge.textContent = "Modo Local Offline";
        } else if (health.status === "connected") {
            badge.className = "badge badge-success";
            badge.textContent = `LLM OK: ${health.configured_model}`;
        } else if (health.status === "model_mismatch") {
            badge.className = "badge badge-gold";
            badge.textContent = `Modelo no listado (${health.configured_model})`;
        } else {
            badge.className = "badge badge-danger";
            badge.textContent = "LLM con error (fallback local)";
        }
    } catch {
        badge.className = "badge badge-danger";
        badge.textContent = "Estado LLM no disponible";
    }
}

async function handleReindex() {
    const btn = document.getElementById("reindex-btn");
    if (!btn) return;
    btn.disabled = true;
    btn.textContent = "⏳ Reindexando...";
    try {
        const res = await fetchJson(`${API_BASE}/api/reindex`, { method: "POST" });
        showBanner(
            `Base reindexada: ${res.details.documents_count} documentos y ${res.details.chunks_count} fragmentos activos.`,
            "success"
        );
        await Promise.all([loadStats(), loadDocuments()]);
    } catch (err) {
        showBanner(`Error al reindexar: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.textContent = "🔄 Reindexar";
    }
}

async function executeQuery(presetQuery = null) {
    const input = document.getElementById("query-input");
    const query = (presetQuery || input.value).trim();
    if (!query) return;

    if (presetQuery) input.value = presetQuery;

    const deptFilter = document.getElementById("dept-filter").value || null;
    const topK = parseInt(document.getElementById("top-k-select").value, 10) || 4;
    const submitBtn = document.getElementById("submit-btn");

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Consultando...';

    const resultCard = document.getElementById("result-card");
    resultCard.classList.remove("active");

    try {
        const data = await fetchJson(`${API_BASE}/api/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                department_filter: deptFilter,
                top_k: topK,
                alpha: currentAlpha
            })
        });

        renderResult(data);
        loadStats();
    } catch (err) {
        showBanner("Error al ejecutar la consulta: " + err.message, "error");
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>⚡</span> Consultar RAG';
    }
}

function formatSnippetHtml(rawSnippet) {
    if (!rawSnippet) return "";
    return escapeHtml(rawSnippet)
        .replace(/«/g, "<mark>")
        .replace(/»/g, "</mark>");
}

function renderResult(data) {
    const resultCard = document.getElementById("result-card");
    const answerEl = document.getElementById("answer-content");
    const telemetryEl = document.getElementById("telemetry-badges");
    const fallbackEl = document.getElementById("fallback-notice");
    const chunksEl = document.getElementById("chunks-container");

    const timings = data.timings || { total_ms: 0, retrieval_ms: 0, generation_ms: 0 };
    telemetryEl.innerHTML = `
        <span class="badge badge-info">⏱️ Total: ${escapeHtml(String(timings.total_ms))} ms</span>
        <span class="badge badge-success">🔍 Búsqueda: ${escapeHtml(String(timings.retrieval_ms))} ms</span>
        <span class="badge badge-gold">🧠 Motor: ${escapeHtml(data.model_used)} (${escapeHtml(String(timings.generation_ms))} ms)</span>
        <span class="badge badge-info">🎯 Confianza: ${Math.round((data.confidence || 0) * 100)}%</span>
    `;

    if (fallbackEl) {
        if (data.fallback_used && data.fallback_reason) {
            fallbackEl.textContent = `⚠️ El LLM externo no respondió (${data.fallback_reason}). Se utilizó el sintetizador extractivo local.`;
            fallbackEl.style.display = "block";
        } else {
            fallbackEl.style.display = "none";
        }
    }

    answerEl.innerHTML = formatMarkdown(data.answer);

    chunksEl.innerHTML = "";
    (data.retrieved_chunks || []).forEach((ch, idx) => {
        const chunkDiv = document.createElement("div");
        chunkDiv.className = "chunk-card";

        const modalities = (ch.modalities || [])
            .map(m => `<span class="badge badge-gold">${m === "lexical" ? "BM25 FTS5" : "Vector Denso"}</span>`)
            .join(" ");

        const docCode = (ch.metadata && ch.metadata.doc_code) ? ch.metadata.doc_code : "DOC";
        const snippetHtml = ch.match_snippet
            ? `<p style="margin-bottom: 8px; color: #f59e0b;"><strong>Coincidencia Léxica:</strong> ${formatSnippetHtml(ch.match_snippet)}</p>`
            : "";

        chunkDiv.innerHTML = `
            <div class="chunk-header">
                <div>
                    <strong>[#${idx + 1}] ${escapeHtml(ch.title)}</strong>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 2px;">
                        Área: <code>${escapeHtml(ch.department_id)}</code> | Ref: <code>${escapeHtml(docCode)}</code>
                    </div>
                </div>
                <div class="chunk-scores">
                    ${modalities}
                    <span class="badge badge-info">RRF: ${Number(ch.rrf_score || 0).toFixed(4)}</span>
                    <span class="badge badge-success">CosSim: ${Number(ch.cosine_sim || 0).toFixed(3)}</span>
                </div>
            </div>
            <div class="chunk-body" style="display: ${idx === 0 ? "block" : "none"};">
                ${snippetHtml}
                <div>${escapeHtml(ch.content)}</div>
            </div>
        `;

        const header = chunkDiv.querySelector(".chunk-header");
        const body = chunkDiv.querySelector(".chunk-body");
        header.addEventListener("click", () => {
            body.style.display = body.style.display === "none" ? "block" : "none";
        });

        chunksEl.appendChild(chunkDiv);
    });

    resultCard.classList.add("active");
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function formatInlineMarkdown(escapedText) {
    return escapedText
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[REF:\s*([^\]]+)\]/gi, '<span class="badge badge-gold" style="font-size:0.7rem;">REF: $1</span>');
}

function formatMarkdown(mdText) {
    if (!mdText) return "";

    // Extract fenced code blocks before inline formatting
    const codeBlocks = [];
    let safe = escapeHtml(mdText).replace(/```([\s\S]*?)```/g, (_, code) => {
        const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
        codeBlocks.push(`<pre><code>${code.trim()}</code></pre>`);
        return token;
    });

    const lines = safe.split(/\r?\n/);
    const out = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];
        const trimmed = line.trim();

        if (!trimmed) {
            i++;
            continue;
        }

        if (/^@@CODEBLOCK_\d+@@$/.test(trimmed)) {
            out.push(trimmed);
            i++;
            continue;
        }

        if (/^---+$/.test(trimmed)) {
            out.push('<hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.12); margin: 1rem 0;">');
            i++;
            continue;
        }

        // Headings
        const hMatch = trimmed.match(/^(#{1,4})\s+(.*)$/);
        if (hMatch) {
            const level = Math.min(4, Math.max(2, hMatch[1].length));
            out.push(`<h${level}>${formatInlineMarkdown(hMatch[2])}</h${level}>`);
            i++;
            continue;
        }

        // Markdown tables
        if (trimmed.startsWith("|")) {
            const tableLines = [];
            while (i < lines.length && lines[i].trim().startsWith("|")) {
                tableLines.push(lines[i].trim());
                i++;
            }
            let tableHtml = "<table>";
            let renderedHeader = false;
            tableLines.forEach((row) => {
                if (/^\|[\s\-:|]+\|$/.test(row)) return;
                const rawCells = row.split("|");
                const cells = rawCells.slice(1, row.endsWith("|") ? -1 : undefined);
                const tag = !renderedHeader ? "th" : "td";
                renderedHeader = true;
                tableHtml += "<tr>" + cells.map(c => `<${tag}>${formatInlineMarkdown(c.trim())}</${tag}>`).join("") + "</tr>";
            });
            tableHtml += "</table>";
            out.push(tableHtml);
            continue;
        }

        // Lists (bulleted or numbered)
        if (/^(?:[-*]|\d+\.)\s+/.test(trimmed)) {
            const isOrdered = /^\d+\.\s+/.test(trimmed);
            const tag = isOrdered ? "ol" : "ul";
            let listHtml = `<${tag}>`;
            while (i < lines.length && /^(?:[-*]|\d+\.)\s+/.test(lines[i].trim())) {
                const itemText = lines[i].trim().replace(/^(?:[-*]|\d+\.)\s+/, "");
                listHtml += `<li>${formatInlineMarkdown(itemText)}</li>`;
                i++;
            }
            listHtml += `</${tag}>`;
            out.push(listHtml);
            continue;
        }

        // Paragraph lines
        const paraLines = [];
        while (
            i < lines.length &&
            lines[i].trim() !== "" &&
            !/^(?:#{1,4}\s+|---+$|\||(?:[-*]|\d+\.)\s+|@@CODEBLOCK_\d+@@$)/.test(lines[i].trim())
        ) {
            paraLines.push(formatInlineMarkdown(lines[i].trim()));
            i++;
        }
        out.push(`<p style="margin-bottom: 0.65rem;">${paraLines.join("<br>")}</p>`);
    }

    let finalHtml = out.join("\n");
    codeBlocks.forEach((block, idx) => {
        finalHtml = finalHtml.replace(`@@CODEBLOCK_${idx}@@`, block);
    });
    return finalHtml;
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

async function loadPlayers() {
    const container = document.getElementById("players-grid");
    if (!container) return;
    try {
        const players = await fetchJson(`${API_BASE}/api/players`);
        container.innerHTML = "";
        players.forEach(p => {
            const card = document.createElement("div");
            card.className = "player-card";
            card.innerHTML = `
                <div class="player-dorsal">#${escapeHtml(p.dorsal)}</div>
                <div class="player-name">${escapeHtml(p.name)} ${p.nickname ? `"${escapeHtml(p.nickname)}"` : ""}</div>
                <div class="player-pos">${escapeHtml(p.position)}</div>
                <div class="player-meta">
                    <span>${escapeHtml(p.age)} años</span>
                    <span>•</span>
                    <span>${escapeHtml(p.height_m)} m</span>
                    <span>•</span>
                    <span>${escapeHtml(p.weight_kg)} kg</span>
                </div>
                <div class="player-notes">${escapeHtml(p.notes || "Sin observaciones")}</div>
                <button type="button" class="player-rag-btn">
                    🔍 Consultar Ficha Médica / GPS
                </button>
            `;
            card.querySelector(".player-rag-btn").addEventListener("click", () => {
                queryPlayer(p.name);
            });
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
        const docs = await fetchJson(`${API_BASE}/api/documents`);
        tableBody.innerHTML = "";
        docs.forEach(d => {
            const row = document.createElement("tr");
            const isVirtualSql = String(d.file_path || "").startsWith("sqlite://");
            const fmt = String(d.source_type || (isVirtualSql ? "sql" : "md")).toUpperCase();
            row.innerHTML = `
                <td><code>${escapeHtml(d.doc_code)}</code></td>
                <td><strong>${escapeHtml(d.title)}</strong></td>
                <td><span class="badge badge-info">${escapeHtml(d.department_name)}</span></td>
                <td>${escapeHtml(d.author)}</td>
                <td><span class="badge badge-gold">${escapeHtml(fmt)}</span></td>
                <td>${escapeHtml(d.date)}</td>
                <td><span class="badge badge-info">${escapeHtml(d.chunk_count)} chunks</span></td>
                <td style="white-space: nowrap;">
                    <button type="button" class="mini-action-btn view-doc-btn">👁️ Ver</button>
                    ${isVirtualSql ? "" : '<button type="button" class="mini-action-btn danger del-doc-btn" style="margin-left: 4px;">🗑️</button>'}
                </td>
            `;
            row.querySelector(".view-doc-btn").addEventListener("click", () => openDocumentModal(d.id));
            const delBtn = row.querySelector(".del-doc-btn");
            if (delBtn) {
                delBtn.addEventListener("click", () => handleDeleteDocument(d.id, d.title));
            }
            tableBody.appendChild(row);
        });
    } catch (e) {
        console.error("Error loading documents:", e);
    }
}

async function openDocumentModal(docId) {
    const modal = document.getElementById("doc-modal");
    const titleEl = document.getElementById("modal-doc-title");
    const metaEl = document.getElementById("modal-doc-meta");
    const contentEl = document.getElementById("modal-doc-content");
    const chunksEl = document.getElementById("modal-doc-chunks");
    if (!modal) return;

    try {
        const data = await fetchJson(`${API_BASE}/api/documents/${docId}`);
        const doc = data.document;
        const fmt = String(doc.source_type || "md").toUpperCase();
        const origPart = doc.original_filename ? ` | Archivo: ${doc.original_filename}` : "";
        titleEl.textContent = `${doc.doc_code} — ${doc.title}`;
        metaEl.textContent = `Área: ${doc.department_id} | Formato: ${fmt}${origPart} | Autor: ${doc.author} | Fecha: ${doc.date} | Tags: ${doc.tags || "-"}`;
        contentEl.innerHTML = formatMarkdown(doc.raw_content);

        chunksEl.innerHTML = "";
        (data.chunks || []).forEach(c => {
            const box = document.createElement("div");
            box.className = "chunk-card";
            box.innerHTML = `
                <div class="chunk-header">
                    <strong>Chunk #${Number(c.chunk_index) + 1}: ${escapeHtml(c.title)}</strong>
                </div>
                <div class="chunk-body">${escapeHtml(c.content)}</div>
            `;
            chunksEl.appendChild(box);
        });
        modal.style.display = "flex";
    } catch (err) {
        showBanner(`No se pudo cargar el documento: ${err.message}`, "error");
    }
}

async function handleDeleteDocument(docId, title) {
    if (!confirm(`¿Eliminar el documento "${title}" de la base RAG?`)) return;
    try {
        const res = await fetchJson(`${API_BASE}/api/documents/${docId}`, { method: "DELETE" });
        showBanner(res.message || "Documento eliminado.", "success");
        await Promise.all([loadStats(), loadDocuments()]);
    } catch (err) {
        showBanner(`Error al eliminar documento: ${err.message}`, "error");
    }
}

async function loadStaff() {
    const container = document.getElementById("staff-grid");
    if (!container) return;
    try {
        const staff = await fetchJson(`${API_BASE}/api/staff`);
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
        const logs = await fetchJson(`${API_BASE}/api/logs`);
        tableBody.innerHTML = "";
        logs.forEach(l => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td>${escapeHtml(l.created_at)}</td>
                <td><strong>${escapeHtml(l.query_text)}</strong></td>
                <td><span class="badge badge-info">${escapeHtml(l.department_filter || "Todos")}</span></td>
                <td>${escapeHtml(l.retrieved_chunks_count)}</td>
                <td><span class="badge badge-success">${escapeHtml(l.total_ms)} ms</span></td>
                <td><code>${Number(l.top_similarity || 0).toFixed(3)}</code></td>
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
        const data = await fetchJson(`${API_BASE}/api/documents`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        showBanner(`Éxito: ${data.message} (${data.chunks_created} fragmentos generados)`, "success");
        document.getElementById("ingest-form").reset();
        setDefaultIngestDate();
        await Promise.all([loadStats(), loadDocuments()]);
    } catch (err) {
        showBanner("Error al indexar: " + err.message, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = "📥 Indexar Documento en Base RAG";
    }
}

function initUploadControls() {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const uploadForm = document.getElementById("upload-form");
    if (!dropzone || !fileInput || !uploadForm) return;

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            addFilesToQueue(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            addFilesToQueue(e.target.files);
            fileInput.value = "";
        }
    });

    uploadForm.addEventListener("submit", handleUploadSubmit);
}

function addFilesToQueue(fileList) {
    const allowedExts = [".pdf", ".docx", ".doc", ".md", ".txt"];
    let rejected = [];
    for (const f of fileList) {
        const dotIdx = f.name.lastIndexOf(".");
        const ext = dotIdx >= 0 ? f.name.slice(dotIdx).toLowerCase() : "";
        if (!allowedExts.includes(ext)) {
            rejected.push(f.name);
            continue;
        }
        const exists = selectedFiles.some(sf => sf.name === f.name && sf.size === f.size);
        if (!exists) {
            selectedFiles.push(f);
        }
    }
    if (rejected.length > 0) {
        showBanner(
            `Formato no soportado en: ${rejected.join(", ")}. Permitidos: PDF, DOCX, DOC, MD, TXT.`,
            "error"
        );
    }
    renderFileQueue();
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getFileIcon(filename) {
    const ext = filename.includes(".") ? filename.split(".").pop().toLowerCase() : "";
    if (ext === "pdf") return "📕";
    if (ext === "docx") return "📘";
    if (ext === "doc") return "📄";
    if (ext === "md") return "📝";
    return "📋";
}

function renderFileQueue() {
    const queueEl = document.getElementById("file-queue");
    const submitBtn = document.getElementById("upload-submit-btn");
    if (!queueEl || !submitBtn) return;

    queueEl.innerHTML = "";
    if (selectedFiles.length === 0) {
        queueEl.style.display = "none";
        submitBtn.disabled = true;
        return;
    }

    queueEl.style.display = "flex";
    submitBtn.disabled = false;

    selectedFiles.forEach((f, idx) => {
        const item = document.createElement("div");
        item.className = "file-queue-item";
        item.innerHTML = `
            <div class="file-queue-left">
                <span>${getFileIcon(f.name)}</span>
                <div>
                    <strong style="color: #fff;">${escapeHtml(f.name)}</strong>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">${formatBytes(f.size)}</div>
                </div>
            </div>
            <button type="button" class="file-remove-btn" title="Quitar de la cola">✕</button>
        `;
        item.querySelector(".file-remove-btn").addEventListener("click", () => {
            selectedFiles.splice(idx, 1);
            renderFileQueue();
        });
        queueEl.appendChild(item);
    });
}

async function handleUploadSubmit(e) {
    e.preventDefault();
    if (selectedFiles.length === 0) return;

    const btn = document.getElementById("upload-submit-btn");
    const resultsEl = document.getElementById("upload-results");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Extrayendo texto, generando chunks e indexando...';

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append("files", file));
    formData.append("department_id", document.getElementById("upload-dept").value);
    formData.append("author", document.getElementById("upload-author").value.trim() || "Staff Técnico");
    formData.append("tags", document.getElementById("upload-tags").value.trim());

    try {
        const data = await fetchJson(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData
        });

        resultsEl.innerHTML = "";
        resultsEl.style.display = "grid";

        (data.details || []).forEach(resItem => {
            const card = document.createElement("div");
            card.className = `upload-result-card ${resItem.status === "success" ? "success" : "error"}`;
            const badgeHtml = resItem.status === "success"
                ? `<span class="badge badge-success">${escapeHtml(resItem.chunks_created)} chunks</span>`
                : `<span class="badge badge-danger">Error</span>`;
            const codeHtml = resItem.doc_code
                ? `<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Código: <code>${escapeHtml(resItem.doc_code)}</code></div>`
                : "";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; margin-bottom:0.4rem;">
                    <strong style="color:#fff; word-break:break-word;">${escapeHtml(resItem.filename)}</strong>
                    ${badgeHtml}
                </div>
                <div style="font-size:0.82rem; color:#cbd5e1;">${escapeHtml(resItem.message)}</div>
                ${codeHtml}
            `;
            resultsEl.appendChild(card);
        });

        if (data.successful > 0 && data.failed === 0) {
            showBanner(`Se indexaron ${data.successful} archivo(s) correctamente en la base RAG.`, "success");
            selectedFiles = [];
            renderFileQueue();
        } else if (data.successful > 0) {
            showBanner(`Carga parcial: ${data.successful} indexado(s), ${data.failed} con error.`, "info");
            selectedFiles = [];
            renderFileQueue();
        } else {
            showBanner(`No se pudo indexar ningún archivo (${data.failed} error(es)).`, "error");
        }

        await Promise.all([loadStats(), loadDocuments()]);
    } catch (err) {
        showBanner(`Error durante la carga multiformato: ${err.message}`, "error");
    } finally {
        btn.disabled = selectedFiles.length === 0;
        btn.innerHTML = "📤 Subir e Indexar Archivos en RAG";
    }
}
