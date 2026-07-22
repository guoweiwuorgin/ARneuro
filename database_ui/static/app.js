const state = { page: 1, pageSize: 40, activeTab: "overview", detail: null, filters: {} };
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
const nl = (value) => escapeHtml(value).replace(/\n/g, "<br>");
const value = (object, key, fallback = "") => object && object[key] !== undefined && object[key] !== null ? object[key] : fallback;

async function request(url, options = {}) {
  const response = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function toast(message, isError = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.remove("hidden");
  window.clearTimeout(window.__toastTimer);
  window.__toastTimer = window.setTimeout(() => element.classList.add("hidden"), 4200);
}

function collectFilters() {
  return {
    q: $("#filterQ").value.trim(), pmid: $("#filterPmid").value.trim(), task: $("#filterTask").value.trim(),
    task_role: $("#filterRole").value, fmri: $("#filterFmri").value, scanner: $("#filterScanner").value.trim(),
    training_status: $("#filterTraining").value, needs: $("#filterNeeds").value, page: String(state.page), page_size: String(state.pageSize),
  };
}

function badge(text, tone = "") { return `<span class="badge ${tone}">${escapeHtml(text)}</span>`; }

async function loadStats() {
  const stats = await request("/api/stats");
  $("#workspacePath").textContent = stats.workspace_path;
  const values = [
    [stats.documents, "eligible documents"], [stats.documents_with_methods, "with Methods"], [stats.documents_with_results, "with Results"],
    [stats.active_task_occurrences, "unique document tasks"], [stats.raw_active_task_occurrences, "raw task records"],
    [stats.passages, "indexed passages"], [stats.training_status_counts.gold || 0, "gold documents"],
  ];
  $("#metrics").innerHTML = values.map(([number, label]) => `<article class="metric"><b>${Number(number).toLocaleString()}</b><span>${label}</span></article>`).join("");
}

async function loadDocuments() {
  state.filters = collectFilters();
  const query = new URLSearchParams(state.filters).toString();
  const data = await request(`/api/documents?${query}`);
  $("#resultCount").textContent = `${data.total.toLocaleString()} matching documents`;
  const pages = Math.max(1, Math.ceil(data.total / data.page_size));
  $("#pageStatus").textContent = `page ${data.page} of ${pages}`;
  $("#prevPage").disabled = data.page <= 1;
  $("#nextPage").disabled = data.page >= pages;
  $("#documentRows").innerHTML = data.items.length ? data.items.map((item) => {
    const structure = [item.has_methods ? badge("Methods", "good") : badge("No Methods", "danger"), item.has_results ? badge("Results", "good") : badge("No Results", "warn"), item.has_tables ? badge("Tables", "good") : ""].join("");
    const trainingTone = item.training_status === "gold" ? "good" : item.training_status === "exclude" ? "danger" : item.training_status === "in_progress" ? "warn" : "";
    return `<tr><td><strong>${escapeHtml(item.pmid)}</strong><br><span class="small muted">${escapeHtml(item.scanner_type || "scanner unknown")}</span></td>
      <td><div class="document-title">${escapeHtml(item.title || "Untitled record")}</div><div class="small">${item.task_roles.map((role) => badge(role)).join("")}</div></td>
      <td class="small">${escapeHtml(item.sample_summary || `Typical: ${item.typical_human_total || "-"}; Patient: ${item.patient_total || "-"}`)}</td>
      <td>${item.task_count}</td><td>${structure}</td><td>${badge(item.training_status, trainingTone)}<br><span class="small muted">${escapeHtml(item.review_status)}</span></td>
      <td><button class="open-button" data-pmid="${escapeHtml(item.pmid)}">Detail</button></td></tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">No document matches these filters.</td></tr>`;
  document.querySelectorAll(".open-button").forEach((button) => button.addEventListener("click", () => openDetail(button.dataset.pmid)));
}

async function openDetail(pmid) {
  try {
    state.detail = await request(`/api/document/${encodeURIComponent(pmid)}`);
    state.activeTab = "overview"; state.selectedTableIndex = 0; state.selectedSection = null;
    $("#detailPmid").textContent = `PMID ${state.detail.document.pmid}`;
    $("#detailTitle").textContent = state.detail.document.title || "Untitled record";
    $("#drawerBackdrop").classList.remove("hidden"); $("#documentDrawer").classList.remove("hidden");
    renderDetail();
  } catch (error) { toast(error.message, true); }
}

function closeDetail() { $("#drawerBackdrop").classList.add("hidden"); $("#documentDrawer").classList.add("hidden"); state.detail = null; }
function currentPmid() { return state.detail.document.pmid; }

function renderDetail() {
  document.querySelectorAll("#tabBar button").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.activeTab));
  const renderers = { overview: renderOverview, sections: renderSections, tasks: renderTasks, tables: renderTables, coordinates: renderCoordinates, training: renderTraining, history: renderHistory };
  $("#detailContent").innerHTML = renderers[state.activeTab]();
  bindDetailEvents();
}

function renderOverview() {
  const detail = state.detail; const info = detail.document.method_info || {}; const flags = detail.quality_flags;
  const acquisition = info.acquisition || {}; const groups = Array.isArray(info.sample_groups) ? info.sample_groups : [];
  const sample = groups.length ? groups.map((group) => `${group.name || "sample"}${group.n ? ` (n=${group.n})` : ""}`).join("; ") : `Typical: ${detail.document.typical_human_total_raw || "-"}; Patient: ${detail.document.patient_total_raw || "-"}`;
  return `<div class="split"><div class="stack"><section class="surface"><h3>Review signals</h3>${flags.length ? flags.map((flag) => `<div class="flag ${flag.level}">${escapeHtml(flag.message)}</div>`).join("") : `<p class="empty">No automatic quality signal.</p>`}</section>
  <section class="surface"><h3>Document metadata</h3><dl class="kv"><dt>PMID</dt><dd>${escapeHtml(detail.document.pmid)}</dd><dt>Study type</dt><dd>${escapeHtml(info.study_type || "unknown")}</dd><dt>Sample groups</dt><dd>${escapeHtml(sample)}</dd><dt>Research modalities</dt><dd>${escapeHtml((info.research_modalities || []).join(", ") || "-")}</dd><dt>Instrument / scanner</dt><dd>${escapeHtml(acquisition.scanner_or_instrument || info.scanner_type || "-")}</dd><dt>Study design</dt><dd>${escapeHtml(info.study_design || "-")}</dd></dl></section></div>
  <section class="surface"><h3>Segmented document</h3><p class="training-help">Use the Segments tab to correct Methods and Results while preserving the original corpus text. Each revision remains linked to this PMID and source section.</p>
  <div class="kv"><dt>Sections</dt><dd>${detail.sections.map((section) => badge(section.section_name)).join("")}</dd><dt>Indexed tasks</dt><dd>${detail.tasks.length}</dd><dt>Detected table candidates</dt><dd>${detail.table_candidates.length}</dd><dt>Task-table links</dt><dd>${detail.links.length}</dd><dt>Coordinates</dt><dd>${detail.coordinates.length}</dd></div></section></div>`;
}

function renderSections() {
  const sections = state.detail.sections;
  if (!sections.length) return `<p class="empty">No segmented section was stored for this document.</p>`;
  const selected = sections.find((section) => section.section_name === state.selectedSection) || sections[0];
  state.selectedSection = selected.section_name;
  const revision = selected.revision || {}; const revised = value(revision, "revised_content", selected.content);
  return `<div class="split"><section class="surface"><h3>Sections</h3><div class="section-list">${sections.map((section) => `<button class="section-choice ${section.section_name === selected.section_name ? "active" : ""}" data-section-select="${escapeHtml(section.section_name)}"><strong>${escapeHtml(section.section_name)}</strong><small>${String(section.content || "").length.toLocaleString()} characters${section.revision && section.revision.review_status ? ` / ${escapeHtml(section.revision.review_status)}` : ""}</small></button>`).join("")}</div></section>
  <section class="surface"><h3>${escapeHtml(selected.section_name)} correction</h3><div class="editor-grid"><div><h4>Original segmented text</h4><textarea readonly>${escapeHtml(selected.content || "")}</textarea></div><div><h4>Curated text</h4><textarea id="sectionRevised">${escapeHtml(revised)}</textarea></div></div><div class="form-grid"><label>Review status<select id="sectionStatus"><option value="draft">Draft</option><option value="reviewed">Reviewed</option><option value="gold">Gold</option></select></label><label>Reviewer<input id="sectionReviewer" value="${escapeHtml(value(revision, "reviewer"))}" placeholder="Initials or reviewer ID"></label><label class="span-two">Revision rationale<textarea id="sectionRationale">${escapeHtml(value(revision, "rationale"))}</textarea></label></div><div class="form-actions"><button class="primary" id="saveSection" data-section="${escapeHtml(selected.section_name)}">Save section revision</button></div></section></div>`;
}

function renderTasks() {
  const tasks = state.detail.tasks;
  if (!tasks.length) return `<p class="empty">No task occurrence is currently indexed for this PMID.</p>`;
  return `<section class="surface"><h3>Task / paradigm extraction review</h3><p class="training-help">Retain the model result as provenance, then enter a corrected name, paradigm category, in-scanner status, and description where needed.</p>${tasks.map((task, index) => {
    const review = task.review || {}; const id = `task-${index}`;
    const mergedCount = Number(task.member_occurrence_count || 1);
    return `<details class="task-item" ${index < 2 ? "open" : ""}><summary class="item-title"><span>${escapeHtml(review.reviewed_task_name || task.final_task_name || task.raw_name || "Unnamed task")}</span><small>${escapeHtml(task.task_role || "unknown")} / ${mergedCount} merged source record${mergedCount === 1 ? "" : "s"}</small></summary>
      <p class="description">${nl(task.description || "No description extracted.")}</p>${task.clue_sentences ? `<div class="clue">${nl(task.clue_sentences)}</div>` : ""}
      <div class="form-grid"><label>Reviewed task / paradigm name<input id="${id}-name" value="${escapeHtml(review.reviewed_task_name || task.final_task_name || task.raw_name || "")}"></label><label>Paradigm category<input id="${id}-role" value="${escapeHtml(review.reviewed_task_role || task.task_role || "")}" placeholder="experimental_task"></label><label>In-scanner status<select id="${id}-scanner"><option ${ (review.in_scanner_status || "unknown") === "unknown" ? "selected" : ""}>unknown</option><option ${review.in_scanner_status === "yes" ? "selected" : ""}>yes</option><option ${review.in_scanner_status === "no" ? "selected" : ""}>no</option><option ${review.in_scanner_status === "mixed" ? "selected" : ""}>mixed</option><option ${review.in_scanner_status === "uncertain" ? "selected" : ""}>uncertain</option></select></label><label>Review status<select id="${id}-status"><option ${ (review.review_status || "draft") === "draft" ? "selected" : ""}>draft</option><option ${review.review_status === "reviewed" ? "selected" : ""}>reviewed</option><option ${review.review_status === "gold" ? "selected" : ""}>gold</option></select></label><label class="span-two">Reviewed description<textarea id="${id}-description">${escapeHtml(review.reviewed_description || task.description || "")}</textarea></label><label>Reviewer<input id="${id}-reviewer" value="${escapeHtml(review.reviewer || "")}"></label><label>Note<textarea id="${id}-note">${escapeHtml(review.note || "")}</textarea></label></div><div class="form-actions"><button class="primary save-task" data-id="${id}" data-key="${escapeHtml(task.occurrence_key)}">Save task review</button></div></details>`;
  }).join("")}</section>`;
}

function tableOptions() { return state.detail.table_candidates.map((table, index) => `<option value="${escapeHtml(table.table_key)}">${index + 1}. ${escapeHtml(table.section_name)}: ${escapeHtml(table.header).slice(0, 90)}</option>`).join(""); }
function taskOptions() { return state.detail.tasks.filter((task) => !Number(task.exclude_from_all_analysis)).map((task, index) => `<option value="${escapeHtml(task.occurrence_key)}">${index + 1}. ${escapeHtml((task.review || {}).reviewed_task_name || task.final_task_name || task.raw_name || "unnamed")}</option>`).join(""); }

function pipeCells(line) {
  const text = String(line || "").trim().replace(/^\|/, "").replace(/\|$/, "");
  return text.split(/(?<!\\)\|/).map((cell) => cell.replace(/\\\|/g, "|").trim());
}

function isMarkdownSeparator(cells) {
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")));
}

function renderMarkdownTable(markdown) {
  const rows = String(markdown || "").split(/\r?\n/).map(pipeCells).filter((cells) => cells.length && cells.some(Boolean));
  if (rows.length < 2 || !isMarkdownSeparator(rows[1])) return `<pre class="markdown-preview">${escapeHtml(markdown)}</pre>`;
  const header = rows[0]; const body = rows.slice(2); const width = header.length;
  const normalized = (cells) => Array.from({ length: width }, (_, index) => cells[index] || "");
  return `<div class="markdown-table-wrap"><table class="markdown-table"><thead><tr>${normalized(header).map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr></thead><tbody>${body.map((cells) => `<tr>${normalized(cells).map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function renderStructuredTable(rows) {
  if (!Array.isArray(rows) || !rows.length) return "";
  const width = Math.max(...rows.map((row) => row.length));
  const normalized = (row) => Array.from({ length: width }, (_, index) => row[index] || "");
  const header = normalized(rows[0]); const body = rows.slice(1);
  return `<div class="markdown-table-wrap"><table class="markdown-table"><thead><tr>${header.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${normalized(row).map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function renderCandidateTable(table) {
  return table.table_format === "html" ? renderStructuredTable(table.table_rows) : renderMarkdownTable(table.source_markdown);
}

function renderTables() {
  const tables = state.detail.table_candidates;
  const selectedIndex = Math.max(0, Math.min(Number(state.selectedTableIndex || 0), Math.max(0, tables.length - 1)));
  const selected = tables[selectedIndex];
  const selectedReview = selected ? selected.review || {} : {};
  const selectedId = `table-${selectedIndex}`;
  const tableViewer = selected ? `<div class="table-number-nav" aria-label="Select table">${tables.map((table, index) => `<button class="table-number ${index === selectedIndex ? "active" : ""}" data-table-select="${index}" title="Table ${index + 1}: ${escapeHtml(table.section_name)}">${index + 1}</button>`).join("")}</div>
    <article class="table-card selected"><div class="item-title"><span>Table ${selectedIndex + 1} of ${tables.length}: ${escapeHtml(selected.section_name)}</span><small>${escapeHtml(selected.header).slice(0, 160)}</small></div>${renderCandidateTable(selected)}
    <div class="form-grid"><label>Activation table?<select id="${selectedId}-activation"><option ${ (selectedReview.is_activation_table || "unknown") === "unknown" ? "selected" : ""}>unknown</option><option ${selectedReview.is_activation_table === "yes" ? "selected" : ""}>yes</option><option ${selectedReview.is_activation_table === "no" ? "selected" : ""}>no</option></select></label><label>Table label<input id="${selectedId}-label" value="${escapeHtml(selectedReview.table_label || "")}"></label><label>Contrast<input id="${selectedId}-contrast" value="${escapeHtml(selectedReview.contrast || "")}"></label><label>Review status<select id="${selectedId}-status"><option ${ (selectedReview.review_status || "draft") === "draft" ? "selected" : ""}>draft</option><option ${selectedReview.review_status === "reviewed" ? "selected" : ""}>reviewed</option><option ${selectedReview.review_status === "gold" ? "selected" : ""}>gold</option></select></label><label>Reviewer<input id="${selectedId}-reviewer" value="${escapeHtml(selectedReview.reviewer || "")}"></label><label>Note<textarea id="${selectedId}-note">${escapeHtml(selectedReview.note || "")}</textarea></label></div><div class="form-actions"><button class="primary save-table" data-id="${selectedId}" data-key="${escapeHtml(selected.table_key)}" data-section="${escapeHtml(selected.section_name)}" data-index="${selected.table_index}">Save table review</button></div></article>` : `<p class="empty">No pipe-delimited Markdown table candidate was detected. You can still preserve a document as gold after reviewing its Results text.</p>`;
  return `<div class="stack"><section class="surface"><h3>Activation table candidates</h3><p class="training-help">Select a table number to inspect it as rows and columns. Confirm activation tables, annotate their contrast, then link them to tasks below.</p>${tableViewer}</section>
  <section class="surface"><h3>Task to table link</h3><div class="form-grid"><label>Task<select id="linkTask">${taskOptions()}</select></label><label>Table<select id="linkTable">${tableOptions()}</select></label><label>Contrast<input id="linkContrast" placeholder="e.g. sentence > fixation"></label><label>Review status<select id="linkStatus"><option>draft</option><option>reviewed</option><option>gold</option></select></label><label class="span-two">Evidence<textarea id="linkEvidence" placeholder="Quoted sentence or rationale that binds this task to the table."></textarea></label><label>Reviewer<input id="linkReviewer"></label></div><div class="form-actions"><button id="saveLink" class="primary" ${tables.length ? "" : "disabled"}>Save task-table link</button></div><h4>Existing links</h4>${state.detail.links.length ? state.detail.links.map((link) => `<div class="link-row"><strong>${escapeHtml(link.occurrence_key)}</strong><span>&#8594;</span><span>${escapeHtml(link.table_key)}</span>${link.contrast ? badge(link.contrast) : ""}${link.review_status ? badge(link.review_status) : ""}</div>`).join("") : `<p class="empty">No task-table link saved yet.</p>`}</section></div>`;
}

function renderCoordinates() {
  const rows = state.detail.coordinates;
  return `<div class="split"><section class="surface"><h3>Add coordinate annotation</h3><p class="training-help">Coordinates are intentionally stored as a separate evidence layer. Link them to both the task and table when possible.</p><div class="form-grid"><label>Task<select id="coordinateTask">${taskOptions()}</select></label><label>Table<select id="coordinateTable">${tableOptions()}</select></label><label>Contrast<input id="coordinateContrast"></label><label>Region<input id="coordinateRegion" placeholder="e.g. left IFG"></label><label>Hemisphere<select id="coordinateHemisphere"><option value="">Unknown</option><option>left</option><option>right</option><option>bilateral</option><option>midline</option></select></label><label>Space<select id="coordinateSpace"><option value="">Unknown</option><option>MNI</option><option>Talairach</option><option>native</option></select></label><label>x<input id="coordinateX" inputmode="decimal"></label><label>y<input id="coordinateY" inputmode="decimal"></label><label>z<input id="coordinateZ" inputmode="decimal"></label><label>Statistic<input id="coordinateStatistic" placeholder="e.g. z=4.15"></label><label>Review status<select id="coordinateStatus"><option>draft</option><option>reviewed</option><option>gold</option></select></label><label>Reviewer<input id="coordinateReviewer"></label><label class="span-two">Evidence<textarea id="coordinateEvidence" placeholder="Table row or source text supporting this coordinate."></textarea></label></div><div class="form-actions"><button id="addCoordinate" class="primary">Add coordinate</button></div></section><section class="surface"><h3>Saved coordinates</h3>${rows.length ? rows.map((row) => `<article class="coordinate-item"><div class="item-title"><span>${escapeHtml(row.region || "Unlabelled region")} (${escapeHtml(row.x)}, ${escapeHtml(row.y)}, ${escapeHtml(row.z)})</span><button class="secondary delete-coordinate" data-coordinate="${row.coordinate_id}">Delete</button></div><p class="small">${escapeHtml(row.hemisphere || "hemisphere unknown")} / ${escapeHtml(row.coordinate_space || "space unknown")} / ${escapeHtml(row.contrast || "contrast unknown")}</p>${row.evidence ? `<div class="clue">${nl(row.evidence)}</div>` : ""}</article>`).join("") : `<p class="empty">No coordinate annotation saved.</p>`}</section></div>`;
}

function renderTraining() {
  const annotation = state.detail.document_annotation || {};
  return `<div class="split"><section class="surface"><h3>Document-level review</h3><p class="training-help">Mark only carefully curated documents as <strong>gold</strong>. Export produces three JSONL sets: task extraction, task-to-table linking, and coordinate extraction.</p><div class="form-grid"><label>Training status<select id="docTraining"><option value="not_selected">not_selected</option><option value="in_progress">in_progress</option><option value="gold">gold</option><option value="exclude">exclude</option></select></label><label>Review status<select id="docReview"><option value="not_started">not_started</option><option value="draft">draft</option><option value="reviewed">reviewed</option><option value="complete">complete</option></select></label><label>Reviewer<input id="docReviewer" value="${escapeHtml(annotation.reviewer || "")}"></label><label class="span-two">Document note<textarea id="docNote">${escapeHtml(annotation.note || "")}</textarea></label></div><div class="form-actions"><button id="saveDocumentAnnotation" class="primary">Save document status</button></div></section><section class="surface"><h3>Fine-tuning readiness</h3><div class="kv"><dt>Methods</dt><dd>${state.detail.sections.some((section) => section.section_name.toLowerCase() === "methods") ? badge("available", "good") : badge("missing", "danger")}</dd><dt>Results</dt><dd>${state.detail.sections.some((section) => section.section_name.toLowerCase() === "results") ? badge("available", "good") : badge("missing", "warn")}</dd><dt>Task annotations</dt><dd>${state.detail.tasks.filter((task) => (task.review || {}).review_status === "gold").length} gold / ${state.detail.tasks.length} total</dd><dt>Table annotations</dt><dd>${state.detail.table_candidates.filter((table) => (table.review || {}).review_status === "gold").length} gold / ${state.detail.table_candidates.length} candidates</dd><dt>Coordinates</dt><dd>${state.detail.coordinates.length}</dd></div><h4>Recommended sequence</h4><ol class="training-help"><li>Correct Methods and Results segmentation.</li><li>Review each task and its evidence sentence.</li><li>Confirm activation tables, links, and coordinates.</li><li>Set the document to gold, then export JSONL.</li></ol></section></div>`;
}

function renderHistory() {
  const rows = state.detail.history;
  return `<section class="surface"><h3>Annotation history</h3>${rows.length ? rows.map((row) => `<article class="history-item"><div class="item-title"><span>${escapeHtml(row.entity_type)} / ${escapeHtml(row.action)}</span><small>${escapeHtml(row.created_at)}</small></div><p class="small">${escapeHtml(row.entity_key)}${row.reviewer ? ` / ${escapeHtml(row.reviewer)}` : ""}</p></article>`).join("") : `<p class="empty">No manual annotation has been written for this PMID.</p>`}</section>`;
}

function optionSelect(selector, selected) { const element = $(selector); if (element) element.value = selected || element.value; }

function bindDetailEvents() {
  document.querySelectorAll("[data-section-select]").forEach((button) => button.addEventListener("click", () => { state.selectedSection = button.dataset.sectionSelect; renderDetail(); }));
  document.querySelectorAll("[data-table-select]").forEach((button) => button.addEventListener("click", () => { state.selectedTableIndex = Number(button.dataset.tableSelect); renderDetail(); }));
  const saveSection = $("#saveSection");
  if (saveSection) saveSection.addEventListener("click", async () => {
    const section = state.detail.sections.find((item) => item.section_name === saveSection.dataset.section); const revision = section.revision || {};
    await save(`/api/document/${encodeURIComponent(currentPmid())}/section/${encodeURIComponent(section.section_name)}`, { base_content_sha256: revision.base_content_sha256 || "", original_content: section.content || "", revised_content: $("#sectionRevised").value, review_status: $("#sectionStatus").value, reviewer: $("#sectionReviewer").value, rationale: $("#sectionRationale").value });
  });
  document.querySelectorAll(".save-task").forEach((button) => button.addEventListener("click", async () => { const id = button.dataset.id; await save(`/api/document/${encodeURIComponent(currentPmid())}/task/${encodeURIComponent(button.dataset.key)}`, { reviewed_task_name: $(`#${id}-name`).value, reviewed_description: $(`#${id}-description`).value, reviewed_task_role: $(`#${id}-role`).value, in_scanner_status: $(`#${id}-scanner`).value, review_status: $(`#${id}-status`).value, reviewer: $(`#${id}-reviewer`).value, note: $(`#${id}-note`).value }); }));
  document.querySelectorAll(".save-table").forEach((button) => button.addEventListener("click", async () => { const id = button.dataset.id; const table = state.detail.table_candidates.find((item) => item.table_key === button.dataset.key); await save(`/api/document/${encodeURIComponent(currentPmid())}/table/${encodeURIComponent(button.dataset.key)}`, { section_name: button.dataset.section, table_index: Number(button.dataset.index), source_markdown: table.source_markdown, is_activation_table: $(`#${id}-activation`).value, table_label: $(`#${id}-label`).value, contrast: $(`#${id}-contrast`).value, review_status: $(`#${id}-status`).value, reviewer: $(`#${id}-reviewer`).value, note: $(`#${id}-note`).value }); }));
  const saveLink = $("#saveLink");
  if (saveLink) saveLink.addEventListener("click", async () => await save(`/api/document/${encodeURIComponent(currentPmid())}/link`, { occurrence_key: $("#linkTask").value, table_key: $("#linkTable").value, contrast: $("#linkContrast").value, evidence: $("#linkEvidence").value, review_status: $("#linkStatus").value, reviewer: $("#linkReviewer").value }));
  const addCoordinate = $("#addCoordinate");
  if (addCoordinate) addCoordinate.addEventListener("click", async () => await save(`/api/document/${encodeURIComponent(currentPmid())}/coordinate`, { occurrence_key: $("#coordinateTask").value, table_key: $("#coordinateTable").value, contrast: $("#coordinateContrast").value, region: $("#coordinateRegion").value, hemisphere: $("#coordinateHemisphere").value, x: $("#coordinateX").value, y: $("#coordinateY").value, z: $("#coordinateZ").value, coordinate_space: $("#coordinateSpace").value, statistic: $("#coordinateStatistic").value, evidence: $("#coordinateEvidence").value, review_status: $("#coordinateStatus").value, reviewer: $("#coordinateReviewer").value }));
  document.querySelectorAll(".delete-coordinate").forEach((button) => button.addEventListener("click", async () => { if (!window.confirm("Delete this coordinate annotation?")) return; try { await request(`/api/coordinate/${button.dataset.coordinate}`, { method: "DELETE" }); await reloadDetail("Coordinate deleted."); } catch (error) { toast(error.message, true); } }));
  const saveDoc = $("#saveDocumentAnnotation");
  if (saveDoc) { optionSelect("#docTraining", value(state.detail.document_annotation, "training_status", "not_selected")); optionSelect("#docReview", value(state.detail.document_annotation, "review_status", "not_started")); saveDoc.addEventListener("click", async () => await save(`/api/document/${encodeURIComponent(currentPmid())}/annotation`, { training_status: $("#docTraining").value, review_status: $("#docReview").value, reviewer: $("#docReviewer").value, note: $("#docNote").value })); }
}

async function save(url, payload) { try { await request(url, { method: "POST", body: JSON.stringify(payload) }); await reloadDetail("Saved."); } catch (error) { toast(error.message, true); } }
async function reloadDetail(message = "Updated.") { const pmid = currentPmid(); state.detail = await request(`/api/document/${encodeURIComponent(pmid)}`); renderDetail(); await loadStats(); await loadDocuments(); toast(message); }

function bindMainEvents() {
  $("#applyFilters").addEventListener("click", () => { state.page = 1; loadDocuments().catch((error) => toast(error.message, true)); });
  $("#resetFilters").addEventListener("click", () => { ["#filterQ", "#filterPmid", "#filterTask", "#filterScanner"].forEach((id) => $(id).value = ""); ["#filterRole", "#filterFmri", "#filterTraining", "#filterNeeds"].forEach((id) => $(id).value = ""); state.page = 1; loadDocuments().catch((error) => toast(error.message, true)); });
  ["#filterQ", "#filterPmid", "#filterTask", "#filterRole", "#filterScanner"].forEach((id) => $(id).addEventListener("keydown", (event) => { if (event.key === "Enter") { state.page = 1; loadDocuments().catch((error) => toast(error.message, true)); } }));
  $("#prevPage").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); loadDocuments().catch((error) => toast(error.message, true)); });
  $("#nextPage").addEventListener("click", () => { state.page += 1; loadDocuments().catch((error) => toast(error.message, true)); });
  $("#refreshData").addEventListener("click", () => Promise.all([loadStats(), loadDocuments()]).then(() => toast("Data refreshed.")).catch((error) => toast(error.message, true)));
  $("#closeDrawer").addEventListener("click", closeDetail); $("#drawerBackdrop").addEventListener("click", closeDetail);
  document.querySelectorAll("#tabBar button").forEach((button) => button.addEventListener("click", () => { state.activeTab = button.dataset.tab; renderDetail(); }));
  $("#exportGold").addEventListener("click", async () => { if (!window.confirm("Export all documents marked as gold to JSONL training files?")) return; try { const result = await request("/api/export", { method: "POST", body: JSON.stringify({}) }); toast(`Exported ${result.documents} gold document(s).`); } catch (error) { toast(error.message, true); } });
}

async function initialize() { bindMainEvents(); try { await Promise.all([loadStats(), loadDocuments()]); } catch (error) { toast(error.message, true); } }
window.addEventListener("DOMContentLoaded", initialize);
