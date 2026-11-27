const API_BASE = window.location.origin;

const state = {
  screen: "login",
  email: "",
  userId: null,
  messages: [],
  isStreaming: false,
  abortController: null,
  pendingActions: [],
  logEventSource: null,
  logLevelFilter: "INFO",
  conversationId: null,
};

// --- DOM refs (cached after DOMContentLoaded) ---

let dom = {};

function cacheDOM() {
  dom = {
    loginScreen: document.getElementById("login-screen"),
    chatScreen: document.getElementById("chat-screen"),
    emailStep: document.getElementById("email-step"),
    otpStep: document.getElementById("otp-step"),
    emailInput: document.getElementById("email-input"),
    otpInput: document.getElementById("otp-input"),
    otpEmail: document.getElementById("otp-email"),
    sendCodeBtn: document.getElementById("send-code-btn"),
    verifyBtn: document.getElementById("verify-btn"),
    loginError: document.getElementById("login-error"),
    loginLoading: document.getElementById("login-loading"),
    userEmail: document.getElementById("user-email"),
    messageList: document.getElementById("message-list"),
    chatInput: document.getElementById("chat-input"),
    sendBtn: document.getElementById("send-btn"),
    confirmOverlay: document.getElementById("confirm-overlay"),
    confirmToolName: document.getElementById("confirm-tool-name"),
    confirmToolArgs: document.getElementById("confirm-tool-args"),
    logPanel: document.getElementById("log-panel"),
    logEntries: document.getElementById("log-entries"),
    logLevelFilter: document.getElementById("log-level-filter"),
    devOtpBanner: document.getElementById("dev-otp-banner"),
    devOtpCode: document.getElementById("dev-otp-code"),
    redisEntries: document.getElementById("redis-entries"),
    tdaView: document.getElementById("tda-view"),
    tdaClusters: document.getElementById("tda-clusters"),
    tdaBettiBar: document.getElementById("tda-betti-bar"),
  };
}

// --- API helpers ---

async function apiPost(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return resp.json();
}

async function apiGet(path) {
  const resp = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return resp.json();
}

// --- Screen rendering ---

function showScreen(name) {
  state.screen = name;
  dom.loginScreen.classList.toggle("active", name === "login");
  dom.chatScreen.classList.toggle("active", name === "chat");
}

function showLoginError(msg) {
  dom.loginError.textContent = msg;
  dom.loginError.classList.remove("hidden");
}

function hideLoginError() {
  dom.loginError.classList.add("hidden");
}

function setLoginLoading(loading) {
  dom.loginLoading.classList.toggle("hidden", !loading);
  dom.sendCodeBtn.disabled = loading;
  dom.verifyBtn.disabled = loading;
}

// --- Auth flow ---

async function handleSendCode() {
  const email = dom.emailInput.value.trim();
  if (!email || dom.sendCodeBtn.disabled) return;

  hideLoginError();
  setLoginLoading(true);
  dom.devOtpBanner.classList.add("hidden");
  dom.devOtpCode.value = "";

  try {
    state.email = email;
    dom.otpEmail.textContent = email;
    dom.otpInput.value = "";
    dom.emailStep.classList.add("hidden");
    dom.otpStep.classList.remove("hidden");
    await apiPost("/auth/otp/send", { email });
    dom.otpInput.focus();
    setLoginLoading(false);
  } catch (e) {
    showLoginError(e.message);
    setLoginLoading(false);
  }
}

async function handleVerify() {
  const code = dom.otpInput.value.trim();
  if (code.length !== 6) return;

  hideLoginError();
  setLoginLoading(true);

  try {
    const data = await apiPost("/auth/otp/verify", {
      email: state.email,
      code,
    });
    state.userId = data.user_id;
    enterChat();
  } catch (e) {
    showLoginError(e.message);
  } finally {
    setLoginLoading(false);
  }
}

function handleBackToEmail() {
  dom.otpStep.classList.add("hidden");
  dom.emailStep.classList.remove("hidden");
  hideLoginError();
  dom.emailInput.focus();
}

async function clearSession() {
  try {
    await fetch(`${API_BASE}/sessions/logout`, {
      method: "POST",
      credentials: "include",
    });
  } catch (_) {}
  state.userId = null;
  state.email = "";
  state.messages = [];
  state.pendingActions = [];
  state.conversationId = null;
}

async function handleLogout() {
  await clearSession();
  dom.messageList.innerHTML = "";
  dom.otpStep.classList.add("hidden");
  dom.emailStep.classList.remove("hidden");
  hideLoginError();
  showScreen("login");
}

function enterChat() {
  dom.userEmail.textContent = state.email || state.userId;
  showScreen("chat");
  dom.chatInput.focus();
  state.conversationId = null;
  loadEvalResults();
}

// --- Message rendering ---

let _userScrolledAway = false;

function shouldAutoScroll() {
  if (_userScrolledAway) return false;
  const el = dom.messageList;
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

function scrollToBottom() {
  dom.messageList.scrollTop = dom.messageList.scrollHeight;
}

function initScrollTracking() {
  const jumpBtn = document.createElement("button");
  jumpBtn.className = "scroll-to-bottom-btn hidden";
  jumpBtn.textContent = "↓ Latest";
  jumpBtn.addEventListener("click", () => {
    _userScrolledAway = false;
    scrollToBottom();
  });
  dom.messageList.parentElement.appendChild(jumpBtn);

  let lastScrollTop = 0;
  dom.messageList.addEventListener(
    "scroll",
    () => {
      const el = dom.messageList;
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      if (atBottom) {
        _userScrolledAway = false;
        jumpBtn.classList.add("hidden");
      } else if (el.scrollTop < lastScrollTop) {
        _userScrolledAway = true;
        jumpBtn.classList.remove("hidden");
      }
      lastScrollTop = el.scrollTop;
    },
    { passive: true },
  );
}

function appendMessage(role, content, extra) {
  const wasNearBottom = shouldAutoScroll();

  const el = document.createElement("div");

  const ROLE_CLASS = {
    user: "msg msg-user",
    assistant: "msg msg-assistant",
    system: "msg msg-system",
    tool_call: "tool-card",
    tool_result: "tool-card",
    tool_pending: "tool-card",
    tool_error: "tool-card tool-error",
    blocked: "tool-card tool-blocked",
  };

  el.className = ROLE_CLASS[role] || "msg msg-system";

  const RENDERERS = {
    user: () => {
      el.textContent = content;
    },
    assistant: () => {
      el.textContent = content;
    },
    system: () => {
      el.textContent = content;
    },
    tool_call: () => renderToolCall(el, content, extra),
    tool_result: () => renderToolResult(el, content, extra),
    tool_pending: () => renderToolPending(el, content, extra),
    tool_error: () => renderToolError(el, content, extra),
    blocked: () => renderBlocked(el, content, extra),
  };

  const renderer = RENDERERS[role];
  if (renderer) {
    renderer();
  } else {
    el.textContent = content;
  }

  if (extra && extra.id) el.id = extra.id;

  dom.messageList.appendChild(el);
  state.messages.push({ role, content, extra });

  if (wasNearBottom) requestAnimationFrame(scrollToBottom);
  return el;
}

function renderToolCall(el, toolName, extra) {
  el.innerHTML = `
    <div class="tool-label">Tool Call</div>
    <div class="tool-name">${esc(toolName)}</div>
    <pre>${esc(JSON.stringify(extra.args, null, 2))}</pre>
  `;
}

function renderToolResult(el, toolName, extra) {
  el.innerHTML = `
    <div class="tool-label">Tool Result</div>
    <div class="tool-name">${esc(toolName)}</div>
    <pre>${esc(truncate(extra.result, 2000))}</pre>
  `;
}

function renderToolPending(el, toolName, extra) {
  el.innerHTML = `
    <div class="tool-label">Pending Approval</div>
    <div class="tool-name">${esc(toolName)}</div>
    <pre>${esc(JSON.stringify(extra.args, null, 2))}</pre>
    <div class="tool-status" id="status-${esc(extra.pendingId)}">Awaiting your approval...</div>
  `;
}

function renderToolError(el, content, extra) {
  el.innerHTML = `
    <div class="tool-label">Error</div>
    ${extra.tool ? `<div class="tool-name">${esc(extra.tool)}</div>` : ""}
    <pre>${esc(content)}</pre>
  `;
}

function renderBlocked(el, content, extra) {
  el.innerHTML = `
    <div class="tool-label">Blocked — ${esc(extra.stage)}</div>
    ${extra.tool ? `<div class="tool-name">${esc(extra.tool)}</div>` : ""}
    <div class="blocked-summary">${esc(content)}</div>
    ${extra.category ? `<div class="blocked-category">${esc(extra.category)}</div>` : ""}
  `;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}

function truncate(s, max) {
  if (!s || s.length <= max) return s || "";
  return s.slice(0, max) + "\n... (truncated)";
}

// --- Thinking indicator ---

function showThinking(iteration) {
  let el = document.getElementById("thinking-indicator");
  if (!el) {
    el = document.createElement("div");
    el.id = "thinking-indicator";
    el.className = "thinking-indicator";
    dom.messageList.appendChild(el);
  }
  el.innerHTML = `
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    Thinking${iteration > 1 ? ` (step ${iteration})` : ""}...
  `;
  if (shouldAutoScroll()) requestAnimationFrame(scrollToBottom);
}

function hideThinking() {
  const el = document.getElementById("thinking-indicator");
  if (el) el.remove();
}

// --- SSE stream parser ---

function parseSSEEvent(text) {
  let eventType = "message";
  let data = "";

  for (const line of text.split("\n")) {
    if (line.startsWith("event: ")) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith("data: ")) {
      data = line.slice(6);
    }
  }

  try {
    return { type: eventType, data: JSON.parse(data) };
  } catch {
    return null;
  }
}

async function streamChat(prompt) {
  _streamingEl = null;
  state.isStreaming = true;
  setStreamingUI(true);

  const abortController = new AbortController();
  state.abortController = abortController;

  let response;
  try {
    response = await fetch(`${API_BASE}/agent/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ prompt, conversation_id: state.conversationId }),
      signal: abortController.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") return;
    appendMessage("system", `Connection error: ${e.message}`);
    state.isStreaming = false;
    setStreamingUI(false);
    return;
  }

  if (!response.ok) {
    if (response.status === 401) {
      appendMessage("system", "Session expired. Redirecting to login...");
      setTimeout(() => handleLogout(), 1500);
      return;
    }
    const err = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    appendMessage("system", `Error: ${err.detail || response.statusText}`);
    state.isStreaming = false;
    setStreamingUI(false);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const eventText of events) {
        if (!eventText.trim()) continue;
        const parsed = parseSSEEvent(eventText);
        if (parsed) handleSSEEvent(parsed);
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") {
      appendMessage("system", `Stream error: ${e.message}`);
    }
  } finally {
    hideThinking();
    state.isStreaming = false;
    state.abortController = null;
    _userScrolledAway = false;
    setStreamingUI(false);
    const jumpBtn = document.querySelector(".scroll-to-bottom-btn");
    if (jumpBtn && shouldAutoScroll()) jumpBtn.classList.add("hidden");
  }
}

// --- SSE event dispatch ---

let _streamingEl = null;

const EVENT_HANDLERS = {
  conversation_started: (d) => {
    state.conversationId = d.conversation_id;
  },
  thinking: (d) => showThinking(d.iteration),
  token: (d) => {
    hideThinking();
    if (!_streamingEl) {
      _streamingEl = appendMessage("assistant", "");
    }
    const wasNearBottom = shouldAutoScroll();
    _streamingEl.textContent += d.token;
    if (wasNearBottom) requestAnimationFrame(scrollToBottom);
  },
  tool_call: (d) => {
    hideThinking();
    _streamingEl = null;
    appendMessage("tool_call", d.tool, { args: d.args });
  },
  tool_pending: (d) => {
    hideThinking();
    _streamingEl = null;
    handleToolPending(d);
  },
  tool_result: (d) =>
    appendMessage("tool_result", d.tool, { result: d.result }),
  tool_confirmed: (d) => handleToolConfirmed(d),
  tool_denied: (d) => handleToolDenied(d),
  eval: (d) => {
    loadEvalResults();
  },
  response: (d) => {
    hideThinking();
    _streamingEl = null;
  },
  error: (d) => {
    hideThinking();
    _streamingEl = null;
    appendMessage("tool_error", d.error, { tool: d.tool });
  },
  blocked: (d) => {
    hideThinking();
    _streamingEl = null;
    appendMessage("blocked", d.reason, {
      stage: d.stage,
      tool: d.tool,
      category: d.category,
    });
  },
  eval_retry: (d) => appendMessage("system", `Retrying ${d.tool}: ${d.reason}`),
};

function handleSSEEvent({ type, data }) {
  const handler = EVENT_HANDLERS[type];
  if (handler) {
    handler(data);
  } else {
    console.warn("Unknown SSE event:", type, data);
  }
}

// --- Gated tool confirmation ---

function handleToolPending(data) {
  const pending = {
    pendingId: data.pending_id,
    tool: data.tool,
    args: data.args,
  };

  state.pendingActions.push(pending);
  appendMessage("tool_pending", data.tool, {
    args: data.args,
    pendingId: data.pending_id,
    id: `pending-${data.pending_id}`,
  });
  showConfirmOverlay(pending);
}

function showConfirmOverlay(pending) {
  dom.confirmToolName.textContent = pending.tool;
  dom.confirmToolArgs.textContent = JSON.stringify(pending.args, null, 2);
  dom.confirmOverlay.classList.remove("hidden");
  dom.confirmOverlay.dataset.pendingId = pending.pendingId;

  const approveBtn = dom.confirmOverlay.querySelector(".btn-approve");
  const denyBtn = dom.confirmOverlay.querySelector(".btn-deny");
  approveBtn.disabled = false;
  denyBtn.disabled = false;
}

function hideConfirmOverlay() {
  dom.confirmOverlay.classList.add("hidden");
}

async function handleApprove() {
  const pendingId = dom.confirmOverlay.dataset.pendingId;
  if (!pendingId) return;

  const btns = dom.confirmOverlay.querySelectorAll("button");
  btns.forEach((b) => (b.disabled = true));

  try {
    await apiPost(`/agent/confirm/${pendingId}`, { approved: true });
  } catch (e) {
    appendMessage("system", `Confirm error: ${e.message}`);
  }
}

async function handleDeny() {
  const pendingId = dom.confirmOverlay.dataset.pendingId;
  if (!pendingId) return;

  const btns = dom.confirmOverlay.querySelectorAll("button");
  btns.forEach((b) => (b.disabled = true));

  try {
    await apiPost(`/agent/confirm/${pendingId}`, { approved: false });
  } catch (e) {
    appendMessage("system", `Confirm error: ${e.message}`);
  }
}

function handleToolConfirmed(data) {
  state.pendingActions = state.pendingActions.filter(
    (p) => p.pendingId !== data.pending_id,
  );
  hideConfirmOverlay();

  const statusEl = document.getElementById(`status-${data.pending_id}`);
  if (statusEl) {
    statusEl.textContent = "Approved";
    statusEl.className = "tool-status approved";
  }

  if (state.pendingActions.length > 0) {
    showConfirmOverlay(state.pendingActions[0]);
  }
}

function handleToolDenied(data) {
  state.pendingActions = state.pendingActions.filter(
    (p) => p.pendingId !== data.pending_id,
  );
  hideConfirmOverlay();

  const statusEl = document.getElementById(`status-${data.pending_id}`);
  if (statusEl) {
    statusEl.textContent = "Denied";
    statusEl.className = "tool-status denied";
  }

  if (state.pendingActions.length > 0) {
    showConfirmOverlay(state.pendingActions[0]);
  }
}

// --- Chat input handling ---

function setStreamingUI(streaming) {
  dom.sendBtn.disabled = streaming;
  dom.chatInput.disabled = streaming;
  if (!streaming) dom.chatInput.focus();
}

function handleInputKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
}

function handleSend() {
  const text = dom.chatInput.value.trim();
  if (!text || state.isStreaming) return;

  _userScrolledAway = false;
  dom.chatInput.value = "";
  dom.chatInput.style.height = "auto";
  appendMessage("user", text);
  streamChat(text);
}

// Auto-resize textarea
function initTextareaResize() {
  dom.chatInput.addEventListener("input", () => {
    dom.chatInput.style.height = "auto";
    dom.chatInput.style.height =
      Math.min(dom.chatInput.scrollHeight, 120) + "px";
  });
}

// --- Log panel ---

const LEVEL_BADGE = {
  DEBUG: "log-badge-debug",
  INFO: "log-badge-info",
  WARNING: "log-badge-warn",
  ERROR: "log-badge-error",
};

const LEVEL_RANK = { DEBUG: 0, INFO: 1, WARNING: 2, ERROR: 3 };

function levelPassesFilter(level) {
  const filter = state.logLevelFilter;
  if (!filter) return true;
  return (LEVEL_RANK[level] ?? 0) >= (LEVEL_RANK[filter] ?? 0);
}

function connectLogStream() {
  disconnectLogStream();
  const url = `${API_BASE}/logs/stream?catchup=0`;
  const es = new EventSource(url, { withCredentials: true });

  es.addEventListener("log", (e) => {
    try {
      const entry = JSON.parse(e.data);
      appendLogEntry(entry);
    } catch (_) {}
  });

  es.onerror = () => {
    // silent reconnect — EventSource handles it automatically
  };

  state.logEventSource = es;
}

function disconnectLogStream() {
  if (state.logEventSource) {
    state.logEventSource.close();
    state.logEventSource = null;
  }
}

function appendLogEntry(entry) {
  const willScroll = logShouldAutoScroll();

  const row = document.createElement("div");
  row.className = "log-entry";
  row.dataset.level = (entry.level || "INFO").toUpperCase();
  if (!levelPassesFilter(row.dataset.level)) row.style.display = "none";
  row.onclick = (e) => {
    if (e.target.closest(".log-details") && window.getSelection().toString())
      return;
    row.classList.toggle("expanded");
  };

  const ts = (entry.timestamp || "").split("T")[1] || entry.timestamp || "";
  const timeStr = ts.replace(/\+.*$/, "").slice(0, 12);
  const level = (entry.level || "INFO").toUpperCase();
  const badgeClass = LEVEL_BADGE[level] || "log-badge-debug";
  const event = entry.event || "";

  const details = {};
  for (const [k, v] of Object.entries(entry)) {
    if (
      !["timestamp", "level", "logger", "event", "log_level"].includes(k) &&
      v !== undefined &&
      v !== null
    ) {
      details[k] = k === "code" && event === "otp.dev_code" ? "******" : v;
    }
  }

  const hasDetails = Object.keys(details).length > 0;
  const INLINE_KEYS = [
    "code",
    "email",
    "tool",
    "verdict",
    "pending_id",
    "duration_ms",
    "error",
  ];
  const inlineParts = [];
  for (const k of INLINE_KEYS) {
    if (details[k] !== undefined && details[k] !== null) {
      inlineParts.push(`${k}=${details[k]}`);
    }
  }
  const inlineStr = inlineParts.length > 0 ? " " + inlineParts.join(" ") : "";

  row.innerHTML = `
    <span class="log-time">${esc(timeStr)}</span>
    <span class="log-level-badge ${badgeClass}">${esc(level)}</span>
    <span class="log-event">${esc(event)}${esc(inlineStr)}${hasDetails ? " ..." : ""}</span>
    ${hasDetails ? `<div class="log-details">${esc(JSON.stringify(details, null, 2))}</div>` : ""}
  `;

  if (event === "otp.dev_code" && entry.code) {
    if (dom.devOtpBanner) {
      dom.devOtpCode.value = entry.code;
      dom.devOtpBanner.classList.remove("hidden");
    }
    if (dom.otpInput) {
      dom.otpInput.value = entry.code;
    }
  }

  dom.logEntries.appendChild(row);

  // Keep max 500 entries in DOM
  while (dom.logEntries.children.length > 500) {
    dom.logEntries.removeChild(dom.logEntries.firstChild);
  }

  if (willScroll) dom.logEntries.scrollTop = dom.logEntries.scrollHeight;
}

function logShouldAutoScroll() {
  const el = dom.logEntries;
  return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
}

function handleLogLevelChange() {
  state.logLevelFilter = dom.logLevelFilter.value;
  for (const row of dom.logEntries.children) {
    row.style.display = levelPassesFilter(row.dataset.level) ? "" : "none";
  }
}

function clearLogPanel() {
  dom.logEntries.innerHTML = "";
}

// --- TDA Panel ---

const CLUSTER_COLORS = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
];
let _tdaEventSource = null;
let _logMode = "flat";
const _tdaExpanded = new Set();

function setLogMode(mode) {
  _logMode = mode;
  document.querySelectorAll(".tda-toggle-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  const logEntries = document.getElementById("log-entries");
  const tdaView = document.getElementById("tda-view");
  logEntries.style.display = mode === "flat" ? "" : "none";
  tdaView.style.display = mode === "tda" ? "" : "none";
  if (mode === "tda") {
    connectTDAStream();
  } else {
    disconnectTDAStream();
  }
}

function connectTDAStream() {
  disconnectTDAStream();
  const url = `${API_BASE}/tda/stream?catchup=true`;
  const es = new EventSource(url, { withCredentials: true });
  es.addEventListener("topology", (e) => {
    try {
      const snapshot = JSON.parse(e.data);
      renderTDAView(snapshot);
    } catch (_) {}
  });
  es.addEventListener("error", (e) => {
    try {
      const data = JSON.parse(e.data);
      document.getElementById("tda-clusters").innerHTML =
        `<div class="tda-empty">${esc(data.error || "TDA error")}</div>`;
    } catch (_) {}
  });
  es.onerror = () => {};
  _tdaEventSource = es;
}

function disconnectTDAStream() {
  if (_tdaEventSource) {
    _tdaEventSource.close();
    _tdaEventSource = null;
  }
}

function renderTDAView(snapshot) {
  const bettiBar = document.getElementById("tda-betti-bar");
  const b0 = snapshot.betti[0] || 0;
  const b1 = snapshot.betti[1] || 0;
  bettiBar.innerHTML =
    `<span>β₀</span> ${b0} components` + `<span>β₁</span> ${b1} loops`;

  renderPersistenceDiagram(snapshot.h0_features, snapshot.h1_features);
  renderClusters(snapshot.nodes, snapshot.edges);
}

function renderPersistenceDiagram(h0, h1) {
  const svg = document.getElementById("tda-persistence-diagram");
  const w = svg.clientWidth || 380;
  const h = 120;
  const pad = 16;

  let maxVal = 0;
  const allFeats = [...(h0 || []), ...(h1 || [])];
  for (const f of allFeats) {
    if (f.birth != null) maxVal = Math.max(maxVal, f.birth);
    if (f.death != null) maxVal = Math.max(maxVal, f.death);
  }
  if (maxVal === 0) maxVal = 1;

  const sx = (v) => pad + ((v || 0) / maxVal) * (w - 2 * pad);
  const sy = (v) => pad + ((v || 0) / maxVal) * (h - 2 * pad);

  let html = `<line x1="${pad}" y1="${pad}" x2="${w - pad}" y2="${h - pad}" stroke="#1a3a1a" stroke-width="1" stroke-dasharray="4"/>`;

  for (const f of h0 || []) {
    if (f.birth == null || f.death == null) continue;
    html += `<circle cx="${sx(f.birth)}" cy="${sy(f.death)}" r="3" fill="#4e79a7" opacity="0.8"/>`;
  }
  for (const f of h1 || []) {
    if (f.birth == null || f.death == null) continue;
    html += `<circle cx="${sx(f.birth)}" cy="${sy(f.death)}" r="3" fill="#e15759" opacity="0.8"/>`;
  }

  html += `<text x="${w - pad}" y="${h - 4}" text-anchor="end" font-size="8" fill="#116611">birth →</text>`;
  html += `<text x="2" y="${pad - 4}" font-size="8" fill="#116611">↓ death</text>`;

  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.innerHTML = html;
}

function renderClusters(nodes, edges) {
  const container = document.getElementById("tda-clusters");
  if (!nodes || nodes.length === 0) {
    container.innerHTML =
      '<div class="tda-empty">No conversation data yet — send a message to see topic clusters.</div>';
    return;
  }

  const sorted = [...nodes].sort((a, b) => b.persistence - a.persistence);

  container.innerHTML = "";
  for (const node of sorted) {
    const color = CLUSTER_COLORS[node.cluster_id % CLUSTER_COLORS.length];
    const groupId = `tda-cluster-${node.cluster_id}`;
    const isExpanded = _tdaExpanded.has(groupId);
    const messages = node.messages || [];

    const group = document.createElement("div");
    group.className = "tda-cluster-group" + (isExpanded ? " expanded" : "");

    group.innerHTML = `
      <div class="tda-cluster-header">
        <span class="tda-cluster-badge" style="background:${color}"></span>
        <span class="tda-cluster-label">${esc(node.label)}</span>
        <span class="tda-cluster-count">${node.size} msg${node.size !== 1 ? "s" : ""}</span>
        <span class="tda-cluster-persistence">p=${node.persistence}</span>
      </div>
      <div class="tda-cluster-body">
        ${messages
          .map(
            (m) =>
              `<div class="tda-cluster-msg tda-role-${esc(m.role)}">
            <span class="tda-msg-role">${esc(m.role)}</span>
            <span class="tda-msg-text">${esc(m.text)}</span>
          </div>`,
          )
          .join("")}
      </div>
    `;

    group.querySelector(".tda-cluster-header").addEventListener("click", () => {
      group.classList.toggle("expanded");
      if (group.classList.contains("expanded")) {
        _tdaExpanded.add(groupId);
      } else {
        _tdaExpanded.delete(groupId);
      }
    });

    container.appendChild(group);
  }
}

// --- Redis Panel ---

let _redisInterval = null;
const _redisExpanded = new Set();

async function refreshRedisPanel() {
  try {
    const resp = await fetch(`${API_BASE}/redis/keys`, {
      credentials: "include",
    });
    if (!resp.ok) {
      dom.redisEntries.innerHTML =
        '<div class="redis-empty">Failed to load</div>';
      return;
    }
    const entries = await resp.json();
    renderRedisEntries(entries);
  } catch {
    dom.redisEntries.innerHTML =
      '<div class="redis-empty">Connection error</div>';
  }
}

function renderRedisEntries(entries) {
  if (!entries.length) {
    dom.redisEntries.innerHTML = '<div class="redis-empty">No keys</div>';
    return;
  }

  entries.sort((a, b) => a.key.localeCompare(b.key));

  const DB_CLASS = {
    sessions: "redis-db-sessions",
    conversations: "redis-db-conversations",
  };

  dom.redisEntries.innerHTML = "";
  for (const entry of entries) {
    const entryId = `${entry.db}:${entry.key}`;
    const el = document.createElement("div");
    el.className =
      "redis-entry" + (_redisExpanded.has(entryId) ? " expanded" : "");

    const ttlText = entry.ttl != null ? `${entry.ttl}s` : "";
    const dbClass = DB_CLASS[entry.db] || "redis-db-sessions";
    const valueStr =
      typeof entry.value === "object"
        ? JSON.stringify(entry.value, null, 2)
        : String(entry.value);

    el.innerHTML = `
      <div class="redis-key-row">
        <span class="redis-db-badge ${dbClass}">${esc(entry.db)}</span>
        <span class="redis-type-badge">${esc(entry.type)}</span>
        <span class="redis-key-name">${esc(entry.key)}</span>
        ${ttlText ? `<span class="redis-ttl">${esc(ttlText)}</span>` : ""}
      </div>
      <pre class="redis-value">${esc(valueStr)}</pre>
    `;

    el.addEventListener("click", () => {
      el.classList.toggle("expanded");
      if (el.classList.contains("expanded")) {
        _redisExpanded.add(entryId);
      } else {
        _redisExpanded.delete(entryId);
      }
    });
    dom.redisEntries.appendChild(el);
  }
}

function startRedisPolling() {
  refreshRedisPanel();
  if (_redisInterval) clearInterval(_redisInterval);
  _redisInterval = setInterval(refreshRedisPanel, 3000);
}

function stopRedisPolling() {
  if (_redisInterval) {
    clearInterval(_redisInterval);
    _redisInterval = null;
  }
}

// --- Benchmark & Eval ---

async function triggerBenchmark() {
  const runBtn = document.getElementById("bench-run-btn");
  const resetBtn = document.getElementById("bench-reset-btn");
  const status = document.getElementById("bench-status");
  if (runBtn) runBtn.disabled = true;
  if (resetBtn) resetBtn.disabled = true;
  if (status) status.textContent = "Running...";

  try {
    const data = await apiPost("/benchmark/run", {});
    renderBenchmarkResults(data);
    if (status) status.textContent = "";
  } catch (e) {
    if (status) status.textContent = "Error";
    document.getElementById("bench-results").innerHTML =
      `<div class="bench-empty">Benchmark error: ${esc(e.message)}</div>`;
  } finally {
    if (runBtn) runBtn.disabled = false;
    if (resetBtn) resetBtn.disabled = false;
  }
}

async function resetEval(silent = false) {
  const btn =
    document.getElementById("bench-reset-btn") ||
    document.getElementById("bench-run-btn");
  if (btn) btn.disabled = true;
  try {
    await fetch(`${API_BASE}/eval/reset`, {
      method: "POST",
      credentials: "include",
    });
    if (!silent) {
      document.getElementById("bench-metrics").innerHTML = "";
      document.getElementById("bench-results").innerHTML =
        `<div class="bench-empty">Eval reset. Send queries to start accumulating results.</div>`;
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function loadEvalResults() {
  const metrics = document.getElementById("bench-metrics");
  const results = document.getElementById("bench-results");

  try {
    const resp = await fetch(`${API_BASE}/eval/results`, {
      credentials: "include",
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    if (!data.total || data.total === 0) {
      metrics.innerHTML = "";
      results.innerHTML = `<div class="bench-empty">No eval data yet. Click Run or send queries to start accumulating results.</div>`;
      return;
    }
    renderBenchmarkResults(data);
  } catch (e) {
    // silently fail — eval panel is supplementary
  }
}

function renderBenchmarkResults(data) {
  const metrics = document.getElementById("bench-metrics");
  const results = document.getElementById("bench-results");

  const pct = (v) => (v != null ? `${(v * 100).toFixed(1)}%` : "—");
  const passColor =
    data.pass_rate >= 0.8
      ? "var(--bench-pass)"
      : data.pass_rate >= 0.5
        ? "var(--bench-warn)"
        : "var(--bench-fail)";

  const routedMs =
    data.routed_latency_ms != null ? data.routed_latency_ms.toFixed(0) : "—";
  const blockedMs =
    data.blocked_latency_ms != null ? data.blocked_latency_ms.toFixed(0) : "—";
  const routedP95 =
    data.routed_p95_ms != null ? data.routed_p95_ms.toFixed(0) : "—";
  const blockedP95 =
    data.blocked_p95_ms != null ? data.blocked_p95_ms.toFixed(0) : "—";
  const meanFaith =
    data.mean_faithfulness_score != null
      ? `${(data.mean_faithfulness_score * 100).toFixed(0)}%`
      : "—";
  const passPowKDetail =
    data.pass_pow_k != null && data.k_runs
      ? `${data.passed}/${data.total} · pass^${data.k_runs} ${pct(data.pass_pow_k)}`
      : `${data.passed}/${data.total}`;

  metrics.innerHTML = `
    <div class="bench-metric-grid">
      <div class="bench-metric">
        <div class="bench-metric-value" style="color:${passColor}">${pct(data.pass_rate)}</div>
        <div class="bench-metric-label">Pass Rate</div>
        <div class="bench-metric-detail">${passPowKDetail}</div>
      </div>
      <div class="bench-metric">
        <div class="bench-metric-value">${pct(data.false_block_rate)}</div>
        <div class="bench-metric-label">False Block</div>
        <div class="bench-metric-detail">relevant Qs blocked</div>
      </div>
      <div class="bench-metric">
        <div class="bench-metric-value">${pct(data.false_pass_rate)}</div>
        <div class="bench-metric-label">False Pass</div>
        <div class="bench-metric-detail">irrelevant Qs answered</div>
      </div>
      <div class="bench-metric">
        <div class="bench-metric-value">${pct(data.tool_accuracy)}</div>
        <div class="bench-metric-label">Tool Accuracy</div>
        <div class="bench-metric-detail">correct tool selected</div>
      </div>
      <div class="bench-metric">
        <div class="bench-metric-value">${pct(data.faithfulness_rate)}</div>
        <div class="bench-metric-label">Faithfulness</div>
        <div class="bench-metric-detail">mean ${meanFaith}</div>
      </div>
      <div class="bench-metric bench-metric-wide">
        <div class="bench-metric-label">Latency</div>
        <div class="bench-latency-split">
          <div class="bench-latency-col">
            <span class="bench-latency-heading">Routed</span>
            <span class="bench-latency-val">${routedMs}ms</span>
            <span class="bench-latency-p95">p95 ${routedP95}ms</span>
          </div>
          <div class="bench-latency-col">
            <span class="bench-latency-heading">Blocked</span>
            <span class="bench-latency-val">${blockedMs}ms</span>
            <span class="bench-latency-p95">p95 ${blockedP95}ms</span>
          </div>
        </div>
      </div>
    </div>
  `;

  // Per-tool breakdown
  let html = "";
  if (data.per_tool && Object.keys(data.per_tool).length > 0) {
    html += `<div class="bench-section-title">PER-TOOL BREAKDOWN</div><div class="bench-tool-grid">`;
    for (const [tool, stats] of Object.entries(data.per_tool)) {
      const label = tool === "_irrelevant" ? "Irrelevant (blocked)" : tool;
      const tColor =
        stats.pass_rate >= 0.8
          ? "var(--bench-pass)"
          : stats.pass_rate >= 0.5
            ? "var(--bench-warn)"
            : "var(--bench-fail)";
      html += `
        <div class="bench-tool-card">
          <div class="bench-tool-name">${esc(label)}</div>
          <div class="bench-tool-stats">
            <span style="color:${tColor}">${pct(stats.pass_rate)}</span>
            <span>${stats.passed}/${stats.total}</span>
            <span>${Math.round(stats.mean_latency_ms)}ms</span>
          </div>
        </div>
      `;
    }
    html += `</div>`;
  }

  // Individual test case / run rows
  const rows = data.results || [];
  if (rows.length > 0) {
    html += `<div class="bench-case-list">`;
    for (const r of rows) {
      const badgeClass = r.passed ? "bench-badge-pass" : "bench-badge-fail";
      const badgeText = r.passed ? "PASS" : "FAIL";
      const expectedLabel = r.expected_blocked
        ? "blocked"
        : r.expected_tool || "none";
      const actualLabel = r.actual_blocked
        ? "blocked"
        : r.actual_tool || "none";
      const trialTag = r.run_index ? ` [r${r.run_index}]` : "";
      html += `
        <div class="bench-case">
          <div class="bench-case-header">
            <span class="bench-badge ${badgeClass}">${badgeText}</span>
            <span class="bench-case-id">${esc(r.case_id)}${esc(trialTag)}</span>
            <span class="bench-case-latency">${Math.round(r.latency_ms)}ms</span>
          </div>
          <div class="bench-case-prompt">${esc(r.prompt)}</div>
          <div class="bench-case-meta">
            <span>Expected: ${esc(expectedLabel)}</span>
            <span>Actual: ${esc(actualLabel)}</span>
          </div>
          ${r.failure_reason ? `<div class="bench-case-fail-reason">${esc(r.failure_reason)}</div>` : ""}
          ${r.baselines && r.baselines.length > 0 ? `
            <div class="bench-baselines">
              <div class="bench-baselines-label">BASELINES</div>
              ${r.baselines.map(b => {
                const faithBadge = b.faithfulness_score != null
                  ? `<span class="bench-baseline-faith ${b.faithfulness_score >= 0.4 ? 'faith-pass' : 'faith-fail'}">${(b.faithfulness_score * 100).toFixed(0)}%</span>`
                  : '';
                const latencyDelta = r.latency_ms > 0 ? ((b.latency_ms / r.latency_ms) * 100).toFixed(0) : '—';
                return `
                  <div class="bench-baseline-row">
                    <span class="bench-baseline-model">${esc(b.model)}</span>
                    <span class="bench-baseline-latency">${Math.round(b.latency_ms)}ms <span class="bench-baseline-delta">(${latencyDelta}%)</span></span>
                    ${faithBadge}
                    ${b.error ? `<span class="bench-baseline-error">${esc(b.error)}</span>` : ''}
                  </div>
                  ${b.response_text ? `
                    <details class="bench-baseline-details">
                      <summary class="bench-baseline-summary">${esc(b.response_text.substring(0, 120))}${b.response_text.length > 120 ? '…' : ''}</summary>
                      <div class="bench-baseline-full">${esc(b.response_text)}</div>
                    </details>` : ''}
                `;
              }).join('')}
            </div>
          ` : ''}
        </div>
      `;
    }
    html += `</div>`;
  }

  results.innerHTML = html;
}

// --- Stress Test ---

const STRESS_QUESTIONS = [
  // Toddler
  "Why is the sky blue?",
  "What sound does a cow make?",
  // Weather — simple
  "What's the weather in Tokyo?",
  // Irrelevant — casual
  "Tell me a joke about penguins",
  // Weather — compound with irrelevant
  "What's the weather in London and also what's the meaning of life?",
  // University level
  "Explain the second law of thermodynamics",
  // Weather — multi-city
  "Compare the weather in Sydney, Berlin, and Cairo right now",
  // Finance professional
  "What are the implications of an inverted yield curve on emerging market debt spreads?",
  // Weather + correction follow-up compound
  "What's the weather on Mars and also in Paris?",
  // Pure irrelevant — code request
  "Write me a Python function that sorts a linked list",
];

let _stressRunning = false;

async function runStressTest() {
  if (_stressRunning || state.isStreaming) return;
  _stressRunning = true;

  const btn = document.getElementById("stress-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  const shuffled = [...STRESS_QUESTIONS].sort(() => Math.random() - 0.5);
  const selected = shuffled.slice(0, 10);

  for (let i = 0; i < selected.length; i++) {
    btn.textContent = `${i + 1}/${selected.length}`;
    appendMessage("user", selected[i]);
    await streamChat(selected[i]);
    if (i < selected.length - 1) {
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  _stressRunning = false;
  btn.disabled = false;
  btn.textContent = "Stress Test";
}

// --- Init ---

document.addEventListener("DOMContentLoaded", async () => {
  cacheDOM();
  initScrollTracking();
  initTextareaResize();

  dom.emailInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSendCode();
  });

  dom.otpInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleVerify();
  });

  try {
    await fetch(`${API_BASE}/session/new`, { method: "POST", credentials: "include" });
  } catch (_) {}

  connectLogStream();
  startRedisPolling();
});
