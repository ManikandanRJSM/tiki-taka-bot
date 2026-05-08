const chatMessages  = document.getElementById('chatMessages');
const userInput     = document.getElementById('userInput');
const sendBtn       = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');

// Set welcome message timestamp
document.getElementById('welcomeTime').textContent = formatTime(new Date());

// ── Suggestion chips ──────────────────────────────────────────
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    userInput.value = chip.dataset.text;
    userInput.focus();
    autoResize();
  });
});

// ── Auto-resize textarea ──────────────────────────────────────
function autoResize() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
}
userInput.addEventListener('input', autoResize);

// ── Send on Enter (Shift+Enter = newline) ─────────────────────
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

// ── Core send logic ───────────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  appendMessage('user', text);
  userInput.value = '';
  userInput.style.height = 'auto';
  setLoading(true);

  try {
    const res  = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      appendMessage('bot', data.error || 'Something went wrong. Please try again.', true);
    } else {
      appendMessage('bot', data.reply);
    }
  } catch {
    appendMessage('bot', 'Could not reach the server. Is the Flask app running?', true);
  } finally {
    setLoading(false);
  }
}

// ── Append a message bubble ───────────────────────────────────
function appendMessage(role, text, isError = false) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role === 'bot' ? 'bot-message' : 'user-message'}`;

  const time = formatTime(new Date());

  if (role === 'bot') {
    wrapper.innerHTML = `
      <div class="avatar">&#9917;</div>
      <div class="bubble ${isError ? 'error-bubble' : ''}">
        <p>${escapeHtml(text)}</p>
        <span class="msg-time">${time}</span>
      </div>`;
  } else {
    wrapper.innerHTML = `
      <div class="bubble">
        <p>${escapeHtml(text)}</p>
        <span class="msg-time">${time}</span>
      </div>`;
  }

  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

// ── Typing / loading state ─────────────────────────────────────
function setLoading(on) {
  sendBtn.disabled = on;
  userInput.disabled = on;
  typingIndicator.classList.toggle('visible', on);
  if (on) scrollToBottom();
}

// ── Helpers ───────────────────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}
