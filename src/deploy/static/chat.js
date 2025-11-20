const storyId = window.storyData?.id || "";
const storyContent = window.storyData?.story || "";
const chatId = window.storyData?.chat_id || "";

const chatWindow = document.getElementById("chat-window");
const chatInput = document.getElementById('chat-input');
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';            // reset
  chatInput.style.height = chatInput.scrollHeight + 'px'; // expand
});

function appendMessage(sender, text) {
  const placeholder = document.getElementById("chat-placeholder");
  if (placeholder) placeholder.remove();

  const msg = document.createElement("div");
  msg.className = sender === "user" ? "user-message" : "bot-message";
  msg.textContent = text;
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function handleSend() {
  console.log("Send triggered");

  const text = chatInput.value.trim();
  if (!text) return;
  appendMessage("user", text);
  chatInput.value = "";

  // thinking bubble
  const thinkingMsg = document.createElement("div");
  thinkingMsg.className = "bot-message thinking-bubble";
  const label = document.createElement("span");
  label.textContent = "Paperboy is typing";
  const dotSpan = document.createElement("span");
  dotSpan.className = "dots";
  thinkingMsg.appendChild(label);
  thinkingMsg.appendChild(dotSpan);
  chatWindow.appendChild(thinkingMsg);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  console.log("Thinking bubble created");

  let dotCount = 0;
  const interval = setInterval(() => {
    dotCount = (dotCount + 1) % 4;
    dotSpan.textContent = ".".repeat(dotCount);
  }, 400);

  const res = await fetch(`/story/${storyId}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      context: storyContent,
      chat_id: chatId
    }),
  });

  const data = await res.json();
  clearInterval(interval);
  thinkingMsg.classList.add("fade-out");

  setTimeout(() => {
    thinkingMsg.remove();
    appendMessage("bot", data.reply);
  }, 400);
}

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
