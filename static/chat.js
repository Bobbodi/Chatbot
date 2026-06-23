const BOT_AVATAR = `<i class="fa-solid fa-user-tie" style="margin-top: 14px;"></i>`;

const USER_AVATAR = `<i class="fa-solid fa-user" style="margin-top: 14px;"></i>`;

const greetingMessage = "Hello! I'm your assistant. How can I help you?";

async function initChat() {
    const chatBox = document.getElementById("chat-box");

    chatBox.innerHTML += `
        <div class="message bot">
            ${BOT_AVATAR}
            <div class="bubble">
                ${greetingMessage}
            </div>
        </div>
    `;
}

window.addEventListener("DOMContentLoaded", initChat);

async function sendMessage() {

    const chatBox =
        document.getElementById("chat-box");

    const input =
        document.getElementById("user-input");

    const message = input.value.trim();

    if (!message) return;

    chatBox.innerHTML +=
        `<div class="message user">
            ${USER_AVATAR}
            <div class="bubble">${escapeHtml(message)}</div>
        </div>`;

    input.value = "";
    input.disabled = true;
    document.getElementById("send-btn").disabled = true;

    // Show "thinking" indicator with animated dots
    const thinkingId = "thinking-" + Date.now();
    chatBox.innerHTML += `
        <div class="message bot" id="${thinkingId}">
            ${BOT_AVATAR}
            <div class="bubble thinking">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
        </div>
    `;
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message
            })
        });

        const data = await response.json();

        let sourceHtml = "";

        for (const source of data.sources) {

            sourceHtml += `
                <div class="source-chunk">
                    <div class="source-distance">
                        <b>Distance:</b>
                        ${source.distance.toFixed(4)}
                    </div>

                    <div class="source-text">
                        ${source.text}
                    </div>
                </div>
            `;
        }

        // Replace thinking indicator with the actual response
        document.getElementById(thinkingId).outerHTML = `
            <div class="message bot">
                ${BOT_AVATAR}
                <div class="bubble">
                    ${marked.parse(data.response)}

                    <details class="sources">
                        <summary>Retrieved Chunks</summary>
                        ${sourceHtml}
                    </details>
                </div>
            </div>
        `;

    } catch (err) {
        document.getElementById(thinkingId).outerHTML = `
            <div class="message bot">
                ${BOT_AVATAR}
                <div class="bubble error">
                    Something went wrong. Please try again.
                </div>
            </div>
        `;
    } finally {
        input.disabled = false;
        document.getElementById("send-btn").disabled = false;
        input.focus();
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

document.getElementById("user-input").addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
        sendMessage();
    }
});

document.getElementById("send-btn").addEventListener("click", sendMessage);