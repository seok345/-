document.body.style.margin = "0";
document.body.style.fontFamily = "'Noto Sans KR', sans-serif";
document.body.style.height = "100vh";
document.body.style.display = "flex";
document.body.style.flexDirection = "column";
document.body.style.justifyContent = "center";
document.body.style.alignItems = "center";
document.body.style.backgroundColor = "#f0f0f0";

document.getElementById("chatContainer").style.backgroundColor = "#ffffff";
document.getElementById("chatContainer").style.borderRadius = "10px";
document.getElementById("chatContainer").style.boxShadow = "0 4px 10px rgba(0, 0, 0, 0.1)";
document.getElementById("chatContainer").style.width = "80%";
document.getElementById("chatContainer").style.maxWidth = "600px";
document.getElementById("chatContainer").style.marginBottom = "20px";
document.getElementById("chatContainer").style.padding = "20px";

document.getElementById("chatBox").style.height = "400px";
document.getElementById("chatBox").style.overflowY = "auto";
document.getElementById("chatBox").style.backgroundColor = "#f8f9fa";
document.getElementById("chatBox").style.borderRadius = "10px";
document.getElementById("chatBox").style.padding = "15px";

document.getElementById("inputArea").style.marginTop = "20px";
document.getElementById("inputArea").style.display = "flex";
document.getElementById("inputArea").style.alignItems = "center";
document.getElementById("inputArea").style.width = "80%";
document.getElementById("inputArea").style.maxWidth = "600px";

document.getElementById("userInput").style.padding = "10px";
document.getElementById("userInput").style.borderRadius = "10px";
document.getElementById("userInput").style.marginRight = "10px";
document.getElementById("userInput").style.flex = "1";

document.getElementById("sendBtn").style.padding = "10px 15px";
document.getElementById("sendBtn").style.borderRadius = "10px";
document.getElementById("sendBtn").style.backgroundColor = "#333333";
document.getElementById("sendBtn").style.color = "white";
document.getElementById("sendBtn").style.border = "none";
document.getElementById("sendBtn").style.cursor = "pointer";

document.getElementById("sendBtn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("responseTime").style.fontSize = "14px";
document.getElementById("responseTime").style.color = "#666";

function sendMessage() {
    const inputText = document.getElementById('userInput').value.trim();
    const chatBox = document.getElementById('chatBox');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const sendBtn = document.getElementById('sendBtn');
    const responseTime = document.getElementById('responseTime');


    chatBox.innerHTML = "";
    chatBox.innerHTML += `<div class="user-message">${inputText}</div>`;
    document.getElementById('userInput').value = '';
    error.style.display = "none";
    loading.style.display = "block";
    sendBtn.disabled = true;

    chatBox.scrollTop = chatBox.scrollHeight;

    const startTime = performance.now();

    $.ajax({
        url: 'http://localhost:8000/v1/chat/completions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            model: 'phi',
            messages: [{ role: 'user', content: inputText }]
        }),
        success: function(response) {
            const result = response.choices?.[0]?.message?.content || '응답이 없습니다.';
            const endTime = performance.now();
            const responseTimeText = ((endTime - startTime) / 1000).toFixed(1);
            chatBox.innerHTML += `<div class="bot-message">${result}</div>`;
            responseTime.textContent = `응답 시간: ${responseTimeText}s`;
            chatBox.scrollTop = chatBox.scrollHeight;
        },
        error: function(xhr, status, error) {
            error.textContent = "서버에 연결할 수 없습니다.";
            error.style.display = "block";
        },
        complete: function() {
            loading.style.display = "none";
            sendBtn.disabled = false;
        }
    });
}

