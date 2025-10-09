import React, { useEffect, useRef, useState } from "react";
import "./index.css";

const SAMPLE = [
  "Halo! Aku Friday, asisten futuristikmu. Mau coba fitur apa hari ini?",
  "Coba ketik: 'show me demo' atau 'cuaca' untuk melihat fitur.",
  "Ingin integrasi API? Aku siap dipanggil lewat backend."
];

function useParticles(canvasRef) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let w = (canvas.width = innerWidth);
    let h = (canvas.height = innerHeight);
    let stopped = false;

    const particles = [];
    const createParticle = () => {
      const x = Math.random() * w;
      return {
        x,
        y: -10 - Math.random() * 200,
        size: 0.8 + Math.random() * 2.2,
        speedY: 0.2 + Math.random() * 0.6,
        length: 14 + Math.random() * 30,
        alpha: 0.06 + Math.random() * 0.14
      };
    };

    for (let i = 0; i < Math.floor(w / 40); i++) particles.push(createParticle());

    const onResize = () => {
      w = canvas.width = innerWidth;
      h = canvas.height = innerHeight;
    };
    window.addEventListener("resize", onResize);

    function frame() {
      if (stopped) return;
      ctx.clearRect(0, 0, w, h);

      // soft vignette gradient
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "rgba(0,0,0,0)");
      g.addColorStop(1, "rgba(0,0,0,0.18)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);

      // particles (meteor-like falling)
      particles.forEach((p, idx) => {
        p.y += p.speedY;
        p.x += Math.sin(p.y * 0.01) * 0.3;
        if (p.y - p.length > h) {
          particles[idx] = createParticle();
          particles[idx].y = -10;
        }
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.length * 0.35, p.y - p.length);
        ctx.strokeStyle = `rgba(255,255,255,${p.alpha})`;
        ctx.lineWidth = p.size;
        ctx.lineCap = "round";
        ctx.stroke();
      });

      requestAnimationFrame(frame);
    }
    frame();

    return () => {
      stopped = true;
      window.removeEventListener("resize", onResize);
    };
  }, [canvasRef]);
}

export default function App() {
  const canvasRef = useRef(null);
  useParticles(canvasRef);

  const [messages, setMessages] = useState([
    { id: 1, who: "bot", text: SAMPLE[0], time: timeNow() },
    { id: 2, who: "bot", text: SAMPLE[1], time: timeNow() }
  ]);
  const [value, setValue] = useState("");
  const [typing, setTyping] = useState(false);
  const messagesRef = useRef();

  useEffect(() => {
    // auto-scroll on new message
    if (!messagesRef.current) return;
    messagesRef.current.scrollTop = messagesRef.current.scrollHeight + 200;
  }, [messages]);

  function timeNow() {
    const d = new Date();
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
  }

  async function sendMessage(text) {
    if (!text.trim()) return;
    const userMsg = { id: Date.now(), who: "user", text: text.trim(), time: timeNow() };
    setMessages((s) => [...s, userMsg]);
    setValue("");
    setTyping(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.trim() })
      });
      const data = await res.json();
      enqueueBot(data.reply);
    } catch (err) {
      enqueueBot("⚠️ Gagal menghubungi backend.");
    }

    setTimeout(() => {
      const reply = SAMPLE[Math.floor(Math.random() * SAMPLE.length)];
      enqueueBot(reply);
    }, 900 + Math.min(text.length * 10, 1200));
  }

  function enqueueBot(txt) {
    setTyping(false);
    setMessages((s) => [...s, { id: Date.now() + 1, who: "bot", text: txt, time: timeNow() }]);
  }

  return (
    <div className="app-wrap">
      <canvas ref={canvasRef} className="bg-layer" />
      <div className="chat-shell" role="main" aria-label="Chat interface">
        {/* LEFT PANEL */}
        <aside className="left-panel" aria-hidden>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div className="orb" aria-hidden>
              {/* small decorative inner orb */}
              <svg width="40" height="40" viewBox="0 0 24 24" style={{ filter: "drop-shadow(0 4px 12px rgba(0, 255, 255, 0.06))" }}>
                <defs>
                  <linearGradient id="g" x1="0" x2="1">
                    <stop offset="0" stopColor="#00FFFF" stopOpacity="0.9"></stop>
                    <stop offset="1" stopColor="#FF00FF" stopOpacity="0.7"></stop>
                  </linearGradient>
                </defs>
                <circle cx="12" cy="12" r="9" fill="url(#g)" fillOpacity="0.18"></circle>
                <circle cx="9" cy="9" r="2.2" fill="#fff" fillOpacity="0.56"></circle>
              </svg>
            </div>
            <div>
              <div className="app-title">Friday • Futuristic AI</div>
              <div className="app-sub">Glassmorphism chat • minimal & calm</div>
            </div>
          </div>

          <div style={{ marginTop: 8 }}>
            <div className="small">Shortcuts</div>
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={() => { setMessages([{ id: Date.now(), who: "bot", text: "Chat reset. Selamat mencoba!", time: timeNow() }]); }}>Reset</button>
              <button className="btn" onClick={() => { setMessages([]); }}>Clear</button>
            </div>
          </div>

          <div style={{ marginTop: 12, fontSize: 13, color: "var(--muted)" }}>
            Tips: gunakan perintah sederhana seperti <code>halo</code>, <code>cuaca</code>, atau <code>show me demo</code>.
          </div>

          <div className="actions" style={{ marginTop: 12 }}>
            <div className="small">v1.0 • Local Demo</div>
          </div>
        </aside>

        {/* CHAT AREA */}
        <section className="chat-area" aria-live="polite">
          <div className="chat-header">
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div className="avatar" aria-hidden>
                <svg width="18" height="18" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.12)" strokeWidth="0.8" fill="none" /></svg>
              </div>
              <div>
                <div className="title">Friday</div>
                <div className="status">Online • Powered by Gemini (demo)</div>
              </div>
            </div>
            <div style={{ marginLeft: "auto", fontSize: 13, color: "var(--muted)" }}>{new Date().toLocaleDateString()}</div>
          </div>

          <div ref={messagesRef} className="messages" role="log">
            {messages.map((m) => (
              <div key={m.id} style={{ display: "flex", alignItems: "flex-end", gap: 12, justifyContent: m.who === "user" ? "flex-end" : "flex-start" }}>
                {m.who === "bot" && <div className="avatar" aria-hidden></div>}
                <div className={`msg ${m.who === "user" ? "user" : "bot"}`} role="article">
                  <div>{m.text}</div>
                  <div className="meta">
                    <div className="small">{m.who === "user" ? "You" : "Friday"}</div>
                    <div className="small" aria-hidden>{m.time}</div>
                  </div>
                </div>
                {m.who === "user" && <div className="avatar" aria-hidden></div>}
              </div>
            ))}

            {typing && (
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <div className="avatar" aria-hidden></div>
                <div className="typing">
                  <div className="dot" />
                  <div className="dot" />
                  <div className="dot" />
                </div>
              </div>
            )}
          </div>

          <div className="composer">
            <textarea
              aria-label="Type a message"
              className="input"
              rows={1}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(value); } }}
              placeholder="Tulis pesan... (Enter untuk kirim)"
            />
            <button className="send-btn" onClick={() => sendMessage(value)} aria-label="Send message">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13" stroke="white" strokeWidth="1.1" strokeLinecap="round"/><path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" strokeWidth="0.7"/></svg>
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
