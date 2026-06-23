"use client";

import React, { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { usePlatform } from "@/context/PlatformContext";
import Markdown from "./Markdown";
import { getApiBase } from "@/lib/evidence";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function FloatingAgent() {
  const router = useRouter();
  const { token, isBackendConnected } = usePlatform();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi! I am the **Gemma Security Copilot**. Ask me to search plates, toggle cameras, approve citations, or show analytics." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input;
    setInput("");
    const newMessages = [...messages, { role: "user" as const, content: userText }];
    setMessages(newMessages);
    setLoading(true);

    try {
      const res = await fetch(`${getApiBase()}/agent/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token || localStorage.getItem("garuda_token")}`
        },
        body: JSON.stringify({
          message: userText,
          history: messages.slice(-10).map(m => ({ role: m.role, content: m.content }))
        })
      });

      if (res.ok) {
        const data = await res.json();
        setMessages(prev => [...prev, { role: "assistant", content: data.text }]);
        
        // Execute UI Actions returned by Gemma Agent
        if (data.ui_action) {
          const { type, path } = data.ui_action;
          if (type === "navigate" && path) {
            router.push(path);
          }
        }
      } else {
        const err = await res.json();
        setMessages(prev => [...prev, { role: "assistant", content: `⚠️ **Agent Error:** ${err.detail || "Failed to process prompt."}` }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: (
          "⚠️ **AI Pipeline Offline**\n\n" +
          "Ollama server or local `gemma3:1b` model could not be reached. " +
          "Make sure Ollama is running on your machine and you have executed:\n" +
          "```bash\n" +
          "ollama run gemma3:1b\n" +
          "```"
        )
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        onClick={() => setOpen(!open)}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          backgroundColor: "var(--border-accent-dark)",
          color: "var(--text-accent)",
          border: "none",
          boxShadow: "0 4px 10px rgba(0, 0, 0, 0.15)",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
          transition: "transform 0.2s"
        }}
        onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.05)")}
        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
      >
        {open ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        )}
      </button>

      {/* Slide-out Floating Chat Window */}
      {open && (
        <div 
          className="floating-chat-window"
          style={{
            position: "fixed",
            bottom: "80px",
            right: "20px",
            width: "360px",
            height: "480px",
            backgroundColor: "var(--card-bg, #FFFFFF)",
            border: "1px solid var(--border-color)",
            borderRadius: "8px",
            boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)",
            zIndex: 1000,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            animation: "slideIn 0.2s ease-out"
          }}
        >
          {/* Header */}
          <div style={{
            padding: "12px",
            backgroundColor: "var(--border-accent)",
            borderBottom: "1px solid var(--border-accent-dark)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <div style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                backgroundColor: isBackendConnected ? "#22c55e" : "#ef4444"
              }}></div>
              <span style={{ fontWeight: "700", fontSize: "12px", color: "var(--text-accent)" }}>
                GEMMA SECURITY COPILOT
              </span>
            </div>
            <button 
              onClick={() => setOpen(false)}
              style={{ background: "none", border: "none", cursor: "pointer", fontSize: "12px" }}
            >
              ✕
            </button>
          </div>

          {/* Messages area */}
          <div 
            ref={scrollRef}
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "12px",
              display: "flex",
              flexDirection: "column",
              gap: "10px",
              backgroundColor: "#FCFCFA"
            }}
          >
            {messages.map((m, idx) => (
              <div 
                key={idx}
                style={{
                  alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  backgroundColor: m.role === "user" ? "var(--border-accent)" : "var(--bg-secondary)",
                  color: m.role === "user" ? "var(--text-accent)" : "var(--text-primary)",
                  border: m.role === "user" ? "1px solid var(--border-accent-dark)" : "1px solid var(--border-color)",
                  borderRadius: "6px",
                  padding: "8px 10px",
                  wordBreak: "break-word"
                }}
              >
                <Markdown content={m.content} />
              </div>
            ))}
            {loading && (
              <div style={{
                alignSelf: "flex-start",
                backgroundColor: "var(--bg-secondary)",
                border: "1px solid var(--border-color)",
                borderRadius: "6px",
                padding: "8px 10px",
                color: "var(--text-muted)",
                fontSize: "11px"
              }}>
                <span className="pulse-green" style={{ marginRight: "6px", width: "6px", height: "6px" }}></span>
                Gemma thinking...
              </div>
            )}
          </div>

          {/* Input Form */}
          <form 
            onSubmit={handleSendMessage}
            style={{
              padding: "10px",
              borderTop: "1px solid var(--border-color)",
              display: "flex",
              gap: "6px",
              backgroundColor: "#FFFFFF"
            }}
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g. Show active cameras..."
              disabled={loading}
              style={{
                flex: 1,
                padding: "6px 10px",
                border: "1px solid var(--border-color)",
                borderRadius: "4px",
                fontSize: "12px",
                outline: "none"
              }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              style={{
                padding: "6px 12px",
                backgroundColor: "var(--border-accent-dark)",
                color: "var(--text-accent)",
                border: "none",
                borderRadius: "4px",
                fontWeight: "bold",
                fontSize: "12px",
                cursor: "pointer"
              }}
            >
              SEND
            </button>
          </form>
        </div>
      )}
    </>
  );
}
