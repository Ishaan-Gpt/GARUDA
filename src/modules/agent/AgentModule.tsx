"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { usePlatform } from "@/context/PlatformContext";
import Markdown from "@/components/ui/Markdown";
import { getApiBase } from "@/lib/evidence";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
}

export default function AgentModule() {
  const router = useRouter();
  const { token, isBackendConnected } = usePlatform();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Load chat sessions from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("garuda_agent_sessions");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
          return;
        }
      } catch (e) {
        console.error("Error loading agent sessions:", e);
      }
    }
    
    // Create initial session if none exist
    const initialId = `session-${Date.now()}`;
    const initialSession: ChatSession = {
      id: initialId,
      title: "New Investigation Session",
      messages: [
        { role: "assistant", content: "Session Initialized. I am the **Garuda Core Console**, your interface coordinator for the enforcement registry. Query any vehicle, review citations, configure cameras, or navigate pages." }
      ],
      createdAt: new Date().toISOString()
    };
    setSessions([initialSession]);
    setActiveSessionId(initialId);
  }, []);

  // Save sessions to localStorage when updated
  const saveSessions = (updated: ChatSession[]) => {
    setSessions(updated);
    localStorage.setItem("garuda_agent_sessions", JSON.stringify(updated));
  };

  const createNewSession = () => {
    const newId = `session-${Date.now()}`;
    const newSession: ChatSession = {
      id: newId,
      title: `Investigation #${sessions.length + 1}`,
      messages: [
        { role: "assistant", content: "New session initialized. How can I assist you with security audits today?" }
      ],
      createdAt: new Date().toISOString()
    };
    const updated = [newSession, ...sessions];
    saveSessions(updated);
    setActiveSessionId(newId);
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter(s => s.id !== id);
    if (updated.length === 0) {
      const resetId = `session-${Date.now()}`;
      const resetSession = {
        id: resetId,
        title: "New Investigation Session",
        messages: [{ role: "assistant" as const, content: "Session Initialized." }],
        createdAt: new Date().toISOString()
      };
      saveSessions([resetSession]);
      setActiveSessionId(resetId);
    } else {
      saveSessions(updated);
      if (activeSessionId === id) {
        setActiveSessionId(updated[0].id);
      }
    }
  };

  const activeSession = sessions.find(s => s.id === activeSessionId);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [activeSession?.messages, loading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading || !activeSession) return;

    const userText = input;
    setInput("");

    // Add user message to active session
    const updatedMessages = [...activeSession.messages, { role: "user" as const, content: userText }];
    const updatedSessions = sessions.map(s => {
      if (s.id === activeSessionId) {
        // Auto update title if it was default
        const title = s.title.startsWith("New Investigation") || s.title.startsWith("Investigation #")
          ? (userText.length > 25 ? userText.slice(0, 25) + "..." : userText)
          : s.title;
        return { ...s, title, messages: updatedMessages };
      }
      return s;
    });
    saveSessions(updatedSessions);
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
          history: activeSession.messages.slice(-12).map(m => ({ role: m.role, content: m.content }))
        })
      });

      if (res.ok) {
        const data = await res.json();
        
        // Append response message
        const finalSessions = sessions.map(s => {
          if (s.id === activeSessionId) {
            return {
              ...s,
              messages: [...updatedMessages, { role: "assistant" as const, content: data.text }]
            };
          }
          return s;
        });
        saveSessions(finalSessions);

        // Execute UI navigation returned by Gemma
        if (data.ui_action) {
          const { type, path } = data.ui_action;
          if (type === "navigate" && path) {
            router.push(path);
          }
        }
      } else {
        const err = await res.json();
        appendErrorMessage(`⚠️ **Agent Error:** ${err.detail || "Failed to process query."}`, updatedMessages);
      }
    } catch (e) {
      appendErrorMessage(
        "⚠️ **Registry Core Database Error**\n\n" +
        "Garuda local database compiler interface could not be reached. " +
        "Verify that your local backend container is active and initialized.",
        updatedMessages
      );
    } finally {
      setLoading(false);
    }
  };

  const appendErrorMessage = (errText: string, currentMsgs: Message[]) => {
    const errorSessions = sessions.map(s => {
      if (s.id === activeSessionId) {
        return {
          ...s,
          messages: [...currentMsgs, { role: "assistant" as const, content: errText }]
        };
      }
      return s;
    });
    saveSessions(errorSessions);
  };

  // Quick Action suggestions
  const triggerSuggestion = (promptText: string) => {
    setInput(promptText);
  };

  const suggestions = [
    { label: "Check Repeat Offenders", prompt: "Search for repeat offenders in the system." },
    { label: "Disable Camera CAM-505", prompt: "Disable the school zone camera CAM-505." },
    { label: "Query Pending Reviews", prompt: "Show me the violations pending review." },
    { label: "Go to Camera Registry", prompt: "Navigate to the camera registry page." },
    { label: "Review Plate 9KX-452", prompt: "Search repeat offenders details for plate 9KX-452." }
  ];

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "200px 1fr 220px",
      gap: "12px",
      height: "calc(100vh - var(--header-height) - 40px)",
      minHeight: "450px"
    }}>
      
      {/* Sidebar: Multiple chat sessions list */}
      <div className="card" style={{ display: "flex", flexDirection: "column", padding: "10px", overflowY: "auto" }}>
        <button 
          onClick={createNewSession}
          className="btn btn-primary"
          style={{ width: "100%", fontSize: "11px", fontWeight: "bold", marginBottom: "12px" }}
        >
          + NEW INVESTIGATION
        </button>
        
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
          <div style={{ fontSize: "9px", fontWeight: "600", color: "var(--text-muted)", textTransform: "uppercase", marginBottom: "4px", paddingLeft: "4px" }}>
            Active Sessions
          </div>
          {sessions.map(s => {
            const isActive = s.id === activeSessionId;
            return (
              <div
                key={s.id}
                onClick={() => setActiveSessionId(s.id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "8px",
                  borderRadius: "4px",
                  backgroundColor: isActive ? "var(--bg-tertiary)" : "transparent",
                  border: isActive ? "1px solid var(--border-accent-dark)" : "1px solid transparent",
                  cursor: "pointer",
                  transition: "background 0.15s"
                }}
              >
                <span style={{ 
                  fontSize: "11px", 
                  fontWeight: isActive ? "700" : "500", 
                  color: isActive ? "var(--text-accent)" : "var(--text-secondary)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: 1,
                  marginRight: "4px"
                }}>
                  {s.title}
                </span>
                <button 
                  onClick={(e) => deleteSession(s.id, e)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--text-muted)",
                    cursor: "pointer",
                    fontSize: "10px"
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = "var(--danger)")}
                  onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
                >
                  ✕
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Panel: Message history feeds */}
      <div className="card" style={{ display: "flex", flexDirection: "column", padding: 0, overflow: "hidden" }}>
        
        {/* Active chat header */}
        <div style={{
          padding: "10px 12px",
          borderBottom: "1px solid var(--border-color)",
          backgroundColor: "#FAF9F6",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}>
          <div>
            <span style={{ fontWeight: "700", fontSize: "13px" }}>COMMAND QUERY CONSOLE</span>
            <span style={{ 
              fontSize: "9px", 
              marginLeft: "8px", 
              backgroundColor: isBackendConnected ? "var(--success-bg)" : "var(--danger-bg)",
              color: isBackendConnected ? "var(--success)" : "var(--danger)",
              padding: "1px 6px",
              borderRadius: "3px",
              fontWeight: "bold"
            }}>
              {isBackendConnected ? "REGISTRY COMPILER ACTIVE" : "RADAR SIMULATOR FALLBACK"}
            </span>
          </div>
          <span className="mono" style={{ fontSize: "10px", color: "var(--text-muted)" }}>
            {activeSessionId.slice(0, 15)}
          </span>
        </div>

        {/* Conversation List */}
        <div 
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "16px",
            display: "flex",
            flexDirection: "column",
            gap: "12px",
            backgroundColor: "#FCFCFA"
          }}
        >
          {activeSession?.messages.map((m, idx) => (
            <div 
              key={idx}
              style={{
                display: "flex",
                flexDirection: "column",
                alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                maxWidth: "80%",
                backgroundColor: m.role === "user" ? "var(--border-accent)" : "var(--bg-secondary)",
                color: m.role === "user" ? "var(--text-accent)" : "var(--text-primary)",
                border: m.role === "user" ? "1px solid var(--border-accent-dark)" : "1px solid var(--border-color)",
                borderRadius: "6px",
                padding: "10px 14px",
                boxShadow: "0 2px 4px rgba(0, 0, 0, 0.02)"
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
              padding: "10px 14px",
              color: "var(--text-muted)"
            }}>
              <span className="pulse-green" style={{ marginRight: "6px", width: "6px", height: "6px" }}></span>
              System is querying platform indexes...
            </div>
          )}
        </div>

        {/* Input submission bar */}
        <form 
          onSubmit={handleSendMessage}
          style={{
            padding: "12px",
            borderTop: "1px solid var(--border-color)",
            display: "flex",
            gap: "8px",
            backgroundColor: "#FFFFFF"
          }}
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your command or question (e.g. Search plate 9KX-452)..."
            disabled={loading}
            style={{
              flex: 1,
              padding: "8px 12px",
              border: "1px solid var(--border-color)",
              borderRadius: "4px",
              fontSize: "12px",
              outline: "none"
            }}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn"
            style={{
              padding: "8px 16px",
              backgroundColor: "var(--border-accent-dark)",
              color: "var(--text-accent)",
              border: "none",
              borderRadius: "4px",
              fontWeight: "bold",
              fontSize: "12px"
            }}
          >
            EXECUTE Command
          </button>
        </form>

      </div>

      {/* Right Column: Suggestion logs */}
      <div className="card" style={{ display: "flex", flexDirection: "column", gap: "10px", overflowY: "auto" }}>
        <div>
          <span style={{ fontWeight: "700", fontSize: "11px", textTransform: "uppercase", color: "var(--text-accent)" }}>
            Quick Prompts
          </span>
          <p style={{ fontSize: "9px", color: "var(--text-muted)", marginTop: "2px" }}>
            Tap suggestions to instantly populate the console input.
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "4px" }}>
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => triggerSuggestion(s.prompt)}
              style={{
                textAlign: "left",
                backgroundColor: "var(--bg-primary)",
                border: "1px solid var(--border-color)",
                borderRadius: "4px",
                padding: "8px",
                fontSize: "11px",
                cursor: "pointer",
                fontWeight: "500",
                color: "var(--text-secondary)",
                transition: "all 0.15s"
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--bg-tertiary)";
                e.currentTarget.style.borderColor = "var(--border-accent-dark)";
                e.currentTarget.style.color = "var(--text-accent)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "var(--bg-primary)";
                e.currentTarget.style.borderColor = "var(--border-color)";
                e.currentTarget.style.color = "var(--text-secondary)";
              }}
            >
              {s.label}
            </button>
          ))}
        </div>

        <div style={{
          marginTop: "auto",
          padding: "8px",
          backgroundColor: "#fef9c3",
          border: "1px dashed #fde047",
          borderRadius: "4px",
          fontSize: "9px",
          color: "#854d0e",
          lineHeight: "1.3"
        }}>
          <b>SYSTEM SAFEGUARDS:</b> The console interface prevents data alterations that compromise database integrity. All camera toggles and citation reviews update standard indexed schemas with full audit log tracking.
        </div>
      </div>

    </div>
  );
}
