"use client";

import React, { useState } from "react";
import { usePlatform } from "@/context/PlatformContext";

export default function AuthModule() {
  const { login, register } = usePlatform();
  const [isLogin, setIsLogin] = useState(true);
  
  // Form fields
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("Operator");

  // Notifications
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    setError(null);
    setLoading(true);

    try {
      if (isLogin) {
        const success = await login(email, password);
        if (!success) {
          setError("Failed to sign in. Verify email is active and credentials are correct.");
        }
      } else {
        const msg = await register(name, email, password, role);
        if (msg.includes("simulate") || msg.toLowerCase().includes("success")) {
          setMessage(msg);
          setIsLogin(true); // Switch to login to enter verified account
        } else {
          setError(msg);
        }
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "80vh",
      padding: "20px",
      backgroundColor: "var(--bg-primary)"
    }}>
      <div style={{
        width: "100%",
        maxWidth: "400px",
        backgroundColor: "var(--card-bg)",
        border: "1px solid var(--border-color)",
        borderRadius: "8px",
        boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.05)",
        overflow: "hidden"
      }}>
        {/* Header decoration */}
        <div style={{
          height: "4px",
          backgroundColor: "var(--border-accent-dark)"
        }}></div>

        <div style={{ padding: "28px" }}>
          {/* Logo Title */}
          <div style={{ textAlign: "center", marginBottom: "24px" }}>
            <img 
              src="/logo.png" 
              alt="Garuda Logo" 
              style={{
                width: "48px",
                height: "48px",
                objectFit: "contain",
                marginBottom: "8px"
              }} 
            />
            <h2 style={{ fontSize: "18px", fontWeight: "700", letterSpacing: "-0.5px", margin: 0 }}>
              {isLogin ? "GARUDA CONTROL PORTAL" : "CREATE ENFORCEMENT ACCOUNT"}
            </h2>
            <p style={{ fontSize: "10px", color: "var(--text-muted)", textTransform: "uppercase", marginTop: "2px" }}>
              {isLogin ? "Enter security credentials to log in" : "Register a new platform officer identity"}
            </p>
          </div>

          {/* Success / Error Alerts */}
          {message && (
            <div style={{
              backgroundColor: "#fef9c3",
              border: "1px solid #fef08a",
              color: "#854d0e",
              padding: "10px 12px",
              borderRadius: "4px",
              fontSize: "11px",
              marginBottom: "16px",
              lineHeight: "1.4"
            }}>
              <b>Registration Successful:</b> {message}
            </div>
          )}

          {error && (
            <div style={{
              backgroundColor: "#fee2e2",
              border: "1px solid #fecaca",
              color: "#991b1b",
              padding: "10px 12px",
              borderRadius: "4px",
              fontSize: "11px",
              marginBottom: "16px",
              lineHeight: "1.4"
            }}>
              <b>Error:</b> {error}
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {!isLogin && (
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "10px", fontWeight: "600", color: "var(--text-secondary)" }}>FULL NAME</label>
                <input 
                  type="text" 
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  required 
                  placeholder="e.g. Officer Keshav"
                  style={{
                    padding: "8px 12px",
                    border: "1px solid var(--border-color)",
                    borderRadius: "4px",
                    fontSize: "12px",
                    outline: "none"
                  }}
                />
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label style={{ fontSize: "10px", fontWeight: "600", color: "var(--text-secondary)" }}>EMAIL ADDRESS</label>
              <input 
                type="email" 
                value={email} 
                onChange={(e) => setEmail(e.target.value)} 
                required 
                placeholder="e.g. keshav@enforcement.gov"
                style={{
                  padding: "8px 12px",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  fontSize: "12px",
                  outline: "none"
                }}
              />
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
              <label style={{ fontSize: "10px", fontWeight: "600", color: "var(--text-secondary)" }}>PASSWORD</label>
              <input 
                type="password" 
                value={password} 
                onChange={(e) => setPassword(e.target.value)} 
                required 
                placeholder="••••••••"
                style={{
                  padding: "8px 12px",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  fontSize: "12px",
                  outline: "none"
                }}
              />
            </div>

            {!isLogin && (
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "10px", fontWeight: "600", color: "var(--text-secondary)" }}>PLATFORM ROLE</label>
                <select 
                  value={role} 
                  onChange={(e) => setRole(e.target.value)}
                  style={{
                    padding: "8px 12px",
                    border: "1px solid var(--border-color)",
                    borderRadius: "4px",
                    fontSize: "12px",
                    outline: "none",
                    backgroundColor: "#FFFFFF"
                  }}
                >
                  <option value="Operator">Operator (Control Room)</option>
                  <option value="Reviewer">Reviewer (Enforcement Officer)</option>
                  <option value="Supervisor">Supervisor</option>
                  <option value="Admin">System Administrator</option>
                </select>
              </div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              className="btn"
              style={{
                marginTop: "8px",
                padding: "10px",
                backgroundColor: "var(--border-accent-dark)",
                color: "var(--text-accent)",
                border: "none",
                borderRadius: "4px",
                fontWeight: "bold",
                fontSize: "12px",
                cursor: loading ? "not-allowed" : "pointer"
              }}
            >
              {loading ? "PROCESSING AUTH SECURELY..." : (isLogin ? "SIGN IN" : "REGISTER OFFICER")}
            </button>
          </form>

          {/* Switch links */}
          <div style={{ textAlign: "center", marginTop: "18px" }}>
            <button 
              onClick={() => {
                setIsLogin(!isLogin);
                setError(null);
                setMessage(null);
              }}
              style={{
                background: "none",
                border: "none",
                color: "var(--text-accent)",
                fontSize: "11px",
                fontWeight: "600",
                cursor: "pointer",
                textDecoration: "underline"
              }}
            >
              {isLogin ? "Create an enforcement officer account instead" : "Have an active account? Sign In"}
            </button>
          </div>

          {/* Test Mode & Autofill */}
          <div style={{
            marginTop: "20px",
            padding: "12px",
            backgroundColor: "var(--bg-secondary)",
            borderRadius: "6px",
            border: "1px solid var(--border-color)",
            display: "flex",
            flexDirection: "column",
            gap: "8px"
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "10px", fontWeight: "700", color: "var(--text-accent)", letterSpacing: "0.5px" }}>
                ⚙️ TEST MODE ACTIVE
              </span>
            </div>
            <div style={{ display: "flex", gap: "6px" }}>
              <button
                type="button"
                onClick={() => {
                  setEmail("amit@controlroom.gov");
                  setPassword("amit123");
                }}
                style={{
                  flex: 1,
                  padding: "6px",
                  fontSize: "10px",
                  fontWeight: "600",
                  backgroundColor: "var(--card-bg)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "4px",
                  cursor: "pointer",
                  color: "var(--text-primary)"
                }}
              >
                Autofill Operator
              </button>
              <button
                type="button"
                onClick={() => {
                  setEmail("keshav@enforcement.gov");
                  setPassword("admin123");
                }}
                style={{
                  flex: 1,
                  padding: "6px",
                  fontSize: "10px",
                  fontWeight: "600",
                  backgroundColor: "var(--border-accent-dark)",
                  border: "1px solid var(--border-accent-dark)",
                  borderRadius: "4px",
                  cursor: "pointer",
                  color: "var(--text-accent)"
                }}
              >
                Autofill Admin
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
