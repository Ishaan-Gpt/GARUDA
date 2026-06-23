"use client";
 
import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { usePlatform, UserRole } from "@/context/PlatformContext";
import AuthModule from "@/modules/auth/AuthModule";
import { 
  HomeIcon, 
  CameraIcon, 
  AlertIcon, 
  UploadIcon, 
  ShieldIcon,
  ChartIcon,
  SettingsIcon, 
  BellIcon,
  CloseIcon,
  CheckIcon
} from "@/components/Icons";
 
interface LayoutShellProps {
  children: React.ReactNode;
}
 
export default function LayoutShell({ children }: LayoutShellProps) {
  const pathname = usePathname() || "";
  const { 
    role, 
    setRole, 
    cameras, 
    notifications, 
    markNotificationRead, 
    clearNotifications,
    isAuthenticated,
    authLoading,
    user,
    logout,
    isBackendConnected
  } = usePlatform();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
 
  React.useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);
 
  // Nav items setup
  const menuItems = [
    { name: "Dashboard", href: "/dashboard", icon: <HomeIcon size={16} /> },
    { name: "Cameras", href: "/cameras", icon: <CameraIcon size={16} /> },
    { name: "Violations", href: "/violations", icon: <AlertIcon size={16} /> },
    { name: "Human Check", href: "/review", icon: <ShieldIcon size={16} /> },
    { name: "Upload Evidence", href: "/evidence", icon: <UploadIcon size={16} /> },
    { name: "Analytics", href: "/analytics", icon: <ChartIcon size={16} /> },
    { name: "Settings", href: "/settings", icon: <SettingsIcon size={16} /> },
  ];

  // Derive stats
  const activeCameras = cameras.filter(c => c.status === "Active").length;
  const unreadNotifications = notifications.filter(n => !n.read).length;

  const handleRoleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setRole(e.target.value as UserRole);
  };

  const isPublicRoute = pathname.startsWith("/public");

  if (authLoading) {
    if (isPublicRoute) {
      return null;
    }
    return (
      <div style={{ display: "flex", alignItems: "center", justifyItems: "center", justifyContent: "center", height: "100vh", backgroundColor: "var(--bg-primary)" }}>
        <div style={{ color: "var(--text-accent)", fontWeight: "bold", fontSize: "14px", fontFamily: "var(--font-mono)" }}>
          INITIALIZING GARUDA SECURE SYSTEM...
        </div>
      </div>
    );
  }

  if (!isAuthenticated && !isPublicRoute) {
    return <AuthModule />;
  }

  if (isPublicRoute) {
    return (
      <div className="public-container" style={{ minHeight: "100vh", backgroundColor: "var(--bg-primary)" }}>
        {children}
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Backdrop overlay for mobile */}
      <div className={`sidebar-overlay ${mobileMenuOpen ? "active" : ""}`} onClick={() => setMobileMenuOpen(false)}></div>
      
      {/* Sidebar Layout */}
      <aside className={`app-sidebar ${mobileMenuOpen ? "mobile-open" : ""}`}>
        <div style={{ padding: "16px", borderBottom: "1px solid var(--border-color)", background: "#FCFCFC" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <img 
              src="/logo.png" 
              alt="Garuda Logo" 
              style={{
                width: "28px",
                height: "28px",
                objectFit: "contain"
              }} 
            />
            <div>
              <div style={{ fontSize: "14px", fontWeight: "700", letterSpacing: "0.5px" }}>GARUDA</div>
              <div style={{ fontSize: "9px", color: "var(--text-muted)", textTransform: "uppercase" }}>Violation Intel</div>
            </div>
          </div>
        </div>

        <nav style={{ flex: 1, overflowY: "auto" }}>
          <ul className="sidebar-menu">
            {menuItems.map((item) => {
              const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
              return (
                <li key={item.name}>
                  <Link href={item.href} className={`sidebar-item ${isActive ? "active" : ""}`}>
                    {item.icon}
                    <span>{item.name}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="sidebar-footer">
          <div className="system-health">
            <div style={{ fontWeight: "600", fontSize: "10px", textTransform: "uppercase", marginBottom: "6px", color: "var(--text-accent)" }}>
              System Health
            </div>
            <div className="health-item">
              <span>Streams Status:</span>
              <span style={{ display: "flex", alignItems: "center", gap: "4px", fontWeight: "bold" }}>
                <span className="pulse-green"></span> {activeCameras}/{cameras.length} UP
              </span>
            </div>
            <div className="health-item">
              <span>Processor Latency:</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: "500" }}>32ms</span>
            </div>
            <div className="health-item">
              <span>FPS Aggregated:</span>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: "500" }}>{activeCameras * 30} Hz</span>
            </div>
          </div>
          <div style={{ textAlign: "center", marginTop: "4px" }}>
            Garuda Agentic Platform v1.2
          </div>
        </div>
      </aside>

      {/* Main Page Content */}
      <div className="main-layout">
        <header className="app-header">
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {/* Mobile Hamburger menu */}
            <button className="hamburger-btn" onClick={() => setMobileMenuOpen(true)}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            </button>
            <div className={isBackendConnected ? "pulse-green" : "pulse-red"}></div>
            <div style={{ fontSize: "11px", fontWeight: "600", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.5px" }}>
              Enforcement Network: {isBackendConnected ? "Online" : "Offline"}
            </div>
          </div>

          <div className="header-actions">
            {/* User profile details & Logout */}
            {user && (
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                  <span style={{ fontSize: "11px", fontWeight: "700", color: "var(--text-primary)" }}>{user.name}</span>
                  <span className={`badge ${role === "Admin" ? "approved" : (role === "Supervisor" ? "escalated" : (role === "Reviewer" ? "review" : "detected"))}`} style={{ fontSize: "8px", padding: "1px 4px", marginTop: "1px" }}>
                    {role.toUpperCase()}
                  </span>
                </div>
                <button 
                  onClick={logout}
                  style={{
                    backgroundColor: "transparent",
                    border: "1px solid var(--border-color)",
                    borderRadius: "4px",
                    color: "var(--danger)",
                    fontSize: "10px",
                    fontWeight: "bold",
                    padding: "4px 8px",
                    cursor: "pointer",
                    transition: "all 0.2s"
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#fee2e2")}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                >
                  LOGOUT
                </button>
              </div>
            )}

            {/* Notification Bell trigger */}
            <button 
              onClick={() => setDrawerOpen(true)}
              style={{
                position: "relative",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "6px",
                borderRadius: "4px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "background 0.2s"
              }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "var(--bg-tertiary)")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              <BellIcon size={18} style={{ color: "var(--text-secondary)" }} />
              {unreadNotifications > 0 && (
                <span style={{
                  position: "absolute",
                  top: "2px",
                  right: "2px",
                  backgroundColor: "var(--danger)",
                  color: "#FFFFFF",
                  fontSize: "9px",
                  fontWeight: "bold",
                  borderRadius: "50%",
                  width: "14px",
                  height: "14px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center"
                }}>
                  {unreadNotifications}
                </span>
              )}
            </button>
          </div>
        </header>

        {/* Content Render area */}
        <main className="content-wrapper">
          {children}
        </main>
      </div>

      {/* Notification Sliding Drawer overlay */}
      <div className={`notification-drawer ${drawerOpen ? "open" : ""}`}>
        <div className="drawer-header">
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <BellIcon size={16} style={{ color: "var(--text-accent)" }} />
            <h3 style={{ fontSize: "13px", fontWeight: "600", color: "var(--text-accent)" }}>Notification Feed</h3>
          </div>
          <button 
            onClick={() => setDrawerOpen(false)}
            style={{ background: "none", border: "none", cursor: "pointer", padding: "2px" }}
          >
            <CloseIcon size={16} />
          </button>
        </div>

        <div className="drawer-content">
          {notifications.length === 0 ? (
            <div style={{ textAlign: "center", padding: "40px 10px", color: "var(--text-muted)" }}>
              No notifications logs active.
            </div>
          ) : (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px", padding: "0 4px" }}>
                <span style={{ fontSize: "10px", color: "var(--text-muted)" }}>{notifications.length} alerts</span>
                <button 
                  onClick={clearNotifications}
                  style={{ background: "none", border: "none", color: "var(--danger)", fontSize: "10px", fontWeight: "600", cursor: "pointer" }}
                >
                  Clear All
                </button>
              </div>

              {notifications.map((n) => (
                <div key={n.id} className={`notification-item ${n.read ? "" : "unread"} ${n.severity}`}>
                  <div className="notification-item-header">
                    <span style={{ textTransform: "uppercase", color: n.severity === "high" ? "var(--danger)" : "var(--text-secondary)" }}>{n.type}</span>
                    <span>{new Date(n.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
                  </div>
                  <p style={{ fontSize: "11px", margin: "4px 0", color: "var(--text-primary)" }}>{n.message}</p>
                  {!n.read && (
                    <button 
                      onClick={() => markNotificationRead(n.id)}
                      style={{ 
                        alignSelf: "flex-end", 
                        background: "none", 
                        border: "none", 
                        color: "var(--text-accent)", 
                        fontSize: "9px", 
                        fontWeight: "700", 
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "2px"
                      }}
                    >
                      <CheckIcon size={10} /> Mark Read
                    </button>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
