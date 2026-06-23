"""
GARUDA — Gemma Agentic Executor Core
======================================
Integrates local gemma3:1b model with SQLite database operations,
enforces safety guardrails, executes SQL tools, and provides
structured UI commands back to the frontend.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .database import AsyncSessionLocal, CameraModel, VehicleModel, ViolationModel, UserModel, AuditLogModel, update_violation_status

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

OLLAMA_URL = f"{settings.OLLAMA_URL.rstrip('/')}/api/chat"
MODEL_NAME = "gemma3:1b"


# ---------------------------------------------------------------------------
# Safety Guardrails
# ---------------------------------------------------------------------------

BLOCKED_KEYWORDS = ["drop", "delete", "truncate", "alter", "destroy", "insert into", "grant", "revoke"]

def check_query_safety(query: str) -> Tuple[bool, str]:
    """Check if the user input contains potentially destructive commands."""
    query_lower = query.lower()
    for kw in BLOCKED_KEYWORDS:
        if kw in query_lower:
            return False, f"Guardrail Alert: The action containing '{kw}' is blocked to protect database integrity."
    return True, ""


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

async def build_system_context(session: AsyncSession) -> str:
    """Query current database counts to feed system knowledge to the Gemma context."""
    try:
        # Total counts
        total_violations = (await session.execute(select(func.count(ViolationModel.id)))).scalar_one() or 0
        pending_review = (await session.execute(select(func.count(ViolationModel.id)).where(ViolationModel.status == "pending"))).scalar_one() or 0
        approved_citations = (await session.execute(select(func.count(ViolationModel.id)).where(ViolationModel.status == "confirmed"))).scalar_one() or 0
        
        # Cameras
        active_cams = (await session.execute(select(func.count(CameraModel.id)).where(CameraModel.status == "active"))).scalar_one() or 0
        total_cams = (await session.execute(select(func.count(CameraModel.id)))).scalar_one() or 0
        
        # Repeat offenders
        repeat_offenders = (await session.execute(select(func.count(VehicleModel.plate)).where(VehicleModel.is_repeat_offender == True))).scalar_one() or 0
        
        context = (
            f"SYSTEM STATE SUMMARY (Current Time: {datetime.utcnow().isoformat()}Z):\n"
            f"- Total Violation Events logged: {total_violations}\n"
            f"- Citations pending human review: {pending_review}\n"
            f"- Citations approved (confirmed): {approved_citations}\n"
            f"- Camera Registry streams active: {active_cams} / {total_cams} UP\n"
            f"- Repeat Offender vehicles registered: {repeat_offenders}\n"
        )
        return context
    except Exception as e:
        logger.error("Error building context: %s", e)
        return "SYSTEM STATE SUMMARY: Database status unavailable."


# ---------------------------------------------------------------------------
# Tool Executor (SQLite Operations)
# ---------------------------------------------------------------------------

async def execute_agent_tool(tool_name: str, params: Dict[str, Any], session: AsyncSession) -> Dict[str, Any]:
    """Execute SQLAlchemy operations safely and return details to feed to the LLM."""
    try:
        if tool_name == "query_violations":
            status = params.get("status")
            v_type = params.get("type")
            q = select(ViolationModel).order_by(ViolationModel.created_at.desc()).limit(5)
            if status:
                q = q.where(ViolationModel.status == status)
            if v_type:
                q = q.where(ViolationModel.violation_type == v_type)
                
            rows = (await session.execute(q)).scalars().all()
            results = []
            for r in rows:
                results.append({
                    "id": r.id,
                    "type": r.violation_type,
                    "plate": r.plate_text,
                    "location": r.location,
                    "timestamp": r.timestamp,
                    "status": r.status,
                    "confidence": r.confidence
                })
            return {"status": "success", "data": results}

        elif tool_name == "search_plate":
            plate = params.get("plate", "").upper().strip()
            if not plate:
                return {"status": "error", "message": "Missing 'plate' search term"}
            row = (await session.execute(select(VehicleModel).where(VehicleModel.plate == plate))).scalar_one_or_none()
            if not row:
                return {"status": "success", "message": f"No repeats registry found for plate {plate}"}
            
            import json
            violations = json.loads(row.violations_json or "[]")
            return {
                "status": "success",
                "data": {
                    "plate": row.plate,
                    "violation_count": row.violation_count,
                    "is_repeat_offender": row.is_repeat_offender,
                    "first_seen": row.first_seen,
                    "last_seen": row.last_seen,
                    "history": violations
                }
            }

        elif tool_name == "query_cameras":
            rows = (await session.execute(select(CameraModel))).scalars().all()
            cams = [{"id": r.id, "location": r.location, "status": r.status, "description": r.description} for r in rows]
            return {"status": "success", "data": cams}

        elif tool_name == "toggle_camera":
            cam_id = params.get("id", "").upper().strip()
            status = params.get("status", "active").lower()
            cam = (await session.execute(select(CameraModel).where(CameraModel.id == cam_id))).scalar_one_or_none()
            if not cam:
                return {"status": "error", "message": f"Camera {cam_id} not found"}
            cam.status = "active" if status in ["active", "active stream", "enabled"] else "offline"
            await session.commit()
            
            # Log audit
            log = AuditLogModel(
                timestamp=datetime.utcnow().isoformat() + "Z",
                actor="AI Gemma Agent",
                action="CAMERA_CONFIG_MODIFIED",
                target=cam_id,
                details=f"Status toggled to {cam.status}"
            )
            session.add(log)
            await session.commit()
            return {"status": "success", "message": f"Camera {cam_id} status updated to {cam.status}."}

        elif tool_name == "update_violation":
            vio_id = params.get("id", "").upper().strip()
            action = params.get("status", "").lower() # confirmed / rejected
            db_status = "confirmed" if "approve" in action or "confirm" in action else "rejected"
            action_code = "CITATION_APPROVED" if db_status == "confirmed" else "CITATION_REJECTED"
            
            obj = await update_violation_status(session, vio_id, db_status, "AI Gemma Agent")
            if not obj:
                return {"status": "error", "message": f"Violation citation {vio_id} not found"}
                
            # Log audit
            log = AuditLogModel(
                timestamp=datetime.utcnow().isoformat() + "Z",
                actor="AI Gemma Agent",
                action=action_code,
                target=vio_id,
                details=f"AI override: Status changed to {db_status}"
            )
            session.add(log)
            await session.commit()
            return {"status": "success", "message": f"Violation citation {vio_id} marked as {db_status}."}

        else:
            return {"status": "error", "message": f"Tool {tool_name} not registered in executor registry"}
            
    except Exception as e:
        logger.error("Tool execution failed: %s", e)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# LLM Response Parser & Loop
# ---------------------------------------------------------------------------

async def execute_agent_loop(user_message: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Main agent execution loop:
    1. Validates query safety.
    2. Gathers DB context.
    3. Prompts local Ollama gemma3:1b.
    4. Parses tool requests, runs SQLite queries, and summarize results.
    """
    # 1. Guardrail Check
    is_safe, error_msg = check_query_safety(user_message)
    if not is_safe:
        return {"text": error_msg, "ui_action": None}

    async with AsyncSessionLocal() as session:
        db_context = await build_system_context(session)

    # Compile prompt
    system_prompt = (
        "You are Gemma-Agent, the omnipotent AI copilot of the GARUDA Traffic Violation Intelligence Platform.\n"
        "You have complete access to the central traffic enforcement system. You can query database records and execute actions.\n"
        "To perform a database query or system action, you MUST output a single Markdown JSON code block containing the tool action details.\n"
        "DO NOT write text alongside the JSON block in the same turn if you are calling a tool.\n\n"
        "AVAILABLE TOOLS:\n"
        "1. Query violations:\n"
        "   ```json\n"
        "   {\"tool\": \"query_violations\", \"status\": \"pending\" | \"confirmed\" | \"rejected\"}\n"
        "   ```\n"
        "2. Search specific license plate repeat records:\n"
        "   ```json\n"
        "   {\"tool\": \"search_plate\", \"plate\": \"PLATE_TEXT\"}\n"
        "   ```\n"
        "3. List or inspect camera registry feeds:\n"
        "   ```json\n"
        "   {\"tool\": \"query_cameras\"}\n"
        "   ```\n"
        "4. Enable/Disable RTSP camera streams:\n"
        "   ```json\n"
        "   {\"tool\": \"toggle_camera\", \"id\": \"CAM-ID\", \"status\": \"active\" | \"offline\"}\n"
        "   ```\n"
        "5. Approve/Reject pending citations:\n"
        "   ```json\n"
        "   {\"tool\": \"update_violation\", \"id\": \"VIO-ID\", \"status\": \"approve\" | \"reject\"}\n"
        "   ```\n"
        "6. Frontend Page Navigation: If the user asks to open/go to a page (Dashboard, Review queue, Settings, Cameras, Upload page, Search, etc.):\n"
        "   ```json\n"
        "   {\"tool\": \"navigate\", \"path\": \"/dashboard\" | \"/cameras\" | \"/violations\" | \"/review\" | \"/evidence\" | \"/search\" | \"/analytics\" | \"/settings\"}\n"
        "   ```\n\n"
        f"CURRENT PLATFORM STATE:\n{db_context}\n\n"
        "Respond clearly and concisely in markdown formats."
    )

    # Prepare chat payload for Ollama
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "messages": messages, "stream": False},
                timeout=15.0
            )
            
            if response.status_code != 200:
                return {
                    "text": "Ollama is running, but failed to process chat. Verify model gemma3:1b is loaded.",
                    "ui_action": None
                }
                
            res_data = response.json()
            llm_text = res_data.get("message", {}).get("content", "")
            
            # Check for JSON code block containing tool call
            tool_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_text, re.DOTALL)
            if not tool_match:
                # If no code block, try parsing any loose json brackets
                tool_match = re.search(r"(\{[\s\n]*\"tool\"[\s\S]*?\})", llm_text)

            if tool_match:
                try:
                    tool_json = json.loads(tool_match.group(1))
                    tool_name = tool_json.get("tool")
                    
                    if tool_name == "navigate":
                        target_path = tool_json.get("path")
                        return {
                            "text": f"Initializing dashboard navigation sequence to: **{target_path}**...",
                            "ui_action": {"type": "navigate", "path": target_path}
                        }
                    
                    # Run DB operation
                    async with AsyncSessionLocal() as session:
                        tool_result = await execute_agent_tool(tool_name, tool_json, session)
                    
                    # Feed execution result back to LLM to summarize
                    messages.append({"role": "assistant", "content": llm_text})
                    messages.append({
                        "role": "system", 
                        "content": f"TOOL_EXECUTION_RESULT: {json.dumps(tool_result)}"
                    })
                    
                    final_response = await client.post(
                        OLLAMA_URL,
                        json={"model": MODEL_NAME, "messages": messages, "stream": False},
                        timeout=15.0
                    )
                    
                    if final_response.status_code == 200:
                        final_text = final_response.json().get("message", {}).get("content", "")
                        return {"text": final_text, "ui_action": None}
                    else:
                        return {
                            "text": f"Tool completed, but failed to summarize: {json.dumps(tool_result)}",
                            "ui_action": None
                        }
                        
                except Exception as parse_err:
                    logger.error("Failed to parse tool call JSON: %s", parse_err)
                    return {"text": llm_text, "ui_action": None}
            
            return {"text": llm_text, "ui_action": None}
            
    except Exception as e:
        logger.error("Ollama connection error: %s", e)
        return {
            "text": (
                "⚠️ **Ollama AI Connection Failure**\n\n"
                f"Unable to establish connectivity to the Ollama instance on `{settings.OLLAMA_URL}`. "
                "Ensure Ollama is running and the `gemma3:1b` model is active using:\n"
                "```bash\n"
                "ollama run gemma3:1b\n"
                "```"
            ),
            "ui_action": None
        }
