"""Customer-facing chat endpoint.

Single POST endpoint shared by WhatsApp, web, and mobile clients. The
``channel`` field on the request distinguishes them. The tenant is resolved
from ``X-API-Key`` (or the configured default when single-tenant mode is on);
every downstream call — vector search, conversation memory, SQL filters — is
scoped to that tenant.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import ORJSONResponse

from app.api.deps import TenantDep, get_graph
from app.chatbot.graph import ChatGraphRunner
from app.core import conversation_log as conv
from app.core.logging import request_id_var
from app.core.tenancy import Tenant
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(default_response_class=ORJSONResponse)


@router.post("", response_model=ChatResponse, summary="Send a customer message")
async def chat(
    body: ChatRequest,
    tenant: Tenant = TenantDep,
    graph: ChatGraphRunner = Depends(get_graph),
) -> ChatResponse:
    request_id = request_id_var.get() or "REQ_unknown"
    conv.start()
    conv.note("webhook", f"http:{body.channel}")
    conv.note("sender", body.customer_external_id or "—")
    conv.note("prompt", body.message)
    result = await graph.handle(
        request_id=request_id,
        tenant_id=tenant.id,
        conversation_id=body.conversation_id,
        customer_query=body.message,
        customer_external_id=body.customer_external_id,
        channel=body.channel,
    )
    conv.note("reply to", body.customer_external_id or body.channel)
    # expose the per-turn trace (flow + tools + memory) for the Streamlit flow UI
    result["trace"] = conv.events()
    conv.flush()
    return ChatResponse(**result)
