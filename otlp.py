"""OTLP/JSON trace built only from stored run state (dispatches, receipts, timings). No prompts, transcripts,
tool arguments, tool results or sensitive values are ever exported."""
import time

INTERNAL, SERVER, CLIENT = 1, 2, 3
OK, ERROR = 1, 2


def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}  # OTLP/JSON encodes int64 as a decimal string
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs(d):
    return [{"key": k, "value": _value(v)} for k, v in d.items() if v is not None]


def build(state):
    tr = state["trace"]
    common = {"ga5.run.id": state["runId"], "ga5.public.marker": state["publicMarker"]}
    open_end = max(time.time_ns(), state["lastEvent"])
    spans = []

    def span(span_id, parent, name, kind, start, end, attrs, error=None, links=None, ok=False):
        end = end if end else open_end
        s = {"traceId": tr["traceId"], "spanId": span_id, "name": name, "kind": kind,
             "startTimeUnixNano": str(start), "endTimeUnixNano": str(max(end, start + 1000)),
             "attributes": _attrs({**common, **attrs})}
        if parent:
            s["parentSpanId"] = parent
        if tr.get("tracestate"):  # continue the caller's vendor state on every span of its trace
            s["traceState"] = tr["tracestate"]
        if links:
            s["links"] = links
        if error:
            s["status"] = {"code": ERROR, "message": error}
        elif ok:
            s["status"] = {"code": OK}
        spans.append(s)
        return s

    m = state["model"]
    spans_end = [tr["end"] or open_end]

    tool_spans = []
    for a in state["actions"]:
        if not a["attempts"]:
            continue  # reserved for approval, never executed
        last = a["attempts"][-1]["receipt"]
        err = None
        if a["state"] in ("failed",) and last:
            err = "timeout" if last.get("errorType") == "timeout" else str(last.get("errorType") or last["status"])
        tool = span(a["spanId"], tr["agentSpanId"], f"execute_tool {a['toolName']}", INTERNAL, a["start"], a["end"], {
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": a["toolName"],
            "gen_ai.tool.call.id": a["callId"], "gen_ai.tool.type": "function", "ga5.action.id": a["actionId"],
            "ga5.phase": a["phase"], "ga5.approval.id": a.get("approvalId"),
            "error.type": err}, error=err, ok=a["state"] == "succeeded")
        tool_spans.append((a, tool))
        for att in a["attempts"]:
            r = att["receipt"] or {}
            attrs = {"ga5.action.id": a["actionId"], "gen_ai.tool.name": a["toolName"],
                     "gen_ai.tool.call.id": a["callId"], "ga5.attempt": att["attempt"],
                     "ga5.receipt.id": r.get("receiptId"), "ga5.receipt.nonce": r.get("nonce"),
                     "http.request.method": "POST", "http.request.resend_count": att["attempt"] - 1}
            err = None
            if r:
                if r.get("status"):
                    attrs["http.response.status_code"] = r["status"]
                if r.get("errorType") == "timeout" or r.get("status") == 0:
                    err = "timeout" if r.get("errorType") in (None, "timeout") else str(r["errorType"])
                elif r["status"] >= 400:
                    err = str(r["status"])
                elif r.get("errorType"):
                    err = str(r["errorType"])
                attrs["error.type"] = err
            span(att["spanId"], a["spanId"], f"POST tool/{a['toolName']}", CLIENT, att["start"], att["end"],
                 attrs, error=err, ok=bool(r) and not err)
            spans_end.append(att["end"] or open_end)

    diags = [(a, s) for a, s in tool_spans if a["phase"] == "diagnostic"]
    if len(diags) > 1:
        ends = [a["end"] for a, _ in diags]
        span(tr["joinSpanId"], tr["agentSpanId"], "incident.join", INTERNAL, min(a["start"] for a, _ in diags),
             max(ends) if all(ends) else None,
             {"ga5.join.count": len(diags)},
             links=[{"traceId": tr["traceId"], "spanId": s["spanId"],
                     "attributes": _attrs({"ga5.action.id": a["actionId"]})} for a, s in diags])

    apr = state.get("approval")
    if apr:
        span(tr["approvalSpanId"], tr["agentSpanId"], "approval_gate", INTERNAL, apr["start"], apr["end"], {
            "ga5.approval.id": apr["approvalId"], "ga5.action.id": apr["actionId"],
            "gen_ai.tool.name": apr["toolName"], "ga5.approval.decision": apr.get("decision"),
            "ga5.approval.nonce": apr.get("nonce"), "ga5.receipt.id": apr.get("receiptId"),
            "ga5.receipt.nonce": apr.get("nonce")}, ok=apr.get("decision") is not None)

    span(tr["chatSpanId"], tr["agentSpanId"], "chat incident-plan", CLIENT, m["start"], m["end"], {
        "gen_ai.operation.name": "chat", "gen_ai.request.model": m["model"],
        "gen_ai.response.model": m["model"] if m.get("ok") else None, "gen_ai.provider.name": m.get("provider"),
        "gen_ai.usage.input_tokens": m.get("input_tokens"), "gen_ai.usage.output_tokens": m.get("output_tokens"),
        "error.type": m.get("error")}, error=m.get("error"), ok=bool(m.get("ok")))

    end = max(e for e in spans_end if e)
    final = state["status"] != "waiting"
    span(tr["agentSpanId"], tr["serverSpanId"], f"invoke_agent {state['agentName']}", INTERNAL, tr["start"] + 1000,
         end if final else None, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": state["agentName"],
                                  "ga5.run.status": state["status"]})
    span(tr["serverSpanId"], tr.get("remoteParent"), "POST /v2/incidents", SERVER, tr["start"],
         (end + 1000) if final else None, {"http.request.method": "POST", "http.route": "/v2/incidents",
                                           "url.path": "/v2/incidents", "http.response.status_code": 200})
    spans.reverse()  # root first
    return {"resourceSpans": [{
        "resource": {"attributes": _attrs({"service.name": "ga5-incident-agent", **common})},
        "scopeSpans": [{"scope": {"name": "ga5.incident-agent", "version": "1.0.0"}, "spans": spans}],
    }]}
