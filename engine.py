"""Receipt-driven state machine. Pure functions over a JSON-serialisable run state; never calls a model."""
import copy
import hashlib
import json
import secrets
import time

import otlp

class InvalidTransition(Exception):
    pass


def new_id(prefix, nbytes=8):
    return f"{prefix}_{secrets.token_hex(nbytes)}"


def span_id():
    while True:
        s = secrets.token_hex(8)
        if int(s, 16):
            return s


def trace_id():
    while True:
        t = secrets.token_hex(16)
        if int(t, 16):
            return t


def now_ns():
    return time.time_ns()


def args_digest(arguments):
    blob = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------- run creation ----------

def new_run(req, plan, model_meta, trace_ctx, received_ns):
    """trace_ctx: {"traceId", "parentSpanId" or None, "tracestate" or None, "flags"}."""
    policy = req["policy"]
    state = {
        "runId": req["runId"],
        "publicMarker": req["publicMarker"],
        "agentName": req.get("agentName") or "incident-response",
        "status": "waiting",
        "diagnosis": {"rootCause": plan["rootCause"], "evidence": plan["evidence"]},
        "planSource": plan.get("source"),
        "effectPlan": plan.get("effect"),
        "alternateEffect": plan.get("alternateEffect"),
        "approvalRequiredFor": list(policy.get("approvalRequiredFor") or []),
        "trace": {
            "traceId": trace_ctx["traceId"], "remoteParent": trace_ctx.get("parentSpanId"),
            "tracestate": trace_ctx.get("tracestate"),
            "serverSpanId": span_id(), "agentSpanId": span_id(), "chatSpanId": span_id(),
            "joinSpanId": span_id(), "approvalSpanId": span_id(),
            "start": received_ns, "end": None,
        },
        "model": model_meta,
        "actions": [],
        "approval": None,
        "chosenEffect": None,
        "suppressed": [],
        "actionLog": [],
        "receiptLog": [],
        "lastEvent": received_ns,
    }
    t = max(now_ns(), model_meta["end"])
    issued = []
    for d in plan["diagnostics"]:
        action = _new_action(state, "diagnostic", d, t)
        issued.append(_dispatch(state, action, t))
    return state, issued, []


def _new_action(state, phase, spec, t):
    action_id = new_id("act")
    action = {
        "actionId": action_id, "callId": new_id("call"), "phase": phase,
        "toolName": spec["toolName"], "arguments": spec["arguments"], "evidence": spec["evidence"],
        "spanId": span_id(), "state": "pending", "start": t, "end": None, "attempts": [],
        "approvalId": None, "approvalNonce": None,
    }
    state["actions"].append(action)
    return action


def _dispatch(state, action, t):
    attempt_no = len(action["attempts"]) + 1
    sid = span_id()
    d = {
        "actionId": action["actionId"], "callId": action["callId"], "phase": action["phase"],
        "toolName": action["toolName"], "arguments": action["arguments"], "evidence": action["evidence"],
        "attempt": attempt_no, "traceparent": f"00-{state['trace']['traceId']}-{sid}-01",
    }
    if state["trace"].get("tracestate"):
        d["tracestate"] = state["trace"]["tracestate"]
    if action.get("approvalId"):
        d["approvalId"] = action["approvalId"]
        d["approvalNonce"] = action["approvalNonce"]
    action["attempts"].append({"attempt": attempt_no, "spanId": sid, "start": t, "end": None,
                               "receipt": None, "dispatch": d})
    action["state"] = "pending"
    state["actionLog"].append(d)
    return d


# ---------- receipts ----------

def _require_str(obj, key, label):
    v = obj.get(key)
    if not isinstance(v, str) or not v.strip():
        raise InvalidTransition(f"{label}.{key} must be a nonempty string")
    return v


def _require_int(obj, key, label):
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, int):
        raise InvalidTransition(f"{label}.{key} must be an integer")
    return v


def _outstanding(action):
    if action["attempts"] and action["attempts"][-1]["receipt"] is None and action["state"] == "pending":
        return action["attempts"][-1]
    return None


def apply_receipt(state, receipt_id, body):
    """Validate the whole receipt first, then apply it to a copy. Returns (new_state, dispatches, approvals)."""
    if state["status"] != "waiting":
        raise InvalidTransition("run is already in a terminal state")
    outcomes = body.get("outcomes", [])
    approvals = body.get("approvals", [])
    if not isinstance(outcomes, list) or not isinstance(approvals, list):
        raise InvalidTransition("outcomes and approvals must be arrays")
    if not outcomes and not approvals:
        raise InvalidTransition("receipt has no outcomes or approvals")

    by_id = {a["actionId"]: a for a in state["actions"]}
    seen = set()
    for i, o in enumerate(outcomes):
        if not isinstance(o, dict):
            raise InvalidTransition("outcome must be an object")
        label = f"outcomes[{i}]"
        aid = _require_str(o, "actionId", label)
        cid = _require_str(o, "callId", label)
        att = _require_int(o, "attempt", label)
        _require_int(o, "status", label)
        if "nonce" in o and not isinstance(o["nonce"], str):
            raise InvalidTransition(f"{label}.nonce must be a string")
        action = by_id.get(aid)
        pending = _outstanding(action) if action else None
        if not pending or action["callId"] != cid or pending["attempt"] != att:
            raise InvalidTransition(f"{label} does not match a pending call")
        if aid in seen:
            raise InvalidTransition(f"{label} duplicates another outcome")
        seen.add(aid)
    pending_approval = state["approval"] if state["approval"] and state["approval"]["decision"] is None else None
    for i, a in enumerate(approvals):
        if not isinstance(a, dict):
            raise InvalidTransition("approval must be an object")
        label = f"approvals[{i}]"
        apr = _require_str(a, "approvalId", label)
        decision = _require_str(a, "decision", label)
        _require_str(a, "nonce", label)
        if not pending_approval or pending_approval["approvalId"] != apr or i > 0:
            raise InvalidTransition(f"{label} does not match a pending approval")
        if decision not in ("approved", "rejected", "denied"):
            raise InvalidTransition(f"{label}.decision is not supported")

    st = copy.deepcopy(state)
    t = max(now_ns(), st["lastEvent"] + 1_000)
    by_id = {a["actionId"]: a for a in st["actions"]}
    issued, approval_requests = [], []

    for o in outcomes:
        action = by_id[o["actionId"]]
        att = _outstanding(action)
        att["receipt"] = {"receiptId": receipt_id, "nonce": o.get("nonce"), "status": o["status"],
                          "errorType": o.get("errorType"), "resultClass": o.get("resultClass")}
        att["end"] = t
        st["receiptLog"].append({"receiptId": receipt_id, **o})
        # Only the HTTP outcome decides: 2xx succeeds, 503 allows one retry, anything else (timeout = status 0) fails.
        # resultClass is the grader's free-form label and is recorded, never interpreted.
        status = o["status"]
        if status == 503:
            action["state"] = "retryable" if att["attempt"] == 1 else "failed"
        elif 200 <= status < 300 and not o.get("errorType"):
            action["state"] = "succeeded"
        else:
            action["state"] = "failed"
        if action["state"] != "retryable":
            action["end"] = t

    diagnostics = [a for a in st["actions"] if a["phase"] == "diagnostic"]
    doomed = any(a["state"] == "failed" for a in diagnostics)
    # 503 retries: exactly one per logical call, and only while the run can still reach its effect.
    for a in st["actions"]:
        if a["state"] == "retryable":
            if a["phase"] == "diagnostic" and doomed:
                a["state"], a["end"] = "failed", t
            else:
                issued.append(_dispatch(st, a, t))

    for a in approvals:
        apr = st["approval"]
        apr.update(decision=a["decision"], nonce=a["nonce"], receiptId=receipt_id, end=t)
        st["receiptLog"].append({"receiptId": receipt_id, **a})
        action = by_id[apr["actionId"]]
        if a["decision"] == "approved":
            action["approvalId"], action["approvalNonce"] = apr["approvalId"], a["nonce"]
            action["start"] = t
            issued.append(_dispatch(st, action, t))
            st["chosenEffect"] = action["toolName"]
        else:
            action["state"] = "rejected"
            st["suppressed"].append(action["toolName"])
            _finish(st, "failed", t)

    effect = next((a for a in st["actions"] if a["phase"] == "effect"), None)
    if st["status"] == "waiting":
        if effect and effect["attempts"]:
            if effect["state"] == "succeeded":
                st["chosenEffect"] = effect["toolName"]
                _finish(st, "completed", t)
            elif effect["state"] == "failed":
                _finish(st, "failed", t)
        elif not effect and not any(_outstanding(a) for a in diagnostics):
            if doomed:
                if st["effectPlan"]:
                    st["suppressed"].append(st["effectPlan"]["toolName"])
                _finish(st, "failed", t)
            elif all(a["state"] == "succeeded" for a in diagnostics):
                spec = st["effectPlan"]  # decided from the evidence at planning time
                if spec is None:
                    _finish(st, "failed", t)
                else:
                    action = _new_action(st, "effect", spec, t)
                    if spec["toolName"] in st["approvalRequiredFor"]:
                        action["state"] = "awaiting_approval"
                        st["approval"] = {"approvalId": new_id("apr"), "actionId": action["actionId"],
                                          "toolName": action["toolName"],
                                          "argumentsDigest": args_digest(action["arguments"]),
                                          "start": t, "end": None, "decision": None, "nonce": None,
                                          "receiptId": None}
                        approval_requests.append(approval_view(st["approval"]))
                    else:
                        issued.append(_dispatch(st, action, t))
                        st["chosenEffect"] = action["toolName"]
    st["lastEvent"] = t
    return st, issued, approval_requests


def _finish(st, status, t):
    st["status"] = status
    st["trace"]["end"] = t


def approval_view(apr):
    return {k: apr[k] for k in ("approvalId", "actionId", "toolName", "argumentsDigest")}


# ---------- views ----------

def outstanding_dispatches(state):
    return [att["dispatch"] for a in state["actions"] if (att := _outstanding(a))]


def pending_approvals(state):
    apr = state.get("approval")
    return [approval_view(apr)] if apr and apr["decision"] is None and state["status"] == "waiting" else []


def view(state, dispatches=None, approvals=None):
    """Response document built only from stored state. dispatches/approvals default to what is outstanding.
    A waiting run reports only its pending work; the logs, chosen effect and trace come with the terminal state."""
    if state["status"] == "waiting":
        return {
            "runId": state["runId"],
            "status": state["status"],
            "diagnosis": state["diagnosis"],
            "dispatches": outstanding_dispatches(state) if dispatches is None else dispatches,
            "approvals": pending_approvals(state) if approvals is None else approvals,
        }
    return {
        "runId": state["runId"],
        "status": state["status"],
        "diagnosis": state["diagnosis"],
        "dispatches": [],
        "approvals": [],
        "chosenEffect": state["chosenEffect"],
        "suppressed": state["suppressed"],
        "actionLog": state["actionLog"],
        "receiptLog": state["receiptLog"],
        "otlp": otlp.build(state),
    }
