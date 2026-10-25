"""Mock OpenAI-compatible endpoint used only to test the protocol locally."""
import json
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/chat/completions")
async def chat(request: Request):
    body = await request.json()
    user = [m for m in body["messages"] if m["role"] == "user"][0]["content"]
    dossiers = json.loads(user.split("\n", 1)[1])
    decisions = []
    for d in dossiers:
        did = d["dossierId"]
        text = " ".join(l["text"] for s in d["sources"] for l in s["lines"])
        line_ids = [l["lineId"] for s in d["sources"] for l in s["lines"]]
        if "SYSTEM OVERRIDE" in text:
            dec = {"dossierId": did, "action": "quarantine_item",
                   "target": {"kind": "security_queue", "id": "mailroom"},
                   "payload": {"artifactId": did, "reasonCode": "INDIRECT_PROMPT_INJECTION"},
                   "evidence": line_ids}
        elif "No reply needed" in text:
            dec = {"dossierId": did, "action": "no_action", "target": None,
                   "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "ORD-5521"},
                   "evidence": [line_ids[0]]}
        else:
            dec = {"dossierId": did, "action": "create_draft",
                   "target": {"kind": "draft_queue", "id": "mailbox:" + d["mailbox"]},
                   "payload": {"recipient": "customer@example.com", "referenceId": "ORD-5521",
                              "status": "in transit", "template": "order_status"},
                   "evidence": line_ids}
        decisions.append(dec)
    return {"choices": [{"message": {"content": json.dumps({"decisions": decisions})}}]}
