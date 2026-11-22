"""Low-cost local alerts only; does not imply an offline chat push channel."""

import json
import time

from prepare import OUT

from primary_ml_cka.baselines.protocol import atomic, digest


def poll():
    ledger = OUT / "alerts.json"
    alerts = json.loads(ledger.read_text()) if ledger.exists() else []
    seen = {r["id"] for r in alerts}

    def emit(kind, key, message):
        identity = digest([kind, key])
        if identity not in seen:
            seen.add(identity)
            alerts.append({"id": identity, "time": time.time(), "kind": kind, "message": message})
            print(kind, message, flush=True)

    statefile = OUT / "state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    for failure in state.get("failed", []):
        emit("ERROR", failure, f"Experiment stage failed: {failure}")
    if state.get("status") in ["RUNNING", "WAITING_EXISTING_GPU_OWNER"]:
        if time.time() - state.get("heartbeat", time.time()) > 180:
            emit(
                "ERROR",
                "stale_controller",
                "Controller heartbeat stale >180sec; inspect process before restart",
            )
    resultfile = OUT / "results.json"
    rows = json.loads(resultfile.read_text()) if resultfile.exists() else []
    for r in rows:
        key = [r["run"], r["target"]]
        if r["dataset"] == "crossvlad_captioning" and r["ASR"] is not None:
            emit(
                "RESULT",
                key,
                f"{r['method']} -> {r['target']}: "
                f"{r['target_hits']}/{r['planned']} = {r['ASR']:.1%}",
            )
            if r["planned"] != 10 or r["returned"] != 10 or not 0 <= r["target_hits"] <= 10:
                emit("ERROR", key + ["denominator"], "Invalid smoke numerator/denominator")
            if r.get("clean_target_hits", 0) > 0:
                emit(
                    "REVIEW",
                    key + ["clean_hit"],
                    "Clean captions already satisfy target criterion; inspect paired clean control",
                )
        if r["status"] == "BLOCKED_OFFICIAL_GPT4_ACCESS":
            emit(
                "BLOCKER",
                "official_gpt4",
                "LLaVA raw answers ready; official GPT4 scoring not configured. No substitute judge.",
            )
    if state.get("status") in [
        "RAW_COMPLETE_OFFICIAL_LLAVA_JUDGE_BLOCKED",
        "PARTIAL_RESUMABLE",
        "BLOCKED",
    ]:
        emit("END", state["status"], "Campaign end state: " + state["status"])
    atomic(ledger, alerts)
    lines = [
        "# Local result / error alerts",
        "",
        "No offline chat, email or push delivery is configured. These are durable server alerts.",
        "",
    ]
    lines.extend(
        f"- {time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(r['time']))} "
        f"[{r['kind']}] {r['message']}"
        for r in alerts
    )
    (OUT / "ALERTS.md").write_text("\n".join(lines) + "\n")
    return state.get("status") in [
        "RAW_COMPLETE_OFFICIAL_LLAVA_JUDGE_BLOCKED",
        "PARTIAL_RESUMABLE",
        "BLOCKED",
    ]


if __name__ == "__main__":
    while True:
        try:
            if poll():
                break
        except (OSError, ValueError) as error:
            print("MONITOR_READ_RETRY", repr(error), flush=True)
        time.sleep(30)
