# -*- coding: utf-8 -*-
"""
Feedback Collector for DPO Training
Stores (prompt, chosen_advisory, rejected_advisory) preference pairs
"""
import sqlite3
import json
import os
import threading
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join("data", "freightsense.db")

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preference_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shipment_id TEXT,
            constraint_json TEXT,
            rejected_advisory TEXT,
            chosen_advisory TEXT,
            dispatcher_id TEXT,
            timestamp TEXT,
            rejection_reason TEXT,
            quality_score_before REAL,
            quality_score_after REAL
        )
    """)
    conn.commit()
    conn.close()

_init_db()

def _async_evaluate_and_log(shipment_id: str, constraints: dict, rejected: str, chosen: str, dispatcher_id: str, reason: str):
    """Runs LLM judge evaluation asynchronously and updates the DB"""
    try:
        from tests.llm_judge import LLMJudge
        judge = LLMJudge()
        
        # We dummy the instruction since we only care about the advisory quality against constraints
        instruction = "Unknown instruction"
        
        score_before = 0.0
        score_after = 0.0
        
        if judge.model:
            res_before = judge.evaluate_single(instruction, constraints, rejected)
            res_after = judge.evaluate_single(instruction, constraints, chosen)
            score_before = res_before.get("overall_score", 0.0)
            score_after = res_after.get("overall_score", 0.0)
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO preference_pairs 
            (shipment_id, constraint_json, rejected_advisory, chosen_advisory, dispatcher_id, timestamp, rejection_reason, quality_score_before, quality_score_after)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (shipment_id, json.dumps(constraints), rejected, chosen, dispatcher_id, datetime.now().isoformat(), reason, score_before, score_after))
        
        conn.commit()
        conn.close()
        
        # Check if we should trigger DPO training
        if get_pair_count() >= 100:
            print("🎉 100+ preference pairs collected! Ready for DPO Fine-tuning.")
            
    except Exception as e:
        print(f"⚠️ Async evaluation failed: {e}")

def log_preference_pair(shipment_id: str, constraints: dict, rejected: str, chosen: str, dispatcher_id: str = "unknown", reason: Optional[str] = None):
    """
    Logs the user's preferred advisory against the AI's rejected advisory.
    Spawns a background thread to run the LLM-as-judge so the API returns immediately.
    """
    t = threading.Thread(target=_async_evaluate_and_log, args=(shipment_id, constraints, rejected, chosen, dispatcher_id, reason))
    t.start()

def get_pair_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM preference_pairs")
    count = cursor.fetchone()[0]
    conn.close()
    return count

def export_for_dpo_training(output_path: str = "data/dpo_training_data.json") -> int:
    """
    Export all pairs as Hugging Face DPO format.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM preference_pairs")
    rows = cursor.fetchall()
    conn.close()
    
    dataset = []
    for row in rows:
        dataset.append({
            "prompt": f"Generate a risk advisory for this freight: {row['constraint_json']}",
            "chosen": row["chosen_advisory"],
            "rejected": row["rejected_advisory"]
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✅ Exported {len(dataset)} pairs to {output_path}")
    return len(dataset)
