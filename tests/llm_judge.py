# -*- coding: utf-8 -*-
"""
LLM-as-Judge Evaluation Framework for FreightSense
Uses Gemini gemini-1.5-flash (free tier) as the evaluator
Scores advisory quality on 4 dimensions automatically
"""
import os
import json
import time
import pandas as pd
from typing import List, Dict
import google.generativeai as genai

JUDGE_SYSTEM_PROMPT = """
You are an expert Indian logistics compliance evaluator with 15 years of experience.
Your job is to evaluate AI-generated freight risk advisories for quality and accuracy.

You will be given:
1. The original freight instruction
2. The extracted constraints (JSON)
3. The AI-generated advisory to evaluate

Score the advisory on these 4 dimensions (0-10 each):

1. FACTUAL_ACCURACY (0-10): Are all regulation references correct? 
   Are the risk flags justified by real Indian logistics laws?
   10 = All facts correct, 0 = Multiple factual errors

2. ACTIONABILITY (0-10): Does the advisory give specific, actionable steps?
   "Avoid Ring Road" is actionable. "There may be risks" is not.
   10 = Every recommendation is specific and immediately actionable

3. COMPLETENESS (0-10): Does the advisory cover ALL constraints from the JSON?
   Missing a HIGH severity constraint = major deduction.
   10 = All constraints addressed, 0 = Major constraints ignored

4. CLARITY (0-10): Is the language clear for a non-technical dispatcher?
   No jargon. Simple sentences. Concrete numbers.
   10 = Perfectly clear, 0 = Confusing or technical

Return ONLY valid JSON. No markdown. No explanation outside the JSON.
Schema:
{
  "factual_accuracy": 0,
  "actionability": 0,
  "completeness": 0,
  "clarity": 0,
  "overall_score": 0.0,
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "improved_advisory": "A better version of the advisory in 3 sentences"
}
"""

class LLMJudge:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            print("⚠️ GEMINI_API_KEY not found. LLMJudge disabled.")
            self.model = None

    def evaluate_single(self, instruction: str, constraints: dict, advisory: str) -> dict:
        if not self.model:
            return {"error": "API key missing"}
            
        prompt = f"""{JUDGE_SYSTEM_PROMPT}
        
---
INSTRUCTION:
{instruction}

CONSTRAINTS:
{json.dumps(constraints, indent=2)}

ADVISORY TO EVALUATE:
{advisory}
"""
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            return json.loads(raw_text.strip())
        except Exception as e:
            print(f"⚠️ LLMJudge evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_batch(self, test_cases: List[dict], output_csv: str = "tests/eval_results.csv") -> pd.DataFrame:
        results = []
        print(f"⚖️ Starting LLM-as-Judge evaluation on {len(test_cases)} cases...")
        
        for idx, case in enumerate(test_cases):
            print(f"Evaluating case {idx+1}/{len(test_cases)}: {case.get('id', f'TEST_{idx}')}")
            
            advisory = case.get("advisory", "")
            if not advisory:
                # If test case doesn't have a generated advisory yet, we skip or run it
                # For this implementation we assume the pipeline was already run and output saved
                continue
                
            eval_res = self.evaluate_single(
                instruction=case.get("instruction", ""),
                constraints=case.get("expected_constraints", {}),
                advisory=advisory
            )
            
            row = {
                "test_id": case.get("id", ""),
                "scenario": case.get("scenario_type", ""),
                "expected_risk": case.get("expected_risk", ""),
            }
            if "error" not in eval_res:
                row.update({
                    "factual_accuracy": eval_res.get("factual_accuracy", 0),
                    "actionability": eval_res.get("actionability", 0),
                    "completeness": eval_res.get("completeness", 0),
                    "clarity": eval_res.get("clarity", 0),
                    "overall_score": eval_res.get("overall_score", 0),
                    "strengths": "; ".join(eval_res.get("strengths", [])),
                    "weaknesses": "; ".join(eval_res.get("weaknesses", []))
                })
            else:
                row["error"] = eval_res["error"]
                
            results.append(row)
            # Rate limit for free tier (15 RPM)
            time.sleep(4)
            
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"✅ Evaluation complete. Saved to {output_csv}")
        return df

    def generate_eval_report(self, results_df: pd.DataFrame) -> str:
        if results_df.empty:
            return "No evaluation results."
            
        mean_scores = results_df[['factual_accuracy', 'actionability', 'completeness', 'clarity', 'overall_score']].mean()
        pass_count = (results_df['overall_score'] >= 7.0).sum()
        total_count = len(results_df)
        pass_rate = (pass_count / total_count) * 100 if total_count > 0 else 0
        
        # Best and worst
        best = results_df.nlargest(3, 'overall_score')
        worst = results_df.nsmallest(3, 'overall_score')
        
        report = f"""# FreightSense LLM-as-Judge Evaluation Report

## Summary
- **Total Test Cases**: {total_count}
- **Pass Rate (Score >= 7.0)**: {pass_rate:.1f}% ({pass_count}/{total_count})

## Mean Scores (out of 10)
| Dimension | Score |
| :--- | :--- |
| Factual Accuracy | {mean_scores['factual_accuracy']:.2f} |
| Actionability | {mean_scores['actionability']:.2f} |
| Completeness | {mean_scores['completeness']:.2f} |
| Clarity | {mean_scores['clarity']:.2f} |
| **Overall Score** | **{mean_scores['overall_score']:.2f}** |

## Top 3 Best Advisories
"""
        for _, row in best.iterrows():
            report += f"- **{row['test_id']}** (Score: {row['overall_score']}): Strengths: {row['strengths']}\n"
            
        report += "\n## Top 3 Needs Improvement\n"
        for _, row in worst.iterrows():
            report += f"- **{row['test_id']}** (Score: {row['overall_score']}): Weaknesses: {row['weaknesses']}\n"
            
        return report
