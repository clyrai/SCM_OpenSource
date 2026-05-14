"""
Cross-Session Synthesis (CSS).

What: can the system answer queries whose evidence is split across sessions?

How: each cross_session_question in the persona becomes a probe query.
The system's response is scored by keyword presence — must contain at least
one correct keyword and none of the leak keywords. Binary per question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .match import score_keyword_response


@dataclass
class CSSResult:
    score: float
    total_questions: int
    correct_answers: int
    per_question: List[Dict[str, Any]]


def score_css(
    ground_truth_questions: Sequence[Mapping[str, Any]],
    responses_by_question_id: Mapping[str, str],
) -> CSSResult:
    """
    Score CSS for one (persona, system) pair.

    `responses_by_question_id` maps each question_id to the system's
    response text from adapter.query(). Missing responses score 0.
    """
    if not ground_truth_questions:
        return CSSResult(score=0.0, total_questions=0, correct_answers=0, per_question=[])

    per_q = []
    correct = 0
    for q in ground_truth_questions:
        qid = q["question_id"]
        response = responses_by_question_id.get(qid, "")
        s = score_keyword_response(
            response,
            must_contain_any_of=q.get("correct_answer_keywords", []),
            must_not_contain=q.get("incorrect_leak_keywords", []),
            min_match_count=q.get("min_correct_keyword_match", 1),
        )
        if s >= 1.0:
            correct += 1
        per_q.append({
            "question_id": qid,
            "evidence_sessions": q.get("evidence_sessions", []),
            "passed": s >= 1.0,
            "response_excerpt": (response or "")[:200],
        })

    return CSSResult(
        score=correct / len(ground_truth_questions),
        total_questions=len(ground_truth_questions),
        correct_answers=correct,
        per_question=per_q,
    )
