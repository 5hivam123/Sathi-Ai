"""
Tests for quiz JSON parsing.
Tests the parsing logic without needing a real LLM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.quiz_generator import _parse_quiz_output, QuizQuestion


VALID_QUIZ_JSON = """
[
  {
    "question": "What does CPU stand for?",
    "options": {
      "A": "Central Processing Unit",
      "B": "Computer Personal Unit",
      "C": "Central Program Unit",
      "D": "Control Processing Utility"
    },
    "correct_answer": "A",
    "explanation": "CPU stands for Central Processing Unit, the main processor of a computer."
  },
  {
    "question": "Which scheduling algorithm is non-preemptive?",
    "options": {
      "A": "Round Robin",
      "B": "FCFS",
      "C": "SJF Preemptive",
      "D": "Priority Preemptive"
    },
    "correct_answer": "B",
    "explanation": "First Come First Served (FCFS) is a non-preemptive scheduling algorithm."
  }
]
"""

QUIZ_JSON_WITH_PREAMBLE = """
Here are the quiz questions you requested:

```json
[
  {
    "question": "What is virtual memory?",
    "options": {
      "A": "RAM stored on disk",
      "B": "A technique extending memory using disk space",
      "C": "A type of cache",
      "D": "Physical memory chips"
    },
    "correct_answer": "B",
    "explanation": "Virtual memory extends physical memory using disk storage."
  }
]
```

Hope this helps!
"""

MALFORMED_JSON = "The quiz questions are: 1. What is an OS? 2. What is a process?"


def test_parse_valid_json():
    result = _parse_quiz_output(VALID_QUIZ_JSON, expected_count=2)
    assert result.is_valid
    assert len(result.questions) == 2
    assert result.questions[0].correct_answer == "A"
    assert result.questions[1].correct_answer == "B"


def test_parse_json_with_preamble():
    result = _parse_quiz_output(QUIZ_JSON_WITH_PREAMBLE, expected_count=1)
    assert result.is_valid
    assert len(result.questions) == 1
    assert result.questions[0].correct_answer == "B"


def test_parse_malformed_returns_error():
    result = _parse_quiz_output(MALFORMED_JSON, expected_count=5)
    assert not result.is_valid
    assert result.error


def test_quiz_answer_scoring():
    q = QuizQuestion(
        question="Test question",
        options={"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
        correct_answer="A",
        explanation="A is correct",
        user_answer="A",
    )
    assert q.is_correct is True

    q.user_answer = "B"
    assert q.is_correct is False


def test_parse_empty_string_returns_error():
    result = _parse_quiz_output("", expected_count=5)
    assert not result.is_valid


def test_parse_correct_answer_case_insensitive():
    """Correct answer should be normalized to uppercase."""
    json_input = """
    [
      {
        "question": "Test?",
        "options": {"A": "Yes", "B": "No", "C": "Maybe", "D": "Never"},
        "correct_answer": "a",
        "explanation": "A is correct"
      }
    ]
    """
    result = _parse_quiz_output(json_input, expected_count=1)
    assert result.is_valid
    assert result.questions[0].correct_answer == "A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
