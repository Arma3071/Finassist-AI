"""Golden evaluation dataset for FinAssist AI.

Each item pairs a question with the document(s) that should be retrieved
(for precision/recall) and a set of keywords the answer is expected to
contain (used as a lightweight relevance/hallucination proxy in the
absence of full human-labeled references).
"""

EVAL_DATASET: list[dict] = [
    {
        "question": "What was Q3 revenue and how much did it grow year-over-year?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["482", "14%", "revenue"],
    },
    {
        "question": "What was the gross margin in Q3 compared to the prior year?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["gross margin", "61.2", "58.7"],
    },
    {
        "question": "How much cash did the company have at quarter end?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["cash", "1.1 billion"],
    },
    {
        "question": "What is the full-year revenue guidance?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["guidance", "1.9", "2.0 billion"],
    },
    {
        "question": "What drove the Q3 revenue growth?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["enterprise", "software", "hardware"],
    },
    {
        "question": "What was the operating income for Q3?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["operating income", "96 million"],
    },
    {
        "question": "How did operating expenses change year-over-year?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["operating expenses", "4%"],
    },
    {
        "question": "What was the company's total debt at quarter end?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["debt", "420 million"],
    },
    {
        "question": "What was the current ratio at quarter end?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["current ratio", "2.3"],
    },
    {
        "question": "Which segment contributed the most to Q3 revenue?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["enterprise", "software", "290 million"],
    },
    {
        "question": "What is the company's outlook for the full year?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["guidance", "momentum", "enterprise"],
    },
    {
        "question": "How does the current ratio reflect the company's liquidity?",
        "relevant_documents": ["sample_report.md"],
        "expected_keywords": ["current ratio", "2.3", "liquidity"],
    },
]
