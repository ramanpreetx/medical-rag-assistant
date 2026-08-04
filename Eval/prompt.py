LABEL_PROMPT = """
You are evaluating retrieval relevance for a Medical RAG system.

Question:
{question}

Retrieved Chunks:
{chunks}

Assign exactly one label to every chunk.

Label Definitions

2 = Direct Answer
The chunk contains the primary information needed to answer the user's question.
If this were the ONLY retrieved chunk, you could answer the question accurately.

1 = Supporting Information
The chunk does NOT answer the question by itself, but it provides information that would reasonably be included in a complete answer.

0 = Not Relevant
The chunk should NOT be used when answering the user's question.
Being about the same disease, body system, or medical topic is NOT enough.

Assign 1 ONLY if the information would actually improve the final answer.

If removing the chunk would not noticeably reduce the quality of the final answer,
assign 0 instead.

When uncertain between 0 and 1, choose 0.
When uncertain between 1 and 2, choose 1.

------------------------------------------------------------

Rules

- Evaluate each chunk independently.
- Do not compare chunks with one another.
- Do not use outside medical knowledge.
- Every chunk must receive exactly one label.
- If uncertain between 1 and 2, assign 1.
- Use the provided Chunk ID exactly.
- Return labels for ALL chunks.
"""