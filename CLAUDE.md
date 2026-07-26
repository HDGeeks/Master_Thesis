1. I am a masters student for NLP
2. I am trying to come up with a thesis expose
3. I have read BTW2025-122.pdf -- it has been summarized into paper_summary.md, do NOT re-read or re-summarize the PDF
4. Do NOT do the same things twice
5. I am looking into ways to improve this concept from its shortcomings
6. At first just ideas as bullet points, later we can vet them with feasibility, difficulty etc
7. Do never use em dashes, do never use emojis, use human simple language as much as possible
8. Always stay on point, do not try to lead me into a rabbit hole
9. When writing code, use very simple logic, well readable, not extreme code that is obviously written by a machine
10. Token and effort optimization rules:
    - Before reading any file, check if it has already been summarized or processed
    - Never re-read a file that was already read in the same session
    - Store results of expensive operations (PDF parsing, summaries) in MD files so they are never repeated
    - Do not re-derive facts already established in the conversation
    - When searching the codebase, use targeted grep or find instead of reading whole files
    - Do not restate or summarize what was just done -- user can see the output
    - Prefer editing existing files over creating new ones
    - Do not create intermediate planning documents unless the user asks
11. TODO management: always keep TODO.md up to date -- update current step, done, and next steps after any meaningful progress
