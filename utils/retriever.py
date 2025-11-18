import os

DATA_PATH = "data/notes.txt"


def get_relevant_context(query: str, max_chars: int = 800) -> str:
    """
    Very simple keyword-based retriever over a local notes file.
    Looks for lines that contain words from the query and returns
    the top matches concatenated together (truncated to max_chars).
    """

    if not os.path.exists(DATA_PATH):
        return "No study notes found."

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Break into lines
    lines = text.splitlines()

    # Simple scoring: count matching words per line
    query_words = [w.lower() for w in query.split() if w.strip()]
    scored = []

    for line in lines:
        line_lower = line.lower()
        score = sum(1 for w in query_words if w in line_lower)
        if score > 0:
            scored.append((score, line))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top lines and join them
    best_lines = [line for _, line in scored[:5]]
    combined = "\n".join(best_lines)

    if not combined:
        return "No matching context found in notes."

    # Limit the overall size
    return combined[:max_chars]
