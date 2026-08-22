"""The prompts the experiments run against, ordered by how predictable they are.

Acceptance is how often the small draft guesses what the large target would have
said, so it depends on the text. These five span the range deliberately: a fixed
schedule the draft over-generalises, prose it must genuinely predict, and code
where the next token is often forced by syntax.
"""

WORKLOADS = {
    "structured": (
        "Monday: gym\nTuesday: gym\nWednesday: gym\nThursday: gym\nFriday:"
    ),
    "factual list": (
        "The capital of France is Paris. The capital of Germany is Berlin. "
        "The capital of Italy is"
    ),
    "open prose": "The history of the printing press begins",
    "code": (
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n\n"
        "def mul(a, b):\n    return"
    ),
    "repetitive": "a b c a b c a b c a b c a b c a b c a b c",
}

# Used for the headline comparison and the k sweep.
DEFAULT = "open prose"
