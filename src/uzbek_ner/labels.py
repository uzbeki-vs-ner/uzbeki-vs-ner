"""Hackathon entity labels and BIO tag vocabulary."""

ENTITY_LABELS: tuple[str, ...] = ("ORG", "NAME", "GEO")

TAGS: tuple[str, ...] = (
    "O",
    "B-ORG",
    "I-ORG",
    "B-NAME",
    "I-NAME",
    "B-GEO",
    "I-GEO",
)

TAG_TO_ID: dict[str, int] = {tag: index for index, tag in enumerate(TAGS)}
ID_TO_TAG: dict[int, str] = {index: tag for tag, index in TAG_TO_ID.items()}
