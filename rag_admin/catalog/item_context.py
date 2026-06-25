"""Human context for catalog browse rows."""

from __future__ import annotations

import re
from typing import Any

from rag_admin.catalog.zim_versions import format_version_label, parse_zim_stamp

SOURCE_META: dict[str, dict[str, str]] = {
    "dotsrc": {
        "label": "Kiwix mirror",
        "tagline": "Fast European mirror of the public ZIM library.",
        "accent": "kiwix",
    },
    "kiwix": {
        "label": "Kiwix official",
        "tagline": "Direct listing from lb.download.kiwix.org.",
        "accent": "kiwix",
    },
    "archive": {
        "label": "Internet Archive",
        "tagline": "Collections, texts, and historical media packages.",
        "accent": "archive",
    },
    "arxiv": {
        "label": "arXiv",
        "tagline": "Research preprints by category or search.",
        "accent": "arxiv",
    },
}

FOLDER_HINTS: dict[str, str] = {
    "wikipedia": "Full Wikipedia snapshots for offline reading and RAG.",
    "wiktionary": "Dictionary and thesaurus entries across languages.",
    "wikibooks": "Open textbooks and instructional material.",
    "wikisource": "Primary source texts and classical literature.",
    "wikinews": "News articles archived for offline use.",
    "wikiquote": "Quotations and attributed sayings.",
    "wikiversity": "Learning resources and course materials.",
    "wikivoyage": "Travel guides and destination articles.",
    "stack_exchange": "Technical Q&A from Stack Overflow and sister sites.",
    "devdocs": "API documentation bundles (Python, JS, and more).",
    "freecodecamp": "Programming curriculum and tutorials.",
    "gutenberg": "Public-domain books from Project Gutenberg.",
    "ifixit": "Repair guides and teardown documentation.",
    "libretexts": "Open STEM textbooks from LibreTexts.",
    "ted": "TED talk transcripts and metadata.",
    "maps": "OpenStreetMap and related map data.",
    "other": "Miscellaneous ZIM packages.",
    "zimit": "Community ZIM builds via Zimit.",
}

ARXIV_PREFIX_HINTS: dict[str, str] = {
    "cs": "Computer science preprints and surveys.",
    "math": "Mathematics papers across subfields.",
    "physics": "Physics research including astrophysics and optics.",
    "stat": "Statistics, methodology, and machine learning theory.",
    "q-bio": "Quantitative biology and bioinformatics.",
    "q-fin": "Quantitative finance and economics.",
    "eess": "Electrical engineering and systems science.",
    "econ": "Economics working papers.",
}

ARXIV_CATEGORY_HINTS: dict[str, str] = {
    "cs.AI": "Artificial intelligence, agents, and reasoning.",
    "cs.LG": "Machine learning models, training, and evaluation.",
    "cs.CL": "NLP, LLMs, and computational linguistics.",
    "cs.CV": "Computer vision and multimodal perception.",
    "stat.ML": "Statistical learning and theory.",
}

IA_COLLECTION_HINTS: dict[str, str] = {
    "opensource": "Community texts and public-domain uploads.",
    "gutenberg": "Mirrored Project Gutenberg corpora.",
    "internetarchivebooks": "Scanned books and lending library texts.",
    "nasa": "NASA publications, media, and technical documents.",
    "prelinger": "Historic films and ephemeral media.",
    "folkscanomy": "Community-uploaded manuals and reference works.",
}


def _kind_for_item(
    source_id: str,
    *,
    name: str,
    href: str,
    is_directory: bool,
    subscribable: bool,
    external_url: str | None,
) -> str:
    lower = name.lower()
    if name.startswith("[Next page]"):
        return "pagination"
    if external_url:
        return "external"
    if is_directory:
        if source_id == "archive" and href.startswith("collection/"):
            return "collection"
        if source_id == "archive" and href.startswith("search/"):
            return "search"
        if source_id == "archive" and href.startswith("item/"):
            return "item"
        if source_id == "arxiv" and href.startswith("cat/"):
            return "category"
        if source_id == "arxiv" and href.startswith("prefix/"):
            return "prefix"
        return "folder"
    if subscribable:
        if lower.endswith(".zim"):
            return "zim"
        if lower.endswith(".pdf"):
            return "pdf"
        if lower.endswith((".txt", ".md")):
            return "text"
        if source_id == "arxiv":
            return "paper"
    return "file"


def _zim_details(name: str, package_key: str | None = None) -> dict[str, str]:
    parsed = parse_zim_stamp(name)
    if parsed is None:
        return {}
    base, stamp = parsed
    parts = base.split("_")
    if len(parts) >= 3:
        corpus, lang, topic = parts[0], parts[1], "_".join(parts[2:])
    else:
        corpus, lang, topic = base, "", base
    return {
        "corpus": corpus.replace("_", " "),
        "language": lang.upper(),
        "topic": topic.replace("_", " "),
        "snapshot": stamp.raw,
        "package_key": package_key or base,
    }


def _folder_hint(source_id: str, name: str, href: str, path: str) -> str:
    key = name.rstrip("/").lower()
    if key in FOLDER_HINTS:
        return FOLDER_HINTS[key]
    if source_id == "archive":
        if href.startswith("collection/"):
            coll = href.split("/", 2)[1]
            return IA_COLLECTION_HINTS.get(coll, "Browse items in this Internet Archive collection.")
        if href.startswith("search/"):
            return "Saved search results from Internet Archive."
        if href.startswith("item/"):
            ident = href.split("/", 2)[1]
            return f"Files available for item {ident}."
    if source_id == "arxiv":
        if href.startswith("prefix/"):
            prefix = href.split("/", 2)[1]
            return ARXIV_PREFIX_HINTS.get(prefix, f"Papers under the {prefix} prefix.")
        if href.startswith("cat/"):
            cat = href.split("/", 2)[1].split("/")[0]
            return ARXIV_CATEGORY_HINTS.get(cat, f"Recent papers in {cat}.")
    if path:
        return f"Subfolder of {path.rstrip('/').split('/')[-1]}."
    return "Open to browse contents."


def describe_item(
    source_id: str,
    *,
    name: str,
    href: str,
    path: str,
    is_directory: bool,
    subscribable: bool,
    external_url: str | None,
    modified: str,
    package_key: str | None = None,
    version_stamp: str | None = None,
    hidden_older_versions: int = 0,
) -> dict[str, Any]:
    kind = _kind_for_item(
        source_id,
        name=name,
        href=href,
        is_directory=is_directory,
        subscribable=subscribable,
        external_url=external_url,
    )
    title = name.rstrip("/")
    if kind == "pagination":
        title = "Next page"
    subtitle = ""
    hint = ""

    if kind == "zim":
        details = _zim_details(name, package_key=package_key)
        if details:
            title = details["topic"].title()
            stamp_label = format_version_label(version_stamp or details["snapshot"])
            subtitle = (
                f"{details['corpus'].title()} · {details['language']} · {stamp_label}"
            )
            hint = (
                "Latest release on this mirror. Subscribe to download, chunk, and embed."
            )
            if hidden_older_versions > 0:
                hint += f" ({hidden_older_versions} older dated build(s) hidden.)"
        else:
            hint = "ZIM offline archive. Subscribe to add it to the knowledge base."
    elif kind == "paper":
        if ":" in name:
            paper_id, _, rest = name.partition(":")
            title = rest.strip()[:100]
            subtitle = paper_id.strip()
        hint = "PDF preprint. Subscribe to extract text and index for RAG."
    elif kind == "pdf":
        hint = "PDF document. Subscribe to extract and embed."
    elif kind == "text":
        hint = "Plain-text source suitable for direct embedding."
    elif kind in ("folder", "collection", "category", "prefix", "item", "search"):
        hint = _folder_hint(source_id, name, href, path)
        if kind == "collection":
            subtitle = "Collection"
        elif kind == "category":
            subtitle = href.split("/", 2)[1].split("/")[0] if href.startswith("cat/") else "Category"
    elif kind == "external":
        hint = "External link (opens in a new tab)."
    elif kind == "pagination":
        hint = "Load more results from this catalog."

    if modified and kind not in ("pagination", "external"):
        meta_date = modified[:10] if len(modified) >= 10 else modified
        subtitle = f"{subtitle} · {meta_date}" if subtitle else meta_date

    return {
        "kind": kind,
        "title": title,
        "subtitle": subtitle.strip(" ·"),
        "hint": hint,
        "kind_label": kind.replace("_", " ").upper(),
    }
