"""
CineMind AI - Chapter Detector
Detects chapter headings from document TOC or regex patterns for chapter-aware chunk metadata.
"""

from dataclasses import dataclass
import re
from typing import Optional


@dataclass
class ChapterInfo:
    title: str
    level: int = 1
    start_page: int = 1
    end_page: Optional[int] = None


class ChapterDetector:
    """Detects and maps document sections and chapters."""

    COMMON_PATTERNS = [
        re.compile(r"^(?:CHAPTER|Chapter)\s+([0-9IVXLCDM]+)[\s:.-]*(.*)$", re.IGNORECASE),
        re.compile(r"^(?:ACT|Act)\s+([0-9IVXLCDM]+)[\s:.-]*(.*)$", re.IGNORECASE),
        re.compile(r"^(?:PART|Part)\s+([0-9IVXLCDM]+)[\s:.-]*(.*)$", re.IGNORECASE),
        re.compile(r"^#+\s+(.+)$"),
    ]

    @staticmethod
    def build_page_to_chapter_map(
        toc: list[list], total_pages: int
    ) -> dict[int, str]:
        """
        Builds a page number -> chapter title mapping from a PDF TOC list.
        Each TOC entry is [level, title, page_number].
        """
        page_map: dict[int, str] = {}
        if not toc:
            return page_map

        # Filter and sort entries by page number
        sorted_toc = sorted([item for item in toc if len(item) >= 3 and item[2] > 0], key=lambda x: x[2])
        if not sorted_toc:
            return page_map

        # Group entries by page or select the most descriptive title
        # For instance, if level 2 is "Reservoir Dogs", combine or use level 2 title
        processed_sections: list[tuple[int, str]] = []
        i = 0
        while i < len(sorted_toc):
            lvl, title, page = sorted_toc[i][:3]
            clean_title = title.strip()
            # If next item has same page and higher level (more specific, e.g. lvl 2), combine or prefer it
            if i + 1 < len(sorted_toc) and sorted_toc[i + 1][2] == page:
                next_lvl, next_title, _ = sorted_toc[i + 1][:3]
                combined = f"{clean_title} - {next_title.strip()}"
                processed_sections.append((page, combined))
                i += 2
            else:
                processed_sections.append((page, clean_title))
                i += 1

        # Now fill pages between checkpoints
        current_idx = 0
        current_title = "Front Matter"

        for p in range(1, total_pages + 1):
            if current_idx < len(processed_sections) and p >= processed_sections[current_idx][0]:
                current_title = processed_sections[current_idx][1]
                current_idx += 1
            page_map[p] = current_title

        return page_map

    @classmethod
    def detect_heading_in_text(cls, text: str) -> Optional[str]:
        """Detects a chapter or section heading in raw text."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:5]:  # check first few lines
            for pattern in cls.COMMON_PATTERNS:
                m = pattern.match(line)
                if m:
                    return line
        return None
