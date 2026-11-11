"""
CineMind AI - Technical Documentation PDF Generator
Parses CineMind_AI_Technical_Documentation.md and renders a publication-quality PDF.
"""

from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Skip running header/footer on cover/first page
        if self._pageNumber > 1:
            # Running header
            self.drawString(
                54,
                750,
                "CineMind AI – Project Architecture, Working & Technology Documentation",
            )
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

            # Running footer
            self.drawString(
                54, 36, "Confidential & Proprietary – CineMind AI Architecture Spec"
            )
            page_text = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(558, 36, page_text)
            self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf():
    base_dir = Path(__file__).resolve().parent.parent
    md_path = base_dir / "CineMind_AI_Technical_Documentation.md"
    pdf_path = base_dir / "CineMind_AI_Technical_Documentation.pdf"

    if not md_path.exists():
        print(f"Error: {md_path} not found!")
        return

    print(f"Reading markdown from {md_path}...")
    md_text = md_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15,
    )

    h1_style = ParagraphStyle(
        "DocH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "DocH3",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "DocBullet",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "DocCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8,
        spaceBefore=4,
    )

    story = []

    # Title Banner
    story.append(Paragraph("CineMind AI – Technical Documentation", title_style))
    story.append(
        Paragraph(
            "Project Architecture, Working & Technology Documentation<br/>"
            "<i>A Complete Reference for B.Tech AIML Students, GenAI Developers & Technical Interviewers</i>",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=colors.HexColor("#d4af37"),
            spaceAfter=15,
        )
    )

    lines = md_text.splitlines()
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []

    def clean_markdown_inline(text: str) -> str:
        # Bold: **text** -> <b>text</b>
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        # Italic: *text* -> <i>text</i>
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        # Inline code: `code` -> <font name="Courier">code</font>
        text = re.sub(
            r"`(.+?)`", r'<font name="Courier" color="#b45309">\1</font>', text
        )
        # Links: [text](url) -> <u>text</u>
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"<u>\1</u>", text)
        return text

    for line in lines:
        stripped = line.strip()

        # Handle code block toggle
        if stripped.startswith("```"):
            if in_code_block:
                # Flush code block
                code_text = "\n".join(code_lines)
                # Escape XML entities in code
                code_text = (
                    code_text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                story.append(Paragraph(code_text.replace("\n", "<br/>"), code_style))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Handle Markdown Tables
        if "|" in line and (line.startswith("|") or line.endswith("|")):
            if not in_table:
                in_table = True
                table_lines = []
            # Skip separator line like |---|---|
            if re.match(r"^\|?[\s\-\:]+\|[\s\-\:\|]+$", stripped):
                continue
            table_lines.append(stripped)
            continue
        else:
            if in_table:
                # Flush table
                in_table = False
                if table_lines:
                    table_data = []
                    for t_row in table_lines:
                        cells = [c.strip() for c in t_row.strip("|").split("|")]
                        para_cells = [
                            Paragraph(clean_markdown_inline(c), body_style)
                            for c in cells
                        ]
                        table_data.append(para_cells)

                    # Build TableFlowable
                    col_count = len(table_data[0])
                    avail_width = 504  # 612 - 108 margins
                    col_w = avail_width / max(1, col_count)
                    t = Table(table_data, colWidths=[col_w] * col_count)
                    t.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                (
                                    "BOTTOMPADDING",
                                    (0, 0),
                                    (-1, -1),
                                    4,
                                ),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.5,
                                    colors.HexColor("#cbd5e1"),
                                ),
                            ]
                        )
                    )
                    story.append(t)
                    story.append(Spacer(1, 8))
                table_lines = []

        if not stripped:
            continue

        # Headings
        if stripped.startswith("# "):
            heading_text = clean_markdown_inline(stripped[2:])
            story.append(Paragraph(heading_text, h1_style))
        elif stripped.startswith("## "):
            heading_text = clean_markdown_inline(stripped[3:])
            story.append(Paragraph(heading_text, h2_style))
        elif stripped.startswith("### "):
            heading_text = clean_markdown_inline(stripped[4:])
            story.append(Paragraph(heading_text, h3_style))
        elif stripped.startswith("#### "):
            heading_text = clean_markdown_inline(stripped[5:])
            story.append(Paragraph(f"<b>{heading_text}</b>", body_style))
        elif stripped.startswith("---"):
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#cbd5e1"),
                    spaceBefore=6,
                    spaceAfter=8,
                )
            )
        elif stripped.startswith("- ") or stripped.startswith("* "):
            bullet_text = clean_markdown_inline(stripped[2:])
            story.append(
                Paragraph(f"&bull; {bullet_text}", bullet_style)
            )
        elif re.match(r"^\d+\.\s", stripped):
            num_match = re.match(r"^(\d+\.)\s*(.*)", stripped)
            prefix = num_match.group(1)
            content = clean_markdown_inline(num_match.group(2))
            story.append(
                Paragraph(f"<b>{prefix}</b> {content}", bullet_style)
            )
        elif stripped.startswith("> "):
            quote_text = clean_markdown_inline(stripped[2:])
            quote_style = ParagraphStyle(
                "QuoteStyle",
                parent=body_style,
                leftIndent=15,
                fontName="Helvetica-Oblique",
                textColor=colors.HexColor("#334155"),
                backColor=colors.HexColor("#f8fafc"),
                borderColor=colors.HexColor("#d4af37"),
                borderWidth=1,
                borderPadding=6,
                spaceAfter=6,
            )
            story.append(Paragraph(quote_text, quote_style))
        else:
            p_text = clean_markdown_inline(stripped)
            story.append(Paragraph(p_text, body_style))

    print("Building publication PDF with dynamic running headers & page counts...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF documentation: {pdf_path} ({pdf_path.stat().st_size} bytes)")


if __name__ == "__main__":
    build_pdf()
