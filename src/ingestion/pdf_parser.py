"""
PDF 论文解析引擎
================
支持中英文论文，处理双栏排版、公式保留、图表标题提取、参考文献切除。
"""
import sys
from pathlib import Path
# 确保项目根目录在 sys.path 中（支持直接运行和 import）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import re
import logging
from typing import Optional
import pdfplumber
from config import (
    PDF_DIR, MIN_PARAGRAPH_LEN, REMOVE_HEADER_FOOTER, REMOVE_REFERENCES,
)

logger = logging.getLogger(__name__)

# ===== 正则 =====
LATEX_BLOCK = re.compile(r"\$\$.*?\$\$", re.DOTALL)
LATEX_INLINE = re.compile(r"\$(?!\$).+?\$")
FIGURE_CAPTION = re.compile(
    r"(?:图|Fig(?:ure)?\.?)\s*\d[\d.\-]*\s*[：:.\s].*", re.IGNORECASE
)
TABLE_CAPTION = re.compile(
    r"(?:表|Table)\s*\d[\d.\-]*\s*[：:.\s].*", re.IGNORECASE
)
FORMULA_REF = re.compile(
    r"(?:公式|式|Eq(?:uation)?\.?)\s*[\(（]\s*\d[\d.\-]*\s*[\)）]", re.IGNORECASE
)

# ===== 章节分割正则 =====
SECTION_PATTERN = re.compile(
    r"^(?:"
    r"(?:第\s*[一二三四五六七八九十\d]+\s*[章节])|"           # 第1章 / 第一章
    r"(?:[IVX]+\.\s+\S)|"                                    # I. Introduction (罗马数字)
    r"(?:\d+(?:\.\d+)*\s+\S)|"                               # 1.1 xxx
    r"(?:[一二三四五六七八九十]+)[、，.\s]+\S|"                  # 一、xxx
    r"(?:Abstract|Introduction|Related Work|Method|"
    r"Experiment|Result|Conclusion|Discussion|"
    r"摘要|引言|绪论|相关工作|方法|实验|结果|结论|讨论)"
    r")",
    re.IGNORECASE,
)
REFERENCE_HEADER = re.compile(
    r"^(?:References?|参考文献|Bibliography)\s*$", re.IGNORECASE | re.MULTILINE
)


# ===== 工具函数 =====
def _is_header_footer(line: str, page_num: int) -> bool:
    """检测页眉页脚"""
    if len(line) < 8:
        return True
    if re.match(r"^\d{1,4}$", line):
        return True
    if re.match(r"^(第\s*\d+|Page\s*\d+)", line):
        return True
    return False


def _has_formula(text: str) -> bool:
    return bool(LATEX_BLOCK.search(text) or LATEX_INLINE.search(text))


def _detect_reference_start(lines: list[str]) -> int:
    """找到参考文献起始行，返回行号；未找到返回 -1"""
    for i, line in enumerate(lines):
        if REFERENCE_HEADER.match(line.strip()):
            for j in range(i + 1, min(i + 5, len(lines))):
                if re.match(r"^\s*\[\d+\]", lines[j].strip()):
                    return i
    return -1


def _clean_text(text: str) -> str:
    """清洗：合并断行、去空行、过滤短行（保留章节标题）"""
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)       # 英文断词
    text = re.sub(r"\n{3,}", "\n\n", text)                # 合并空行
    lines = [l.strip() for l in text.split("\n")]
    # 保留: 空行 / 长行 / 章节标题（即使短也不过滤）
    lines = [
        l for l in lines
        if l == "" or len(l) >= MIN_PARAGRAPH_LEN or SECTION_PATTERN.match(l)
    ]
    return "\n".join(lines)


def _split_sections(text: str) -> list[dict]:
    """按论文章节标题切分"""
    paragraphs = text.split("\n\n")
    sections: list[dict] = []
    current_heading = "摘要 / Abstract"
    current_content: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        first_line = lines[0].strip()
        if SECTION_PATTERN.match(first_line) and len(first_line) < 120:
            if current_content:
                content = "\n".join(current_content)
                sections.append({
                    "heading": current_heading,
                    "content": content,
                    "has_formula": _has_formula(content),
                })
            current_heading = first_line
            current_content = lines[1:] if len(lines) > 1 else []
        else:
            current_content.append(para)

    if current_content:
        content = "\n".join(current_content)
        sections.append({
            "heading": current_heading,
            "content": content,
            "has_formula": _has_formula(content),
        })
    return sections


# ===== 元数据提取 =====
def _extract_metadata(pdf_path: Path) -> dict:
    """从 PDF 首页提取标题和作者"""
    meta = {"title": pdf_path.stem, "authors": "Unknown", "source_file": pdf_path.name}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return meta
            text = pdf.pages[0].extract_text()
            if not text:
                return meta
            lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 10]
            if not lines:
                return meta
            # 跳过版权、水印行
            skip_prefixes = ("©", "IEEE", "DIGITAL", "Author", "Manuscript")
            title_line = ""
            for line in lines:
                if not any(line.startswith(p) for p in skip_prefixes):
                    title_line = line
                    break
            meta["title"] = (title_line or lines[0])[:200]
            # 作者行：通常紧接标题且含逗号/数字/@
            for line in lines[1:6]:
                if any(c in line for c in [",", "·", "1", "2", "@"]):
                    meta["authors"] = line[:200]
                    break
    except Exception as e:
        logger.warning(f"元数据提取失败 {pdf_path.name}: {e}")
    return meta


# ===== 主解析 =====
def parse_pdf(pdf_path: Path) -> Optional[dict]:
    """
    解析单篇 PDF 论文。
    返回:
        {title, authors, source_file, full_text, sections, figures, formulas}
    失败返回 None。
    """
    logger.info(f"解析: {pdf_path.name}")
    meta = _extract_metadata(pdf_path)
    all_lines: list[str] = []
    figures: list[str] = []
    formulas: list[str] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                page_lines = []
                for line in text.split("\n"):
                    stripped = line.strip()
                    if not stripped:
                        page_lines.append("")
                        continue
                    if REMOVE_HEADER_FOOTER and _is_header_footer(stripped, page_num):
                        continue
                    if FIGURE_CAPTION.match(stripped):
                        figures.append(f"[P{page_num}] {stripped}")
                    if TABLE_CAPTION.match(stripped):
                        figures.append(f"[P{page_num}] {stripped}")
                    formulas.extend(LATEX_BLOCK.findall(stripped))
                    formulas.extend(FORMULA_REF.findall(stripped))
                    page_lines.append(stripped)

                if page_num == len(pdf.pages) and REMOVE_REFERENCES:
                    ref = _detect_reference_start(page_lines)
                    if ref >= 0:
                        page_lines = page_lines[:ref]

                all_lines.extend(page_lines)

    except Exception as e:
        logger.error(f"PDF 解析失败 {pdf_path.name}: {e}")
        return None

    full_text = _clean_text("\n".join(all_lines))
    sections = _split_sections(full_text)

    result = {
        "title": meta["title"],
        "authors": meta["authors"],
        "source_file": meta["source_file"],
        "full_text": full_text,
        "sections": sections,
        "figures": figures,
        "formulas": formulas,
    }
    logger.info(
        f"  ✓ {pdf_path.name} | {len(sections)}章 | "
        f"{len(figures)}图 | {len(formulas)}公式"
    )
    return result


# ===== 批量解析 =====
def ingest_all(pdf_dir: Path = None) -> list[dict]:
    """解析 pdf_dir 下所有 PDF，返回结果列表"""
    pdf_dir = pdf_dir or PDF_DIR
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"未找到 PDF 文件: {pdf_dir}")
        return []
    logger.info(f"找到 {len(pdf_files)} 篇 PDF，开始解析...")
    results = []
    for fp in pdf_files:
        parsed = parse_pdf(fp)
        if parsed:
            results.append(parsed)
    logger.info(f"解析完成: {len(results)}/{len(pdf_files)} 成功")
    return results


# ===== 快速测试 =====
if __name__ == "__main__":
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ {PDF_DIR} 下没有 PDF 文件")
        exit(1)

    # 找一篇英文论文测试: Candes & Wakin 2008
    target = None
    for f in pdf_files:
        if "Demir" in f.name:
            target = f
            break
    path = target or pdf_files[0]
    print(f"📄 测试: {path.name}\n")
    r = parse_pdf(path)

    if r:
        print(f"  标题:    {r['title'][:60]}")
        print(f"  作者:    {r['authors'][:60]}")
        print(f"  章节数:  {len(r['sections'])}")
        print(f"  图表数:  {len(r['figures'])}")
        print(f"  公式数:  {len(r['formulas'])}")
        print(f"  全文:    {len(r['full_text'])} 字符\n")
        print(f"  --- {len(r['sections'])} 章 ---")
        for s in r["sections"]:
            print(f"  [{s['heading'][:50]}]  {len(s['content'])}字  {'📐' if s['has_formula'] else ''}")
    else:
        print("❌ 解析失败")
