// Generates Diplom_2026.docx from markdown chapter sources in
// docs/diploma/chapters/ following the same formatting rules as
// build_kursovaya.js (KFU «Памятка по оформлению ВКР»):
//   * A4, margins L=30/R=15/T=20/B=20 mm
//   * Times New Roman 14, line spacing 1.5
//   * First-line indent 1.25 cm
//   * Page numbers bottom-centre, no number on title page
//   * Headings centred uppercase, no period
//   * References per ГОСТ 7.1-2003
//
// Workflow:
//   1. Each chapter is authored as plain markdown in
//      docs/diploma/chapters/NN-name.md
//   2. This script parses minimal markdown (paragraphs, ## headings,
//      ### subheadings, — / 1. lists) into docx-js Paragraphs
//   3. Title page + table of contents are hand-built (same approach
//      as build_kursovaya.js)
//   4. References list is read from a separate JSON to avoid
//      rebuilding the script on every reference addition
//
// This script is deliberately a SKELETON — at this stage it produces
// only the title page + a hard-coded chapter list pointing at the
// markdown drafts. The full text-to-docx parser will be filled in
// closer to thesis submission, when the chapters have stabilised.

const fs = require('fs');
const path = require('path');
const docx = require('/opt/homebrew/lib/node_modules/docx');
const {
  Document, Packer, Paragraph, TextRun,
  Table, TableRow, TableCell,
  AlignmentType, NumberFormat, PageNumber,
  Footer, HeadingLevel, PageBreak,
  PositionalTab, PositionalTabAlignment,
  PositionalTabRelativeTo, PositionalTabLeader,
  BorderStyle, WidthType, ShadingType, VerticalAlign,
} = docx;

// ── Page geometry (same as kursovaya — KFU GOST) ──────────────────
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: 1134, right: 850, bottom: 1134, left: 1701, header: 720, footer: 720 };
const FONT = 'Times New Roman';
const SZ_BODY = 28; // 14pt
const LINE = { line: 360, lineRule: 'auto' };
const LINE_SINGLE = { line: 240, lineRule: 'auto' };
const INDENT = 708; // 1.25 cm

// ── Body paragraph helpers (same set as kursovaya) ────────────────
function pBody(text, opts = {}) {
  return new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    indent: opts.noIndent ? undefined : { firstLine: INDENT },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    children: [
      new TextRun({ text, font: FONT, size: SZ_BODY, bold: !!opts.bold }),
    ],
  });
}

function pStructural(text, { firstSection = false } = {}) {
  return new Paragraph({
    spacing: { ...LINE, before: 0, after: 240 },
    alignment: AlignmentType.CENTER,
    pageBreakBefore: !firstSection,
    children: [
      new TextRun({ text, font: FONT, size: SZ_BODY, bold: true, allCaps: true }),
    ],
  });
}

function pTOC(label, page, { indent = 0, bold = false } = {}) {
  return new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    indent: { left: indent },
    children: [
      new TextRun({ text: label, font: FONT, size: SZ_BODY, bold }),
      new TextRun({
        font: FONT, size: SZ_BODY, bold,
        children: [
          new PositionalTab({
            alignment: PositionalTabAlignment.RIGHT,
            relativeTo: PositionalTabRelativeTo.MARGIN,
            leader: PositionalTabLeader.DOT,
          }),
          page,
        ],
      }),
    ],
  });
}

function pageFooter() {
  return new Footer({
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({
            font: FONT, size: SZ_BODY,
            children: [PageNumber.CURRENT],
          }),
        ],
      }),
    ],
  });
}

// ── Minimal markdown → Paragraphs converter ──────────────────────
//
// Supports:
//   "## ..." → centered uppercase structural heading
//   "### ..." → bold left-aligned section heading
//   "— ..." or "- ..." → dash list
//   "1. ..." → numbered list
//   "" (blank) → paragraph break
//   anything else → body paragraph (justified, first-line indent)
//
// Markdown-bold (**text**) is parsed into runs with bold=true; the
// rest is plain text. Code fences ``` are skipped (not used in our
// chapter drafts).

// Render a markdown-style pipe table into a docx Table.
function makeMarkdownTable(headerCells, dataRows) {
  // Use full content width (9356 DXA = 165 mm) split equally.
  const totalW = 9356;
  const nCols = headerCells.length;
  const colW = Math.floor(totalW / nCols);
  const widths = Array(nCols).fill(colW);
  // Adjust last to make sum exactly totalW.
  widths[nCols - 1] = totalW - colW * (nCols - 1);

  const border = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
  const borders = { top: border, bottom: border, left: border, right: border };

  function makeCell(text, opts = {}) {
    return new TableCell({
      borders,
      width: { size: opts.width, type: WidthType.DXA },
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({
        alignment: opts.bold ? AlignmentType.CENTER : AlignmentType.LEFT,
        spacing: { ...LINE_SINGLE, before: 0, after: 0 },
        children: [new TextRun({
          text: text.replace(/\*\*/g, ''),
          font: FONT, size: 24,  // 12pt for tables
          bold: !!opts.bold || /^\*\*.*\*\*$/.test(text),
        })],
      })],
    });
  }

  const rows = [];
  rows.push(new TableRow({
    tableHeader: true,
    children: headerCells.map((c, i) =>
      makeCell(c.trim(), { width: widths[i], bold: true }),
    ),
  }));
  for (const row of dataRows) {
    rows.push(new TableRow({
      children: row.map((c, i) => makeCell(c.trim(), { width: widths[i] })),
    }));
  }

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: widths,
    rows,
  });
}

function parseChapterMarkdown(md) {
  const lines = md.split('\n');
  const paras = [];
  let buffer = [];
  let inCode = false;
  let tableHeader = null;     // array of header cell strings
  let tableSeparator = false; // true when we've consumed the |---| line
  let tableRows = [];         // array of arrays

  function flushBuffer() {
    if (buffer.length === 0) return;
    const text = buffer.join(' ').trim();
    buffer = [];
    if (!text) return;
    paras.push(pBody(text));
  }

  function flushTable() {
    if (tableHeader == null) return;
    if (tableRows.length > 0 || tableSeparator) {
      paras.push(makeMarkdownTable(tableHeader, tableRows));
      // Add a blank paragraph after table for spacing.
      paras.push(new Paragraph({
        spacing: { ...LINE, before: 60, after: 60 },
        children: [new TextRun({ text: '', font: FONT, size: SZ_BODY })],
      }));
    }
    tableHeader = null;
    tableSeparator = false;
    tableRows = [];
  }

  for (const raw of lines) {
    const line = raw.trim();

    // Code fences — skip until closed (we don't render code in body)
    if (line.startsWith('```')) {
      flushBuffer();
      flushTable();
      inCode = !inCode;
      continue;
    }
    if (inCode) continue;

    // Markdown pipe-table: lines start and end with '|'.
    if (line.startsWith('|') && line.endsWith('|') && line.length >= 3) {
      flushBuffer();
      const cells = line.slice(1, -1).split('|');
      // Detect separator row like |---|---|
      const isSeparator = cells.every(c => /^[\s:]*-+[\s:]*$/.test(c));
      if (tableHeader == null) {
        tableHeader = cells;
      } else if (isSeparator) {
        tableSeparator = true;
      } else {
        tableRows.push(cells);
      }
      continue;
    } else if (tableHeader != null) {
      // Just exited a table block.
      flushTable();
    }

    // Blank line → paragraph break
    if (!line) {
      flushBuffer();
      continue;
    }

    // Headings
    if (line.startsWith('## ')) {
      flushBuffer();
      flushTable();
      paras.push(pStructural(line.slice(3).toUpperCase()));
      continue;
    }
    if (line.startsWith('### ')) {
      flushBuffer();
      flushTable();
      paras.push(new Paragraph({
        spacing: { ...LINE, before: 240, after: 120 },
        indent: { firstLine: INDENT },
        children: [
          new TextRun({ text: line.slice(4), font: FONT, size: SZ_BODY, bold: true }),
        ],
      }));
      continue;
    }

    // Dash / hyphen list
    if (line.startsWith('— ') || line.startsWith('- ') || line.startsWith('* ')) {
      flushBuffer();
      const item = line.slice(2);
      paras.push(new Paragraph({
        spacing: { ...LINE, before: 0, after: 0 },
        indent: { left: INDENT, hanging: 284 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: '— ' + item, font: FONT, size: SZ_BODY }),
        ],
      }));
      continue;
    }

    // Numbered list (1. 2. ...)
    const numMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      flushBuffer();
      const idx = numMatch[1];
      const item = numMatch[2];
      paras.push(new Paragraph({
        spacing: { ...LINE, before: 0, after: 0 },
        indent: { left: INDENT, hanging: 360 },
        alignment: AlignmentType.JUSTIFIED,
        children: [
          new TextRun({ text: `${idx}. ${item}`, font: FONT, size: SZ_BODY }),
        ],
      }));
      continue;
    }

    // Strip leading "> ..." quote marker (used for editorial notes)
    if (line.startsWith('> ')) {
      // Skip — these are draft annotations, not part of the thesis
      continue;
    }

    // Default: collect into paragraph buffer
    buffer.push(line);
  }
  flushBuffer();
  flushTable();
  return paras;
}

function loadChapter(chapterFile) {
  const fp = path.join(__dirname, '..', 'docs', 'diploma', 'chapters', chapterFile);
  if (!fs.existsSync(fp)) {
    console.warn(`[build_diploma] missing chapter: ${chapterFile} — skipping`);
    return [];
  }
  const md = fs.readFileSync(fp, 'utf-8');
  return parseChapterMarkdown(md);
}

// ──────────────────────────────────────────────────────────────────
// CONTENT: title page + TOC + chapters
// ──────────────────────────────────────────────────────────────────

const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE_SINGLE, before: 0, after: 0 },
    children: [new TextRun({ text: 'МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ', font: FONT, size: SZ_BODY, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE_SINGLE, before: 60, after: 240 },
    children: [new TextRun({ text: '«Казанский (Приволжский) федеральный университет»', font: FONT, size: SZ_BODY, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE_SINGLE, before: 0, after: 0 },
    children: [new TextRun({ text: 'ИНСТИТУТ ВЫЧИСЛИТЕЛЬНОЙ МАТЕМАТИКИ И ИНФОРМАЦИОННЫХ ТЕХНОЛОГИЙ', font: FONT, size: SZ_BODY, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE_SINGLE, before: 60, after: 480 },
    children: [new TextRun({ text: 'КАФЕДРА ТЕОРЕТИЧЕСКОЙ КИБЕРНЕТИКИ', font: FONT, size: SZ_BODY, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 0, after: 480 },
    children: [new TextRun({ text: 'Специальность (направление): 01.03.02 — Прикладная математика и информатика', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 0, after: 240 },
    children: [new TextRun({ text: 'ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА', font: FONT, size: 32, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 0, after: 720 },
    children: [new TextRun({
      text: 'Многомасштабное прогнозирование финансовых временных рядов: ансамблевые методы с автоматической детекцией концептуального дрейфа',
      font: FONT, size: SZ_BODY, bold: true,
    })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    children: [new TextRun({ text: 'Работа завершена:', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    children: [new TextRun({ text: 'Студент гр. 09-313', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 240 },
    children: [new TextRun({ text: '«___»_________ 20__ г.   _______________   Н. С. Староверов', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 240, after: 0 },
    children: [new TextRun({ text: 'Работа допущена к защите:', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    children: [new TextRun({ text: 'Научный руководитель,', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 0 },
    children: [new TextRun({ text: 'доцент кафедры теоретической кибернетики', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    spacing: { ...LINE, before: 0, after: 240 },
    children: [new TextRun({ text: '«___»_________ 20__ г.   _______________   А. Ф. Гайнутдинова', font: FONT, size: SZ_BODY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { ...LINE, before: 720, after: 0 },
    children: [new TextRun({ text: 'Казань — 2027', font: FONT, size: SZ_BODY })],
  }),
];

const body = [];

// Table of contents (page numbers placeholder — refined when content stabilises)
body.push(pStructural('ОГЛАВЛЕНИЕ', { firstSection: true }));
body.push(pTOC('ВВЕДЕНИЕ', '4'));
body.push(pTOC('ГЛАВА 1. ТЕОРЕТИЧЕСКИЕ ОСНОВЫ', '8'));
body.push(pTOC('ГЛАВА 2. АРХИТЕКТУРА ВЕБ-СЕРВИСА (ДНЕВНОЙ ГОРИЗОНТ)', '22'));
body.push(pTOC('ГЛАВА 3. ЭКСПЕРИМЕНТЫ НА ДНЕВНОМ ГОРИЗОНТЕ', '32'));
body.push(pTOC('ГЛАВА 4. АРХИТЕКТУРА ВЫСОКОЧАСТОТНОГО СЛАЙСА', '40'));
body.push(pTOC('ГЛАВА 5. ЭКСПЕРИМЕНТЫ НА 1-МИНУТНОМ ГОРИЗОНТЕ', '50'));
body.push(pTOC('ГЛАВА 6. ДЕТЕКЦИЯ КОНЦЕПТУАЛЬНОГО ДРЕЙФА', '58'));
body.push(pTOC('ЗАКЛЮЧЕНИЕ', '66'));
body.push(pTOC('СПИСОК ИСПОЛЬЗОВАННЫХ ИСТОЧНИКОВ', '69'));
body.push(pTOC('ПРИЛОЖЕНИЯ', '72'));

// Chapter content from markdown drafts
const chapters = [
  '01-introduction.md',
  // Theory + Daily-architecture will be ported from Курсовая
  // closer to submission; for now just include the experiments
  // chapter (which has the new extended-experiment data).
  '04-daily-experiments.md',
  '05-hft-architecture.md',
  '06-hft-experiments.md',
  '07-drift-detection.md',
  '08-conclusion.md',
];

for (const ch of chapters) {
  const paras = loadChapter(ch);
  if (paras.length === 0) continue;
  // Force page break between chapters
  body.push(...paras);
}

// ──────────────────────────────────────────────────────────────────
// Build the document
// ──────────────────────────────────────────────────────────────────

const doc = new Document({
  creator: 'Староверов Н.С.',
  title: 'ВКР — Многомасштабное прогнозирование финансовых временных рядов',
  styles: {
    default: { document: { run: { font: FONT, size: SZ_BODY } } },
  },
  sections: [
    {
      properties: { page: { size: PAGE, margin: MARGIN } },
      children: titlePage,
    },
    {
      properties: {
        page: {
          size: PAGE,
          margin: MARGIN,
          pageNumbers: { start: 2, formatType: NumberFormat.DECIMAL },
        },
      },
      footers: { default: pageFooter() },
      children: body,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, '..', 'docs', 'diploma', 'Diplom_2026.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, '(' + buf.length + ' bytes)');
});
