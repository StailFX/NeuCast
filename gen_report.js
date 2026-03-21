const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat,
} = require("docx");

// ── Helpers ──
const DXA_CM = (cm) => Math.round(cm * 567); // cm to DXA
const PAGE_W = DXA_CM(21);   // A4 width
const PAGE_H = DXA_CM(29.7); // A4 height
const ML = DXA_CM(2.5);
const MR = DXA_CM(1.5);
const MT = DXA_CM(2);
const MB = DXA_CM(1);
const CONTENT_W = PAGE_W - ML - MR;

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, font: "Times New Roman", size: 28, ...opts }));
  } else if (Array.isArray(text)) {
    text.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun({ text: t, font: "Times New Roman", size: 28 }));
      else runs.push(new TextRun({ font: "Times New Roman", size: 28, ...t }));
    });
  }
  return new Paragraph({
    spacing: { line: 360, after: 120 },
    alignment: opts.center ? AlignmentType.CENTER : (opts.justify !== false ? AlignmentType.JUSTIFIED : AlignmentType.LEFT),
    indent: opts.noIndent ? undefined : { firstLine: DXA_CM(1.25) },
    ...(opts.paragraphOpts || {}),
    children: runs,
  });
}

function heading(text, level = 1) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : (level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3),
    spacing: { before: 360, after: 200, line: 360 },
    alignment: level === 1 ? AlignmentType.CENTER : AlignmentType.LEFT,
    children: [new TextRun({ text, font: "Times New Roman", size: level === 1 ? 32 : (level === 2 ? 30 : 28), bold: true })],
  });
}

function tableRow(cells, header = false) {
  return new TableRow({
    children: cells.map((text, i) => new TableCell({
      borders,
      margins: cellMargins,
      width: { size: Math.floor(CONTENT_W / cells.length), type: WidthType.DXA },
      shading: header ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
      children: [new Paragraph({
        spacing: { line: 276 },
        alignment: AlignmentType.LEFT,
        children: [new TextRun({ text: String(text), font: "Times New Roman", size: 24, bold: header })],
      })],
    })),
  });
}

function simpleTable(headers, rows) {
  const colW = headers.map(() => Math.floor(CONTENT_W / headers.length));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: colW,
    rows: [
      tableRow(headers, true),
      ...rows.map(r => tableRow(r)),
    ],
  });
}

function emptyLine() {
  return new Paragraph({ spacing: { after: 0, line: 360 }, children: [] });
}

// ── Document ──
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 28 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 180 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [
    // ════════════ TITLE PAGE ════════════
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MT, bottom: MB, left: ML, right: MR },
        },
      },
      children: [
        emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        emptyLine(), emptyLine(), emptyLine(),
        p("ОТЧЁТ-СПРАВКА", { center: true, bold: true, size: 36, noIndent: true }),
        p("по проекту", { center: true, noIndent: true }),
        emptyLine(),
        p("NeuCast — AI-платформа прогнозирования финансовых активов", { center: true, bold: true, size: 32, noIndent: true }),
        emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        p("Стекинг-ансамбль: TCN + CatBoost + XGBoost + LightGBM", { center: true, noIndent: true }),
        p("с Monte Carlo симуляциями для многодневных прогнозов", { center: true, noIndent: true }),
        emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        emptyLine(), emptyLine(), emptyLine(), emptyLine(), emptyLine(),
        p("2026", { center: true, noIndent: true }),
      ],
    },

    // ════════════ TABLE OF CONTENTS ════════════
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MT, bottom: MB, left: ML, right: MR },
        },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ font: "Times New Roman", size: 20, color: "888888", children: [PageNumber.CURRENT] })],
        })] }),
      },
      children: [
        heading("СОДЕРЖАНИЕ", 1),
        emptyLine(),
        p("1. Введение и постановка задачи", { noIndent: true }),
        p("2. Теоретические основы", { noIndent: true }),
        p("  2.1. Временные свёрточные сети (TCN)", { noIndent: true }),
        p("  2.2. Механизм Squeeze-and-Excitation", { noIndent: true }),
        p("  2.3. Multi-Scale Inception", { noIndent: true }),
        p("  2.4. Градиентный бустинг (CatBoost, XGBoost, LightGBM)", { noIndent: true }),
        p("  2.5. Стекинг-ансамбль (Ridge meta-learner)", { noIndent: true }),
        p("  2.6. Monte Carlo GBM (Geometric Brownian Motion)", { noIndent: true }),
        p("  2.7. Технические индикаторы", { noIndent: true }),
        p("3. Архитектура системы", { noIndent: true }),
        p("  3.1. Общая архитектура", { noIndent: true }),
        p("  3.2. Архитектура нейронной сети TCN", { noIndent: true }),
        p("  3.3. Pipeline предсказания", { noIndent: true }),
        p("  3.4. Схема базы данных", { noIndent: true }),
        p("4. Реализация", { noIndent: true }),
        p("  4.1. Стек технологий", { noIndent: true }),
        p("  4.2. Структура проекта", { noIndent: true }),
        p("  4.3. Feature Engineering", { noIndent: true }),
        p("  4.4. Обучение модели", { noIndent: true }),
        p("  4.5. Асинхронная обработка (Celery + Redis)", { noIndent: true }),
        p("  4.6. Развёртывание (Docker)", { noIndent: true }),
        p("5. Результаты и метрики", { noIndent: true }),
        p("6. Заключение", { noIndent: true }),
      ],
    },

    // ════════════ MAIN CONTENT ════════════
    {
      properties: {
        page: {
          size: { width: PAGE_W, height: PAGE_H },
          margin: { top: MT, bottom: MB, left: ML, right: MR },
        },
      },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ font: "Times New Roman", size: 20, color: "888888", children: [PageNumber.CURRENT] })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "NeuCast — AI Financial Forecasting Platform", font: "Times New Roman", size: 18, color: "AAAAAA" })],
        })] }),
      },
      children: [
        // ═══ 1. ВВЕДЕНИЕ ═══
        heading("1. ВВЕДЕНИЕ И ПОСТАНОВКА ЗАДАЧИ", 1),

        p("Прогнозирование финансовых временных рядов является одной из ключевых задач количественного анализа. Современные подходы на основе глубокого обучения позволяют выявлять сложные нелинейные зависимости в данных, которые недоступны классическим статистическим методам."),
        p("NeuCast — это полнофункциональная веб-платформа для прогнозирования финансовых активов, использующая стекинг-ансамбль из четырёх моделей машинного обучения. Система получает исторические данные через Yahoo Finance API, применяет обширный набор технических индикаторов и генерирует прогнозы с доверительными интервалами на основе Monte Carlo симуляций."),

        p([{ text: "Цель проекта: ", bold: true }, "разработка интеллектуальной системы прогнозирования цен финансовых активов с веб-интерфейсом, обеспечивающей точность прогноза MAPE < 2% на тестовой выборке."]),

        p([{ text: "Задачи:", bold: true }]),
        p("1) Реализовать multi-target TCN (Temporal Convolutional Network) с двойным выходом: регрессия log-return и классификация направления движения цены.", { noIndent: true }),
        p("2) Построить стекинг-ансамбль из 4 моделей (TCN + CatBoost + XGBoost + LightGBM) с Ridge meta-learner.", { noIndent: true }),
        p("3) Разработать pipeline feature engineering с 28 техническими индикаторами.", { noIndent: true }),
        p("4) Реализовать Monte Carlo прогнозирование (GBM, 1000 симуляций) для многодневных прогнозов.", { noIndent: true }),
        p("5) Создать веб-приложение (FastAPI) с асинхронной обработкой задач (Celery + Redis).", { noIndent: true }),
        p("6) Обеспечить контейнеризацию и развёртывание (Docker Compose, PostgreSQL, Nginx).", { noIndent: true }),

        // ═══ 2. ТЕОРИЯ ═══
        heading("2. ТЕОРЕТИЧЕСКИЕ ОСНОВЫ", 1),

        // 2.1 TCN
        heading("2.1. Временные свёрточные сети (TCN)", 2),

        p("Temporal Convolutional Network (TCN) — архитектура глубокого обучения, специально разработанная для обработки последовательных данных. В отличие от рекуррентных сетей (LSTM, GRU), TCN использует причинные свёрточные слои (causal convolutions), которые гарантируют, что предсказание на момент t зависит только от данных до момента t."),

        p([{ text: "Ключевые принципы TCN:", bold: true }]),

        p([{ text: "Dilated Causal Convolutions. ", bold: true }, "Для увеличения рецептивного поля без роста числа параметров TCN использует расширенные свёртки (dilated convolutions). Фактор расширения (dilation rate) растёт экспоненциально: d = 1, 2, 4, 8, 16, 32. При размере ядра k = 3 рецептивное поле составляет:"]),

        p("RF = k × (2^L - 1) = 3 × (2^6 - 1) = 189 временных шагов", { center: true, noIndent: true, paragraphOpts: { spacing: { before: 120, after: 120, line: 360 } } }),

        p("где L = 6 — число слоёв TCN. Это означает, что сеть может учитывать контекст в 189 торговых дней (~9 месяцев), что существенно превышает входное окно в 60 дней."),

        p([{ text: "Residual Connections. ", bold: true }, "Каждый TCN-блок использует остаточные связи (residual connections). Если входная и выходная размерности не совпадают, применяется линейная проекция через Conv1D(1). Выходное значение: y = GELU(F(x) + Res(x)), где F(x) — результат свёрточных операций, Res(x) — либо x (если размерности совпадают), либо Conv1D(x)."]),

        p([{ text: "Преимущества TCN перед LSTM:", bold: true }]),
        p("1) Параллельная обработка: свёрточные операции обрабатывают всю последовательность одновременно, в отличие от последовательного LSTM.", { noIndent: true }),
        p("2) Стабильный градиент: отсутствие проблемы vanishing/exploding gradient.", { noIndent: true }),
        p("3) Гибкое рецептивное поле: экспоненциальный рост через dilation.", { noIndent: true }),
        p("4) Меньшее потребление памяти при обучении.", { noIndent: true }),

        // 2.2 SE
        heading("2.2. Механизм Squeeze-and-Excitation (SE)", 2),

        p("Squeeze-and-Excitation Block — механизм адаптивной рекалибровки каналов (channel attention). Впервые предложен в работе Hu et al. (2018) для компьютерного зрения. В контексте финансовых временных рядов SE-блок обучается взвешивать каналы (признаки) по их важности для текущего прогноза."),

        p([{ text: "Алгоритм SE-блока:", bold: true }]),
        p("1) Squeeze: глобальное усреднение по временной оси (Global Average Pooling). Результат — вектор размерности (channels,).", { noIndent: true }),
        p("2) Excitation: два полносвязных слоя. Первый сжимает до channels/ratio (ratio = 4) с активацией ReLU, второй восстанавливает размерность с активацией Sigmoid.", { noIndent: true }),
        p("3) Scale: поэлементное умножение исходного тензора на полученные веса.", { noIndent: true }),

        p("Формально: SE(x) = x ⊙ σ(W₂ · ReLU(W₁ · GAP(x))), где GAP — Global Average Pooling, σ — сигмоида, ⊙ — поэлементное умножение."),

        // 2.3 Inception
        heading("2.3. Multi-Scale Inception", 2),

        p("Inception-вход — параллельное применение свёрток с разными размерами ядер для захвата паттернов различного масштаба. В NeuCast используются три параллельные ветви:"),

        simpleTable(
          ["Ветвь", "Размер ядра", "Фильтры", "Назначение"],
          [
            ["Branch 1", "3", "48", "Краткосрочные паттерны (3 дня)"],
            ["Branch 2", "5", "48", "Среднесрочные паттерны (неделя)"],
            ["Branch 3", "7", "48", "Долгосрочные тренды"],
          ],
        ),
        emptyLine(),

        p("Выходы всех ветвей конкатенируются, формируя тензор размерности (batch, 60, 144), который затем нормализуется через LayerNormalization."),

        // 2.4 Boosting
        heading("2.4. Градиентный бустинг", 2),

        p("Градиентный бустинг — ансамблевый метод, последовательно строящий слабые модели (деревья решений), каждая из которых корректирует ошибки предыдущих. В NeuCast используются три реализации:"),

        p([{ text: "CatBoost ", bold: true }, "— gradient boosting от Яндекса. Особенности: нативная обработка категориальных признаков, Ordered Boosting для предотвращения target leakage, symmetric trees. Гиперпараметры: iterations=1000, learning_rate=0.03, depth=7, l2_leaf_reg=3, random_strength=0.5, bagging_temperature=0.3, early_stopping=80."]),

        p([{ text: "XGBoost ", bold: true }, "(eXtreme Gradient Boosting) — высокоэффективная реализация gradient boosting с L1/L2 регуляризацией, Column Subsampling и Shrinkage. Гиперпараметры: n_estimators=1000, learning_rate=0.03, max_depth=7, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0."]),

        p([{ text: "LightGBM ", bold: true }, "(Light Gradient Boosting Machine) от Microsoft — использует leaf-wise стратегию роста деревьев вместо level-wise, что даёт более точные модели при той же глубине. Гиперпараметры: n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63, subsample=0.8, colsample_bytree=0.7."]),

        p([{ text: "Признаки для бустинга. ", bold: true }, "Из скользящего окна в 60 дней для каждого из 9 входных столбцов извлекаются 7 агрегированных статистик: последнее значение, среднее, стандартное отклонение, разность (last - first), MA за 5 дней, минимум, максимум. Итого: 9 × 7 = 63 признака."]),

        // 2.5 Stacking
        heading("2.5. Стекинг-ансамбль (Ridge meta-learner)", 2),

        p("Стекинг (Stacked Generalization, Wolpert, 1992) — метод комбинирования моделей, при котором предсказания базовых моделей используются как входные признаки для мета-модели."),

        p([{ text: "Архитектура стекинга в NeuCast:", bold: true }]),
        p("1) Базовые модели: TCN (deep learning) + CatBoost + XGBoost + LightGBM (gradient boosting).", { noIndent: true }),
        p("2) Мета-модель: Ridge Regression (линейная регрессия с L2-регуляризацией, alpha=1.0).", { noIndent: true }),
        p("3) Обучение мета-модели: только на предсказаниях train-части (нет утечки данных).", { noIndent: true }),
        p("4) Финальный прогноз: Ridge комбинирует предсказания всех моделей с обученными весами.", { noIndent: true }),

        p("Использование Ridge вместо более сложных мета-моделей обусловлено малой размерностью входа (4 признака) и необходимостью предотвращения переобучения мета-модели."),

        // 2.6 Monte Carlo
        heading("2.6. Monte Carlo GBM (Geometric Brownian Motion)", 2),

        p("Для прогнозирования на несколько дней вперёд используется Geometric Brownian Motion — стандартная модель в количественных финансах (Black-Scholes framework). Эволюция цены задаётся стохастическим дифференциальным уравнением:"),

        p("dS = μ·S·dt + σ·S·dW", { center: true, noIndent: true, paragraphOpts: { spacing: { before: 120, after: 120, line: 360 } } }),

        p("где S — цена актива, μ — drift (среднее направление), σ — волатильность, dW — Винеровский процесс (случайное блуждание)."),

        p([{ text: "Дискретная форма:", bold: true }]),
        p("S(t+1) = S(t) × exp((μ - σ²/2)·Δt + σ·√Δt·Z)", { center: true, noIndent: true }),
        emptyLine(),
        p("где Z ~ N(0, 1) — стандартное нормальное распределение."),

        p([{ text: "Параметры в NeuCast:", bold: true }]),
        p("1) Число симуляций: N = 1000 траекторий.", { noIndent: true }),
        p("2) Drift: μ = 0.7 × model_signal + 0.3 × μ_hist — взвешенная комбинация сигнала модели (70%) и исторического среднего (30%).", { noIndent: true }),
        p("3) Волатильность: σ = std(log_returns[-60:]) — оценка по последним 60 дням.", { noIndent: true }),
        p("4) Доверительные интервалы: перцентили 5%, 25%, 50% (медиана), 75%, 95%.", { noIndent: true }),

        // 2.7 Indicators
        heading("2.7. Технические индикаторы", 2),

        p("Система использует 28 признаков, из которых 5 базовых (OHLCV) и 23 производных:"),

        simpleTable(
          ["Индикатор", "Формула/Параметры", "Назначение"],
          [
            ["RSI (14)", "100 - 100/(1 + avg_gain/avg_loss)", "Перекупленность/перепроданность"],
            ["MACD (12/26/9)", "EMA(12) - EMA(26), Signal = EMA(MACD, 9)", "Моментум и тренд"],
            ["MA_5, 10, 20, 50", "Simple Moving Average", "Уровни поддержки/сопротивления"],
            ["Bollinger Bands", "MA(20) ± 2×std(20)", "Волатильность и каналы"],
            ["BB_pct", "(Close - Lower) / (Upper - Lower)", "Позиция в канале (0-1)"],
            ["ATR (14)", "MA(TrueRange, 14)", "Средняя волатильность"],
            ["ROC_5, ROC_10", "pct_change(5), pct_change(10)", "Темп изменения"],
            ["Momentum", "Close - Close(t-10)", "Абсолютный моментум"],
            ["Volatility_20", "std(daily_returns, 20)", "Режим волатильности"],
            ["Volume_Ratio", "Volume / MA(Volume, 20)", "Аномалии объёма"],
            ["log_return", "ln(Close_t / Close_{t-1})", "Целевая переменная"],
          ],
        ),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ 3. АРХИТЕКТУРА ═══
        heading("3. АРХИТЕКТУРА СИСТЕМЫ", 1),

        heading("3.1. Общая архитектура", 2),

        p("Система построена по микросервисной архитектуре с чётким разделением ответственности:"),
        emptyLine(),

        simpleTable(
          ["Компонент", "Технология", "Назначение"],
          [
            ["Web Server", "FastAPI + Uvicorn", "HTTP API, SSR шаблоны, REST endpoints"],
            ["Task Queue", "Celery + Redis", "Фоновые ML задачи, прогресс-трекинг"],
            ["Database", "PostgreSQL / SQLite", "Пользователи, история прогнозов, котировки"],
            ["ML Engine", "TensorFlow + Sklearn", "TCN, бустинг, стекинг, Monte Carlo"],
            ["Data Source", "Yahoo Finance API", "Исторические котировки (OHLCV)"],
            ["Reverse Proxy", "Nginx + Let's Encrypt", "SSL, кэширование, балансировка"],
            ["Containers", "Docker Compose", "Оркестрация всех сервисов"],
          ],
        ),

        emptyLine(),
        p("Взаимодействие компонентов: пользователь отправляет запрос через веб-интерфейс, FastAPI направляет задачу в Celery через Redis. Celery worker загружает данные с Yahoo Finance, выполняет предсказание ансамблем моделей и сохраняет результат в Redis. Клиент получает уведомление через polling (каждые 1.5 секунды) и перенаправляется на страницу результатов."),

        // 3.2 TCN Architecture
        heading("3.2. Архитектура нейронной сети TCN", 2),

        p([{ text: "Вход: ", bold: true }, "(batch_size, 60, 9) — 60 временных шагов × 9 признаков."]),
        emptyLine(),

        simpleTable(
          ["Слой", "Операция", "Выход"],
          [
            ["1. Inception", "3 параллельные Conv1D (k=3,5,7) × 48 фильтров + Concat + LayerNorm", "(batch, 60, 144)"],
            ["2. TCN Block 1", "Conv1D(144, k=3, d=1) → LN → GELU → Drop(0.15) → Conv1D → LN → SE → Residual", "(batch, 60, 144)"],
            ["3. TCN Block 2", "Аналогично, dilation_rate=2", "(batch, 60, 144)"],
            ["4. TCN Block 3", "dilation_rate=4", "(batch, 60, 144)"],
            ["5. TCN Block 4", "dilation_rate=8", "(batch, 60, 144)"],
            ["6. TCN Block 5", "dilation_rate=16", "(batch, 60, 144)"],
            ["7. TCN Block 6", "dilation_rate=32", "(batch, 60, 144)"],
            ["8. Fusion", "GlobalAvgPool ⊕ LastTimestep → Concat", "(batch, 288)"],
            ["9. Dense 1", "Dense(192, GELU) → Dropout(0.25)", "(batch, 192)"],
            ["10. Dense 2", "Dense(96, GELU) → Dropout(0.2)", "(batch, 96)"],
            ["11a. Return Head", "Dense(48, GELU) → Dense(1, linear)", "(batch, 1)"],
            ["11b. Direction Head", "Dense(48, GELU) → Dense(1, sigmoid)", "(batch, 1)"],
          ],
        ),

        emptyLine(),
        p([{ text: "Multi-task обучение: ", bold: true }, "общая функция потерь L = 1.0 × MSE(return) + 0.5 × BCE(direction). Веса 1.0 и 0.5 отражают приоритет регрессии доходности над классификацией направления."]),
        p([{ text: "Direction head adjustment: ", bold: true }, "если direction > 0.6, а предсказанный return < 0 — return корректируется на +|return|×0.5. Аналогично для direction < 0.4 с положительным return. Это позволяет модели самокорректировать знак прогноза."]),

        // 3.3 Pipeline
        heading("3.3. Pipeline предсказания", 2),

        p("Полный pipeline при получении запроса пользователя:"),
        p("1) Загрузка данных: yfinance.download(ticker, start, end, interval='1d').", { noIndent: true }),
        p("2) Feature engineering: расчёт 23 производных индикаторов, удаление NaN.", { noIndent: true }),
        p("3) Нормализация: MinMaxScaler([0,1]) обучается только на train-данных (80%).", { noIndent: true }),
        p("4) Формирование последовательностей: скользящее окно 60 дней → (X, y).", { noIndent: true }),
        p("5) Fine-tuning TCN: clone_model → обучение 10 эпох на train, batch=32.", { noIndent: true }),
        p("6) Предсказание TCN: log returns + direction для всех данных.", { noIndent: true }),
        p("7) Обучение и предсказание CatBoost, XGBoost, LightGBM на 63 агрегированных признаках.", { noIndent: true }),
        p("8) Стекинг: Ridge meta-learner комбинирует 4 предсказания.", { noIndent: true }),
        p("9) Реконструкция цен: P(t) = P(t-1) × exp(pred_return).", { noIndent: true }),
        p("10) Monte Carlo (если days_ahead > 0): 1000 GBM симуляций.", { noIndent: true }),
        p("11) Расчёт метрик: MAE, RMSE, MAPE, R², Direction Accuracy.", { noIndent: true }),

        // 3.4 DB
        heading("3.4. Схема базы данных", 2),

        p("Реляционная база данных содержит 7 таблиц:"),
        emptyLine(),

        simpleTable(
          ["Таблица", "Ключевые поля", "Связи"],
          [
            ["roles", "id, name", "1:N → users"],
            ["users", "id, username, email, password, role_id, created_at", "N:1 → roles, 1:N → prediction_history"],
            ["tickers", "id, symbol, name", "1:N → market_data, 1:N → predictions"],
            ["market_data", "id, ticker_id, date, OHLCV", "N:1 → tickers, 1:N → indicators"],
            ["indicators", "id, market_data_id, name, value", "N:1 → market_data"],
            ["model_info", "id, name, parameters(JSON), mae, rmse", "1:N → predictions"],
            ["predictions", "id, model_id, ticker_id, date, predicted_close", "N:1 → model_info"],
            ["prediction_history", "id, user_id, ticker, dates, model_name, mape, result_json", "N:1 → users"],
          ],
        ),

        emptyLine(),
        p("Хэширование паролей: SHA-256. Сессии: cookie-based (httponly). Результаты прогнозов сохраняются в JSON для повторного просмотра."),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ 4. РЕАЛИЗАЦИЯ ═══
        heading("4. РЕАЛИЗАЦИЯ", 1),

        heading("4.1. Стек технологий", 2),

        simpleTable(
          ["Категория", "Технология", "Версия"],
          [
            ["Backend", "FastAPI (ASGI framework)", "≥ 0.100.0"],
            ["ASGI Server", "Uvicorn", "≥ 0.23.0"],
            ["Templates", "Jinja2", "≥ 3.1.0"],
            ["Deep Learning", "TensorFlow / Keras", "≥ 2.13.0"],
            ["Gradient Boosting", "CatBoost", "≥ 1.2"],
            ["Gradient Boosting", "XGBoost", "≥ 2.0.0"],
            ["Gradient Boosting", "LightGBM", "≥ 4.0.0"],
            ["ML Utils", "scikit-learn", "≥ 1.3.0"],
            ["Data", "pandas, numpy", "≥ 2.0, ≥ 1.24"],
            ["Finance API", "yfinance", "≥ 0.2.18"],
            ["Task Queue", "Celery", "≥ 5.3.0"],
            ["Message Broker", "Redis", "7-alpine"],
            ["Database", "PostgreSQL", "16-alpine"],
            ["ORM", "SQLAlchemy", "≥ 2.0.0"],
            ["PDF Reports", "fpdf2", "≥ 2.7.0"],
            ["Charts", "Plotly.js", "CDN"],
            ["Container", "Docker Compose", "v3.8"],
            ["Web Server", "Nginx", "alpine"],
          ],
        ),

        // 4.2 Structure
        heading("4.2. Структура проекта", 2),

        p("Проект организован по модульному принципу:"),
        emptyLine(),

        simpleTable(
          ["Директория/Файл", "Назначение"],
          [
            ["app/__init__.py", "Экспорт FastAPI приложения"],
            ["app/main.py", "Роуты, middleware, startup, auth"],
            ["app/prediction.py", "ML pipeline: предобработка, TCN, бустинг, стекинг, Monte Carlo"],
            ["app/celery_worker.py", "Celery задачи для фоновых прогнозов"],
            ["app/db.py", "SQLAlchemy engine, session factory"],
            ["app/models.py", "ORM модели (7 таблиц)"],
            ["app/layers.py", "Кастомные Keras слои (TCNBlock, SEBlock, AttentionLayer)"],
            ["weights/", "Обученные модели (best_model.h5, model_config.json, scaler.pkl)"],
            ["training/", "Скрипты обучения (retrain_tcn.py, retrain_returns.py и др.)"],
            ["templates/", "Jinja2 шаблоны (landing, form, predict, waiting, login, register)"],
            ["static/", "Статика (логотип, favicon)"],
            ["Dockerfile", "Образ контейнера (python:3.10-slim)"],
            ["docker-compose.yml", "Оркестрация 5 сервисов"],
          ],
        ),

        // 4.3 Feature Engineering
        heading("4.3. Feature Engineering", 2),

        p("Процесс подготовки признаков из сырых данных OHLCV:"),
        p("1) Интерполяция: метод 'time' для заполнения пропусков в торговых днях.", { noIndent: true }),
        p("2) Расчёт 23 технических индикаторов (RSI, MACD, MA, BB, ATR, ROC, Momentum и др.).", { noIndent: true }),
        p("3) Вычисление log returns: ln(Close_t / Close_{t-1}) — целевая переменная.", { noIndent: true }),
        p("4) Удаление NaN (первые ~50 строк из-за MA_50).", { noIndent: true }),
        p("5) Проверка достаточности данных: минимум 61 строка (SEQ_LEN + 1).", { noIndent: true }),
        p("6) Нормализация MinMaxScaler([0, 1]) — обучается только на train данных для предотвращения утечки.", { noIndent: true }),
        p("7) Формирование обучающих последовательностей: скользящее окно 60 дней.", { noIndent: true }),

        p([{ text: "Важно: ", bold: true }, "нормализация обучается исключительно на train-выборке (первые 80% данных). Тестовые данные нормализуются с теми же параметрами, что гарантирует честность оценки метрик."]),

        // 4.4 Training
        heading("4.4. Обучение модели", 2),

        p("Обучение TCN выполняется скриптом training/retrain_tcn.py:"),
        emptyLine(),

        simpleTable(
          ["Параметр", "Значение", "Обоснование"],
          [
            ["Optimizer", "AdamW (lr=0.001, weight_decay=1e-4)", "Адаптивный с L2 регуляризацией"],
            ["Loss", "MSE(1.0) + BCE(0.5)", "Multi-task: приоритет регрессии"],
            ["Epochs", "120 (max)", "С early stopping"],
            ["Batch size", "64", "Баланс скорости и стабильности"],
            ["EarlyStopping", "patience=20, monitor=val_loss", "Предотвращение переобучения"],
            ["ReduceLROnPlateau", "factor=0.5, patience=7, min_lr=1e-6", "Адаптивный learning rate"],
            ["Train/Val/Test", "70% / 15% / 15%", "Честная оценка"],
            ["Fine-tuning at inference", "10 epochs, batch=32, lr=0.0003", "Адаптация к конкретному тикеру"],
          ],
        ),

        emptyLine(),
        p([{ text: "Fine-tuning при инференсе: ", bold: true }, "при каждом запросе модель клонируется и дообучается на train-части запрошенных данных. Это позволяет адаптировать веса к конкретному инструменту и временному периоду, сохраняя базовые знания предобученной модели."]),

        // 4.5 Celery
        heading("4.5. Асинхронная обработка (Celery + Redis)", 2),

        p("Предсказание занимает 30-120 секунд (fine-tuning + 4 модели), поэтому используется фоновая обработка:"),

        p([{ text: "Архитектура очереди:", bold: true }]),
        p("1) FastAPI получает запрос и создаёт задачу в Redis через Celery.", { noIndent: true }),
        p("2) Пользователь перенаправляется на страницу ожидания (waiting.html).", { noIndent: true }),
        p("3) JavaScript polling каждые 1.5 сек проверяет статус задачи через /api/task/{id}.", { noIndent: true }),
        p("4) Celery worker выполняет предсказание и сохраняет результат в Redis.", { noIndent: true }),
        p("5) При SUCCESS пользователь перенаправляется на страницу результатов.", { noIndent: true }),

        p([{ text: "Оптимизации:", bold: true }]),
        p("1) Semaphore: максимум 2 одновременных предсказания (предотвращение OOM).", { noIndent: true }),
        p("2) TTL Cache: кэширование результатов на 1 час для идентичных запросов.", { noIndent: true }),
        p("3) Worker concurrency: 2 процесса, max 50 задач на процесс (перезапуск для освобождения памяти).", { noIndent: true }),
        p("4) Soft/Hard time limits: 5/6 минут.", { noIndent: true }),
        p("5) Lazy model loading: модель загружается в дочернем процессе при первой задаче (совместимость с fork()).", { noIndent: true }),

        // 4.6 Docker
        heading("4.6. Развёртывание (Docker)", 2),

        p("Docker Compose оркестрирует 5 сервисов:"),
        emptyLine(),

        simpleTable(
          ["Сервис", "Образ", "Порт", "Особенности"],
          [
            ["postgres", "postgres:16-alpine", "5433:5432", "Persistent volume, healthcheck"],
            ["redis", "redis:7-alpine", "internal", "Persistent volume, healthcheck"],
            ["app", "python:3.10-slim (build)", "8100:8100", "Volumes: ./weights:/app/weights"],
            ["celery", "python:3.10-slim (build)", "-", "concurrency=2, max-tasks=50"],
            ["nginx", "nginx:alpine", "80, 443", "Profile: standalone (опционально)"],
          ],
        ),

        emptyLine(),
        p("Продакшн-сервер: VPS (4 CPU, 8 GB RAM, Ubuntu 22.04). PostgreSQL и Redis работают в Docker-контейнерах, приложение и Celery — через systemd (для прямого доступа к GPU/CPU). Nginx — системный, с Let's Encrypt SSL."),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ 5. РЕЗУЛЬТАТЫ ═══
        heading("5. РЕЗУЛЬТАТЫ И МЕТРИКИ", 1),

        p("Результаты оценки на тестовой выборке (Gold Futures, GC=F):"),
        emptyLine(),

        simpleTable(
          ["Метрика", "Значение", "Описание"],
          [
            ["MAE", "$33.90", "Средняя абсолютная ошибка"],
            ["RMSE", "~$45", "Корень из среднеквадратичной ошибки"],
            ["MAPE", "1.04%", "Средняя абсолютная процентная ошибка"],
            ["R²", "0.9962", "Коэффициент детерминации"],
            ["Direction Accuracy", "58%", "Точность предсказания направления"],
          ],
        ),

        emptyLine(),
        p([{ text: "MAPE = 1.04% ", bold: true }, "означает, что средняя ошибка прогноза составляет около $33 при цене золота ~$3000, что является конкурентоспособным результатом для финансового прогнозирования."]),

        p([{ text: "R² = 0.9962 ", bold: true }, "показывает, что модель объясняет 99.62% дисперсии целевой переменной. Это высокое значение обусловлено тем, что модель предсказывает log returns и реконструирует цены, что даёт инерционное преимущество (prices highly autocorrelated)."]),

        p([{ text: "Direction Accuracy = 58% ", bold: true }, "— точность предсказания направления (рост/падение). Значение >50% показывает, что модель лучше случайного угадывания, хотя направление остаётся более сложной задачей, чем регрессия цены."]),

        p([{ text: "Сравнение моделей:", bold: true }]),
        p("Ensemble стабильно показывает лучшие результаты, чем любая из отдельных моделей. TCN превосходит бустинг на трендовых участках, а бустинг — на локальных паттернах. Стекинг через Ridge автоматически определяет оптимальные веса."),

        emptyLine(),
        p([{ text: "Генерация отчётов: ", bold: true }, "система генерирует PDF-отчёты с графиками (цена + прогноз, Bollinger Bands, RSI, MACD), ключевыми метриками и доверительными интервалами Monte Carlo."]),

        new Paragraph({ children: [new PageBreak()] }),

        // ═══ 6. ЗАКЛЮЧЕНИЕ ═══
        heading("6. ЗАКЛЮЧЕНИЕ", 1),

        p("В рамках данного проекта разработана полнофункциональная AI-платформа прогнозирования финансовых активов NeuCast, реализующая современный подход к финансовому прогнозированию на основе стекинг-ансамбля."),

        p([{ text: "Основные результаты:", bold: true }]),
        p("1) Реализована multi-target TCN с двойным выходом (log return + direction), использующая 6 TCN-блоков с экспоненциально растущим dilation, SE-механизмом и Multi-Scale Inception входом.", { noIndent: true }),
        p("2) Построен стекинг-ансамбль из 4 моделей (TCN + CatBoost + XGBoost + LightGBM) с Ridge meta-learner, достигающий MAPE = 1.04% и R² = 0.9962 на тестовой выборке.", { noIndent: true }),
        p("3) Разработан pipeline из 28 технических индикаторов с корректной нормализацией (fit только на train).", { noIndent: true }),
        p("4) Реализованы Monte Carlo симуляции (1000 GBM траекторий) для многодневных прогнозов с доверительными интервалами.", { noIndent: true }),
        p("5) Создано веб-приложение на FastAPI с Celery + Redis для асинхронной обработки, кэшированием (TTL 1 час) и контролем конкурентности (semaphore).", { noIndent: true }),
        p("6) Обеспечена контейнеризация через Docker Compose (5 сервисов) и развёрнут продакшн на VPS с Nginx + Let's Encrypt SSL.", { noIndent: true }),

        p([{ text: "Направления развития:", bold: true }]),
        p("1) Добавление Transformer-based моделей (Temporal Fusion Transformer).", { noIndent: true }),
        p("2) Real-time данные через WebSocket API.", { noIndent: true }),
        p("3) Автоматический подбор гиперпараметров (Optuna).", { noIndent: true }),
        p("4) Расширение набора активов (акции, криптовалюты, индексы).", { noIndent: true }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("/Users/stailfx/Desktop/NeuCast_Report.docx", buffer);
  console.log("Done: /Users/stailfx/Desktop/NeuCast_Report.docx");
});
