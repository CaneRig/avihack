"""Графики для документов и слайдов: SVG без сторонних библиотек -> docs/img/.

Числа взяты из реестра docs/06_experiments.md (в скобках — номер эксперимента). Запуск: python scripts/make_charts.py
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # палитра проверена на различимость при дальтонизме
INK, INK2, GRID, SURF, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeeea"
W, LEFT, RIGHT, ROW = 800, 335, 75, 30
FONT = "font-family:'Segoe UI',Helvetica,Arial,sans-serif"


def num(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}".replace(".", ",")


def frame(title: str, sub: str, h: int, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" aria-label="{title}">'
            f'<rect width="{W}" height="{h}" fill="{SURF}"/>'
            f'<text x="20" y="30" style="{FONT};font-size:17px;font-weight:600" fill="{INK}">{title}</text>'
            f'<text x="20" y="50" style="{FONT};font-size:12.5px" fill="{INK2}">{sub}</text>{body}</svg>')


def axis(x0: float, x1: float, ticks: list[float], top: int, bottom: int, digits: int) -> tuple[str, callable]:
    def sx(v: float) -> float:
        return LEFT + (v - x0) / (x1 - x0) * (W - LEFT - RIGHT)
    s = ""
    for t in ticks:
        s += f'<line x1="{sx(t):.1f}" y1="{top}" x2="{sx(t):.1f}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>'
        s += f'<text x="{sx(t):.1f}" y="{bottom + 16}" text-anchor="middle" style="{FONT};font-size:11px" fill="{INK2}">{num(t, digits)}</text>'
    return s, sx


def legend(items: list[tuple[str, str]], y: int) -> str:
    s, x = "", 20
    for name, col in items:
        s += f'<rect x="{x}" y="{y - 9}" width="11" height="11" rx="2" fill="{col}"/><text x="{x + 17}" y="{y + 1}" style="{FONT};font-size:12px" fill="{INK2}">{name}</text>'
        x += 30 + len(name) * 6.6
    return s


def label(y: float, text: str, bold: bool = False) -> str:
    return f'<text x="{LEFT - 12}" y="{y + 4:.1f}" text-anchor="end" style="{FONT};font-size:12.5px{";font-weight:600" if bold else ""}" fill="{INK}">{text}</text>'


def bars(file: str, title: str, sub: str, rows: list[tuple[str, float]], x1: float, ticks: list[float], digits: int = 3,
         unit: str = "", accent: set[int] = frozenset(), ref: tuple[float, str] | None = None) -> None:
    """Одна серия, полосы от нуля."""
    top, bottom = 70, 70 + len(rows) * ROW
    s, sx = axis(0, x1, ticks, top, bottom, 2 if x1 <= 1 else 0)
    for i, (name, v) in enumerate(rows):
        y = top + i * ROW + ROW / 2
        s += label(y, name, i in accent)
        s += f'<rect x="{LEFT}" y="{y - 7:.1f}" width="{sx(v) - LEFT:.1f}" height="14" rx="4" fill="{C2 if i in accent else C1}"/>'
        s += f'<text x="{sx(v) + 6:.1f}" y="{y + 4:.1f}" style="{FONT};font-size:12px;font-weight:600" fill="{INK}">{num(v, digits)}{unit}</text>'
    if ref:
        s += f'<line x1="{sx(ref[0]):.1f}" y1="{top - 4}" x2="{sx(ref[0]):.1f}" y2="{bottom}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>'
        s += f'<text x="{sx(ref[0]) - 5:.1f}" y="{top - 8}" text-anchor="end" style="{FONT};font-size:11px" fill="{INK2}">{ref[1]}</text>'
    (OUT / file).write_text(frame(title, sub, bottom + 30, s), encoding="utf-8")


def dots(file: str, title: str, sub: str, series: list[str], rows: list[tuple[str, list[float | None]]], x0: float, x1: float,
         ticks: list[float], band: tuple[float, float, str] | None = None, ref: tuple[float, str] | None = None, accent: set[int] = frozenset()) -> None:
    """Точечный график с обрезанной осью (полосы с обрезанной осью искажали бы длину)."""
    cols = [C1, C2, C3]
    top = 96 if len(series) > 1 else 78
    bottom = top + len(rows) * ROW
    s, sx = axis(x0, x1, ticks, top, bottom, 2)
    if len(series) > 1:
        s += legend(list(zip(series, cols)), 72)
    if band:
        s += f'<rect x="{sx(band[0]):.1f}" y="{top}" width="{sx(band[1]) - sx(band[0]):.1f}" height="{bottom - top}" fill="{BAND}"/>'
        s += f'<text x="{sx(band[1]) + 5:.1f}" y="{top + 11}" style="{FONT};font-size:11px" fill="{INK2}">{band[2]}</text>'
    if ref:
        s += f'<line x1="{sx(ref[0]):.1f}" y1="{top}" x2="{sx(ref[0]):.1f}" y2="{bottom}" stroke="{INK2}" stroke-width="1.5" stroke-dasharray="4 3"/>'
        s += f'<text x="{sx(ref[0]) - 5:.1f}" y="{top + 11}" text-anchor="end" style="{FONT};font-size:11px" fill="{INK2}">{ref[1]}</text>'
    for i, (name, vals) in enumerate(rows):
        y = top + i * ROW + ROW / 2
        s += label(y, name, i in accent)
        s += f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        pts = [(v, cols[k]) for k, v in enumerate(vals) if v is not None]
        for v, col in pts:
            s += f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="{SURF}" stroke-width="2"/>'
        hi, lo = max(pts), min(pts)
        s += f'<text x="{sx(hi[0]) + 10:.1f}" y="{y + 4:.1f}" style="{FONT};font-size:12px;font-weight:600" fill="{INK}">{num(hi[0])}</text>'
        if len(pts) > 1 and hi[0] - lo[0] > 1e-9:
            s += f'<text x="{sx(lo[0]) - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" style="{FONT};font-size:12px" fill="{INK2}">{num(lo[0])}</text>'
    (OUT / file).write_text(frame(title, sub, bottom + 30, s), encoding="utf-8")


def stacked(file: str, title: str, sub: str, parts: list[str], rows: list[tuple[str, list[float]]]) -> None:
    cols = [C1, C3, C2]
    top, bottom = 96, 96 + len(rows) * ROW
    s, sx = axis(0, 100, [0, 25, 50, 75, 100], top, bottom, 0)
    s += legend(list(zip(parts, cols)), 72)
    for i, (name, vals) in enumerate(rows):
        y, x = top + i * ROW + ROW / 2, 0.0
        s += label(y, name)
        for v, col in zip(vals, cols):
            if v > 0:
                w = sx(x + v) - sx(x)
                s += f'<rect x="{sx(x) + 1:.1f}" y="{y - 8:.1f}" width="{max(w - 2, 1):.1f}" height="16" rx="3" fill="{col}"/>'
                if w > 34:
                    s += f'<text x="{sx(x) + w / 2:.1f}" y="{y + 4:.1f}" text-anchor="middle" style="{FONT};font-size:11.5px;font-weight:600" fill="#ffffff">{num(v, 1 if v % 1 else 0)} %</text>'
            x += v
    (OUT / file).write_text(frame(title, sub, bottom + 30, s), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dots("01_classification_progress.svg", "Классификация на публичном тесте: что дал каждый шаг",
         "Micro-F1 по категориям 102–106. Ось начинается с 0,78.", ["138 уникальных текстов", "все 208 строк"],
         [("Ключевые слова (D1)", [0.830, 0.826]), ("TF-IDF + логистическая регрессия (C1)", [0.802, 0.805]),
          ("LLM читает всё объявление (C16)", [0.842, 0.821]), ("LLM-судья по найденным строкам (C17)", [0.871, 0.856]),
          ("Судья + словарь, уточнённый по данным (C19)", [0.890, 0.866])],
         0.78, 0.94, [0.78, 0.82, 0.86, 0.90, 0.94], ref=(0.924, "потолок для всех строк: 0,924"), accent={4})
    bars("02_hard_examples.svg", "Трудные примеры: один судья против трёх экспертов",
         "Micro-F1 на 59 примерах с отрицаниями, чужими услугами, «только в пакете» (C23–C25, разведка).",
         [("Ключевые слова", 0.565), ("Строгий судья", 0.807), ("Мягкий судья", 0.845), ("Мягкий судья, роль «другое» снимает", 0.863),
          ("Три эксперта: мягкий + строгий + арбитр", 0.894)], 1.0, [0, 0.25, 0.5, 0.75, 1.0], accent={4})
    dots("03_ideas.svg", "Идеи, проверенные поверх судьи", "Micro-F1 на 138 уникальных текстах публичного теста (C18–C22). Ось начинается с 0,80.", ["результат"],
         [("Модель T-lite Q5 вместо Q4 (C18)", [0.850]), ("Модель Qwen3.5-9B (C18)", [0.856]), ("Три модели, решение большинством (C20)", [0.866]),
          ("Примеры из теста в подсказке (C21)", [0.853]), ("Решение по ближайшим объявлениям (C22)", [0.854]),
          ("Словарь, уточнённый по данным (C19)", [0.890])],
         0.80, 0.92, [0.80, 0.83, 0.86, 0.89, 0.92], band=(0.867, 0.871, "судья до изменений: 0,867–0,871"), accent={5})
    stacked("04_drafts.svg", "Черновики: что происходит с ответом модели", "Доля из 40 пар «объявление, категория» (G2, G3). Шаблон — запасной текст из фраз исходника.",
            ["прошёл проверки сразу", "исправлен перегенерацией", "заменён шаблоном"],
            [("Объявление + фрагменты, без перегенерации", [55, 0, 45]), ("Только объявление", [37.5, 20, 42.5]),
             ("Объявление + фрагменты", [55, 30, 15]), ("Только фрагменты и общие условия", [62.5, 22.5, 15])])
    bars("05_speed.svg", "Время на один вызов: ноутбук и настольная видеокарта", "Медиана, секунды. Одна и та же модель T-lite Q4 (C17, C18, G2, G3).",
         [("Судья, ноутбук (RTX 2080 Max-Q)", 15.6), ("Судья, второй компьютер", 2.0), ("Черновик с перегенерацией, ноутбук", 21.4),
          ("Черновик с перегенерацией, второй компьютер", 2.7)], 24, [0, 6, 12, 18, 24], digits=1, unit=" с", accent={1, 3})
    bars("06_label_noise.svg", "Почему на публичном тесте нельзя честно получить 0,95", "Micro-F1. Согласие — на почти одинаковых текстах: они размечены по-разному (C18, C22).",
         [("Согласие разметчиков между собой", 0.727), ("Наш судья, все 208 строк", 0.866),
          ("Наш судья, 138 уникальных текстов", 0.890), ("Потолок при одном ответе на текст", 0.924),
          ("Судья там, где разметчики нашли услугу", 0.929)], 1.0, [0, 0.25, 0.5, 0.75, 1.0], accent={3})
    print("графики:", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))
