#!/usr/bin/env python3
"""
Конвертер HTML-документации КОМПАС-3D в Markdown для RAG-индексации.

Основан на анализе ~350 уникальных сигнатур тегов, извлечённых
из блока контента (div#hmpagebody_scroller).

Использование:
    python convert_kompas_html_to_md.py /path/to/html_docs/ -o /path/to/md_output/
    python convert_kompas_html_to_md.py /path/to/html_docs/ -o /path/to/md_output/ --tree hmcontent.js
"""
import re
import json
from pathlib import Path
import os
import re
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

from bs4 import BeautifulSoup, Tag, NavigableString, Comment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def parse_toc_tree(toc_path: Path) -> dict:
    """
    Парсит hmcontent.js (дерево документации HelpMaker).
    Возвращает {filename: {title, breadcrumbs, level, topic_type, has_children}}.
    """
    try:
        content = toc_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        logger.error(f"Не удалось прочитать {toc_path}: {e}")
        return {}

    # Извлекаем JS-объект {...} из hmLoadTOC({...})
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    if first_brace == -1 or last_brace == -1:
        logger.error("Не найдены скобки {} в hmcontent.js")
        return {}

    js_object = content[first_brace:last_brace + 1]

    # Конвертируем JS → JSON
    # 1. Ключи без кавычек → с кавычками
    json_str = re.sub(r'^\s*(\w+)\s*:', r'  "\1":', js_object, flags=re.MULTILINE)
    # 2. Trailing commas
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON TOC: {e}")
        return {}

    # Обходим дерево
    result = {}
    _walk_toc_items(data.get('items', []), [], result)

    logger.info(f"Распарсено {len(result)} записей из дерева документации")
    return result


def _walk_toc_items(items: list, breadcrumb_parts: list[str], result: dict):
    """Рекурсивно обходит дерево TOC."""
    for item in items:
        if not isinstance(item, dict):
            continue

        title = item.get('cp', '').strip()
        filename = item.get('hf', '').strip()
        level = item.get('lv', 0)
        topic_type = item.get('tp', '')
        children = item.get('items', [])

        current_bc = breadcrumb_parts + [title] if title else breadcrumb_parts

        if filename:
            result[filename] = {
                'title': title,
                'breadcrumbs': ' → '.join(current_bc),
                'level': level,
                'topic_type': topic_type,
                'has_children': len(children) > 0,
            }

        if children:
            _walk_toc_items(children, current_bc, result)

def _fix_bold_spacing(self, text: str) -> str:
    """
    Исправляет слипшиеся слова вокруг **bold** маркеров.
    
    **Текущим**является → **Текущим** является
    команду**Сделать   → команду **Сделать
    В**КОМПАС-3D**можно → В **КОМПАС-3D** можно
    """
    # Буквы и цифры (кириллица + латиница)
    LETTER = r'[а-яА-ЯёЁa-zA-Z0-9]'
    
    # Шаг 1: Находим все **...** пары и нормализуем пробелы вокруг
    def add_spaces_around_bold(match):
        before = match.group(1)  # символ перед **
        content = match.group(2)  # содержимое между ** и **
        after = match.group(3)    # символ после **
        
        result = before
        # Пробел перед открывающим **
        if re.match(LETTER, before):
            result += ' '
        result += f'**{content}**'
        # Пробел после закрывающего **
        if re.match(LETTER, after):
            result += ' '
        result += after
        
        return result
    
    # Паттерн: (символ_перед)**контент**(символ_после)
    text = re.sub(
        r'(.)' + r'\*\*' + r'([^*]+?)' + r'\*\*' + r'(.)',
        add_spaces_around_bold,
        text
    )
    
    return text

def _js_object_to_json(js_text: str) -> str:
    """
    Конвертирует JS-объект в валидный JSON.
    Обрабатывает: ключи без кавычек, trailing commas.
    """
    # 1. Ключи без кавычек → с кавычками
    #    В начале строки (после пробелов): слово перед двоеточием
    result = re.sub(r'^\s*(\w+)\s*:', r'  "\1":', js_text, flags=re.MULTILINE)
    
    # 2. Trailing commas: ,} → } и ,] → ]
    result = re.sub(r',(\s*[}\]])', r'\1', result)
    
    return result


# ============================================================================
# КОНФИГУРАЦИЯ МАППИНГА
# ============================================================================

@dataclass
class ConversionConfig:
    """Конфигурация конвертации."""
    # Включать ли метаданные (breadcrumbs, title) в начало документа
    include_metadata: bool = True
    # Включать ли "Смотрите также" секцию
    include_see_also: bool = True
    # Включать ли локальное оглавление (p_Z_LOC_TOC)
    include_local_toc: bool = False
    # Включать ли футер
    include_footer: bool = False
    # Включать ли содержимое dropdown-toggle (способы вызова команды)
    include_dropdowns: bool = True
    # Максимальная глубина вложенности списков
    max_list_depth: int = 4
    # Включать ли изображения
    include_images: bool = True
    # Базовый путь для изображений (относительный)
    image_base_path: str = "images/"
    # Включать ли подписи к изображениям
    include_image_captions: bool = True
    # Обрабатывать ли таблицы как markdown-таблицы или блоки текста
    tables_as_markdown: bool = True
    # Включать ли блоки "в разработке" (underway)
    include_underway: bool = False


# --- Классы, которые нужно ПОЛНОСТЬЮ ИГНОРИРОВАТЬ (удалить вместе с содержимым) ---
CLASSES_REMOVE_WITH_CONTENT = {
    'topicfooter',      # Футер "© ООО АСКОН..."
}

# --- Классы-обёртки, которые нужно РАЗВЕРНУТЬ (убрать тег, оставить содержимое) ---
CLASSES_UNWRAP = {
    'f_bodytext',       # Просто текст — обёртка не нужна
    'f_Table_Cell',     # Текст в ячейке таблицы
    'f_Table_List',     # Текст элемента списка в таблице
    'f_Normal',         # Обычный текст
    'f_Table_Para_NWA', # Текст абзаца в таблице
    'f_Table_Para_0',   # Текст абзаца в таблице
    'f_Table_Section',  # Текст секции таблицы
    'f_Image_Left',     # Контейнер картинки слева
    'f_Image_Center',   # Контейнер картинки по центру
}

# --- Маппинг CSS-классов параграфов на markdown-элементы ---
# Формат: class -> (prefix, suffix, is_block)

PARAGRAPH_CLASS_MAP = {
    # ...всё как раньше, но:

    # Заголовки — НЕ добавляем ** тут, потому что span внутри уже добавит
    'p_Head4':              ('#### ', '', True),
    'p_Section_Title':      ('#### ', '', True),

    # Procedure — просто текст, форматирование идёт от вложенного span


    # Table_Title — аналогично
    

    # ... остальное без изменений
}

PARAGRAPH_CLASS_MAP = {
    # Основной текст
    'p_bodytext':               ('', '', True),
    'p_noNum':                  ('', '', True),
    'p_noNum2':                 ('', '', True),
    'p_Normal':                 ('', '', True),

    # Заголовки и структурные элементы
    'p_Head4':              ('#### ', '', True),
    'p_Section_Title':      ('#### ', '', True),
    'p_Procedure':          ('', '', True),
    'p_Table_Title':        ('', '', True),

    # Списки — маркированные (разные уровни вложенности)
    'p_bulleted':               ('- ', '', True),
    'p_bulleted2':              ('  - ', '', True),
    'p_bulleted3':              ('    - ', '', True),

    # Списки — нумерованные
    'p_Numbered':               ('1. ', '', True),
    'p_Numbered2':              ('   1. ', '', True),

    # Таблицы
    'p_Table_Cell':             ('', '', True),
    'p_Table_Cell_Center':      ('', '', True),
    'p_Table_Col_Name':         ('', '', True),
    'p_Table_Col_Name_Center':  ('', '', True),
    'p_Table_List':             ('- ', '', True),
    'p_Table_Para_NWA':         ('', '', True),
    'p_Table_Para_0':           ('', '', True),
    'p_Table_Section':          ('', '', True),

    # Изображения и подписи
    'p_Image_Center':           ('', '', True),
    'p_Image_Left':             ('', '', True),
    'p_Picture_Title':          ('*', '*', True),
    'p_ImageCaption':           ('*', '*', True),

    # Специальные блоки
    'p_SeeAlso':                ('---\n\n', '', True),
    'p_Z_LOC_TOC':              ('', '', True),     # Локальное оглавление
    'p_underway':               ('> ⚠️ ', '', True),
    'p_Z_New':                  ('', '', True),

    # Стартовая страница
    'p_StartPage_Link':         ('- ', '', True),
    'p_StartPage_Sect':         ('### ', '', True),

    # Термины (глоссарий)
    'p_Term_Name':              ('**', '**', True),
    'p_Term_Litera':            ('## ', '', True),
}

# --- Маппинг inline-стилей на markdown-форматирование ---
# Ключ: frozenset CSS-свойств. Значение: (prefix, suffix)
INLINE_STYLE_MAP = {
    'font-weight':      ('**', '**'),
    'font-style':       ('*', '*'),
    'text-decoration':  ('', ''),   # подчёркивание — в md нет стандартного способа
}


class HTMLToMarkdownConverter:
    """Конвертер HTML документации КОМПАС-3D в Markdown."""

    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self._list_counters: list[int] = []
        self._in_table = False
        self._current_table_data: list[list[str]] = []
        self._current_row: list[str] = []

    def convert_file(self, html_path: Path) -> Optional[str]:
        """
        Конвертирует HTML-файл в Markdown.

        Returns:
            Строка с markdown-содержимым или None если файл не содержит контента.
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except (UnicodeDecodeError, OSError) as e:
            logger.warning(f"Не удалось прочитать {html_path}: {e}")
            return None

        soup = BeautifulSoup(html_content, 'html.parser')
        return self.convert_soup(soup, source_path=html_path)

    def convert_soup(self, soup: BeautifulSoup,
                     source_path: Optional[Path] = None) -> Optional[str]:
        """Конвертирует BeautifulSoup объект в Markdown."""
        content_div = soup.find('div', id='hmpagebody_scroller')
        if content_div is None:
            logger.debug(f"Блок контента не найден: {source_path}")
            return None

        parts = []

        # --- Метаданные ---
        if self.config.include_metadata:
            metadata = self._extract_metadata(soup)
            if metadata:
                parts.append(metadata)

        # --- Основной контент ---
        content_md = self._process_content_block(content_div)
        if content_md:
            parts.append(content_md)

        if not parts:
            return None

        result = '\n\n'.join(parts)

        # Постобработка
        result = self._postprocess(result)

        return result

    # ========================================================================
    # ИЗВЛЕЧЕНИЕ МЕТАДАННЫХ
    # ========================================================================

    def _extract_metadata(self, soup: BeautifulSoup) -> str:
        """Извлекает метаданные документа (заголовок, breadcrumbs)."""
        parts = []

        # Заголовок страницы
        title_el = soup.find('p', class_='topictitle')
        if title_el:
            title_text = self._get_clean_text(title_el)
            if title_text:
                parts.append(f'# {title_text}')

        # Breadcrumbs (путь в дереве)
        breadcrumbs_el = soup.find('p', id='ptopic_breadcrumbs')
        if breadcrumbs_el:
            bc_links = breadcrumbs_el.find_all('a', href=True)
            if bc_links:
                bc_path = ' → '.join(
                    self._get_clean_text(a) for a in bc_links
                    if self._get_clean_text(a)
                )
                if bc_path:
                    parts.append(f'> **Путь:** {bc_path}')

        return '\n\n'.join(parts) if parts else ''

    # ========================================================================
    # ОБРАБОТКА БЛОКА КОНТЕНТА
    # ========================================================================

    def _process_content_block(self, content_div: Tag) -> str:
        """Обрабатывает основной блок контента."""
        blocks = []

        for child in content_div.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    blocks.append(text)
                continue
            if not isinstance(child, Tag):
                continue

            block_md = self._process_block_element(child)
            if block_md is not None:
                blocks.append(block_md)

        return '\n\n'.join(b for b in blocks if b.strip())

    def _process_block_element(self, element: Tag) -> Optional[str]:
        """
        Обрабатывает блочный элемент верхнего уровня.
        Возвращает markdown-строку или None для удаления.
        """
        classes = set(element.get('class', []))
        tag = element.name

        # --- Элементы для полного удаления ---
        if classes & CLASSES_REMOVE_WITH_CONTENT:
            if not self.config.include_footer and 'topicfooter' in classes:
                return None

        # Футер
        if 'topicfooter' in classes:
            if self.config.include_footer:
                return f"---\n\n{self._get_clean_text(element)}"
            return None

        # --- Таблицы ---
        if tag == 'table':
            return self._process_table(element)

        # --- div-обёртки ---
        if tag == 'div':
            return self._process_div(element)

        # --- Параграфы ---
        if tag == 'p':
            return self._process_paragraph(element)

        # --- Изображения на верхнем уровне ---
        if tag == 'img':
            return self._process_image(element)

        # --- Разделители ---
        if tag == 'hr':
            return '---'

        # --- Ссылки-якоря на верхнем уровне ---
        if tag == 'a' and 'hmanchor' in classes:
            return None  # Якоря удаляем

        # --- Прочие блочные элементы ---
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            level = int(tag[1])
            text = self._process_inline_content(element)
            if text:
                return f'{"#" * level} {text}'

        # --- Неопознанный элемент — пытаемся обработать содержимое ---
        logger.debug(f"Неопознанный блочный элемент: <{tag}> classes={classes}")
        return self._process_inline_content(element)

    # ========================================================================
    # ОБРАБОТКА ПАРАГРАФОВ
    # ========================================================================

       
    def _process_paragraph(self, p_tag: Tag) -> Optional[str]:
        """Обрабатывает один <p> элемент."""
        classes = set(p_tag.get('class', []))

        # Определяем тип параграфа
        para_class = None
        for cls in classes:
            if cls in PARAGRAPH_CLASS_MAP:
                para_class = cls
                break

        if para_class:
            prefix, suffix, skip_marker = PARAGRAPH_CLASS_MAP[para_class]
        else:
            prefix, suffix, skip_marker = ('', '', False)

        # Получаем inline-содержимое (с сохранёнными пробелами)
        content = self._process_inline_content(p_tag, skip_list_marker=skip_marker)

        # Теперь strip() — на уровне параграфа это безопасно
        content = content.strip()

        if not content:
            return None

        return f'{prefix}{content}{suffix}'

    def _is_anchor_only_paragraph(self, p_tag: Tag) -> bool:
        """Проверяет, содержит ли параграф только якоря."""
        for child in p_tag.children:
            if isinstance(child, NavigableString):
                if str(child).strip():
                    return False
            elif isinstance(child, Tag):
                if 'hmanchor' not in child.get('class', []):
                    return False
        return True

    def _process_local_toc_entry(self, p_tag: Tag) -> str:
        """Обрабатывает запись локального оглавления."""
        link = p_tag.find('a', class_='topiclink')
        if link:
            text = self._get_clean_text(link)
            href = link.get('href', '')
            return f'- [{text}](#{href})'
        return f'- {self._get_clean_text(p_tag)}'

    def _process_image_paragraph(self, p_tag: Tag) -> Optional[str]:
        """Обрабатывает параграф с изображением."""
        if not self.config.include_images:
            return None

        img = p_tag.find('img')
        if img:
            return self._process_image(img)
        return None

    # ========================================================================
    # ОБРАБОТКА INLINE-СОДЕРЖИМОГО
    # ========================================================================
    
    def _process_inline_content(self, element: Tag,
                                skip_list_marker: bool = False) -> str:
        """
        Рекурсивно обрабатывает inline-содержимое элемента.
        
        ВАЖНО: пробелы внутри span-ов значимы!
        HTML: <span style="font-weight: bold;">Текущим</span><span> является слой</span>
        Пробел перед "является" — внутри второго span, нельзя обрезать.
        """
        parts: list[str] = []
        first_child = True

        for child in element.children:
            if isinstance(child, Comment):
                continue

            if isinstance(child, NavigableString):
                text = str(child)
                # &nbsp; → пробел
                text = text.replace('\xa0', ' ')
                # Схлопываем множественные пробелы/переносы внутри строки
                # НО сохраняем ведущие и завершающие пробелы — они значимы!
                text = re.sub(r'[ \t]+', ' ', text)
                text = text.replace('\n', ' ')
                # НЕ делаем strip() — пробел в начале/конце значим
                if text:
                    parts.append(text)
                continue

            if not isinstance(child, Tag):
                continue

            tag = child.name
            classes = set(child.get('class', []))
            style = child.get('style', '')

            # --- Якоря (hmanchor) — пропускаем ---
            if 'hmanchor' in classes:
                continue

            # --- Маркеры списков (•, 1.) — пропускаем ---
            if skip_list_marker and first_child and self._is_list_marker(child):
                first_child = False
                continue

            # --- <br> → перенос ---
            if tag == 'br':
                parts.append('  \n')
                first_child = False
                continue

            # --- <img> ---
            if tag == 'img':
                img_md = self._process_image(child)
                if img_md:
                    parts.append(img_md)
                first_child = False
                continue

            # --- <a> ссылки ---
            if tag == 'a':
                link_md = self._process_link(child)
                if link_md:
                    parts.append(link_md)
                first_child = False
                continue

            # --- Маркеры списков через стиль ---
            if self._is_list_marker(child):
                if skip_list_marker:
                    first_child = False
                    continue

            # --- Inline-форматирование через стили ---
            styled_text = self._apply_inline_styles(child, style, classes)
            if styled_text is not None:
                parts.append(styled_text)
                first_child = False
                continue

            # --- Прозрачные контейнеры (unwrap) ---
            if classes & CLASSES_UNWRAP:
                inner = self._process_inline_content(child)
                if inner:
                    parts.append(inner)
                first_child = False
                continue

            # --- Специальные span-классы ---
            span_result = self._process_special_span(child, classes, style)
            if span_result is not None:
                parts.append(span_result)
                first_child = False
                continue

            # --- Рекурсия для прочих элементов ---
            inner = self._process_inline_content(child)
            if inner:
                parts.append(inner)

            first_child = False

        # Собираем результат — НЕ strip(), пробелы значимы на этом уровне
        return ''.join(parts)

    def _is_list_marker(self, element: Tag) -> bool:
        """Проверяет, является ли элемент маркером списка (•, 1., и т.д.)."""
        classes = set(element.get('class', []))
        style = element.get('style', '')

        # Спаны с display:inline-block и margin-left — это маркеры
        if 'display' in style and 'margin-left' in style:
            text = self._get_clean_text(element).strip()
            # Маркер: •, 1., 2., и т.д.
            if text in ('•', '–', '—') or re.match(r'^\d+\.$', text):
                return True

        return False

    def _apply_inline_styles(self, element: Tag, style: str,
                            classes: set) -> Optional[str]:
        """Применяет inline-стили: bold, italic, underline и т.д."""
        if not style:
            return None

        # Рекурсивно получаем содержимое (пробелы сохранены!)
        inner = self._process_inline_content(element)
        if not inner:
            return None

        # Определяем маркеры форматирования
        is_bold = 'font-weight: bold' in style or 'font-weight:bold' in style
        is_italic = 'font-style: italic' in style or 'font-style:italic' in style

        if is_bold and is_italic:
            prefix, suffix = '***', '***'
        elif is_bold:
            prefix, suffix = '**', '**'
        elif is_italic:
            prefix, suffix = '*', '*'
        else:
            # Нет форматирования — возвращаем как есть
            return inner

        # КЛЮЧЕВОЕ: сохраняем пробелы СНАРУЖИ маркеров
        # " Текущим " + bold → " **Текущим** "
        # Извлекаем ведущие и завершающие пробелы
        stripped = inner.strip()
        if not stripped:
            return inner  # только пробелы — возвращаем как есть

        leading = inner[:len(inner) - len(inner.lstrip())]
        trailing = inner[len(inner.rstrip()):]

        return f'{leading}{prefix}{stripped}{suffix}{trailing}'

    def _process_special_span(self, element: Tag, classes: set,
                            style: str) -> Optional[str]:
        inner = self._process_inline_content(element)

        if 'f_Head4' in classes:
            # НЕ оборачиваем — paragraph уже даст ####
            return inner

        if 'f_Procedure' in classes:
            # Оборачиваем в bold — paragraph НЕ оборачивает
            return f'**{inner}**' if inner else None

        if 'f_Section_Title' in classes:
            # НЕ оборачиваем — paragraph уже даст ####
            return inner

        if 'f_Table_Title' in classes:
            return f'**{inner}**' if inner else None

        # Суперскрипт
        if 'f_Z_SUPER' in classes:
            return f'^{inner}^' if inner else None

        # Субскрипт
        if 'f_Z_SUB' in classes:
            return f'~{inner}~' if inner else None

        # Заголовок столбца таблицы
        if 'f_Table_Col_Name' in classes or 'f_Table_Col_Name_Center' in classes:
            return f'**{inner}**' if inner else None

        # "Смотрите также"
        if 'f_SeeAlso' in classes:
            return f'**{inner}**' if inner else None

        # Подпись к картинке
        if 'f_Picture_Title' in classes:
            return f'*{inner}*' if inner else None

        # ImageCaption
        if 'f_ImageCaption' in classes:
            return f'*{inner}*' if inner else None

        # Запись стартовой страницы
        if 'f_StartPage_Link' in classes:
            return inner

        if 'f_StartPage_Sect' in classes:
            return f'**{inner}**' if inner else None

        # Локальное оглавление
        if 'f_Z_LOC_TOC' in classes:
            return inner

        # "В разработке"
        if 'f_underway' in classes:
            return inner

        # "Новое"
        if 'f_Z_New' in classes:
            return inner

        # Термин
        if 'f_Term_Name' in classes:
            return f'**{inner}**' if inner else None

        if 'f_Term_Litera' in classes:
            return inner

        # Элементы с font-weight в стиле
        if style:
            styled = self._apply_inline_styles(element, style, classes)
            if styled is not None:
                return styled

        # Обёртки
        if classes & CLASSES_UNWRAP:
            return inner

        # Прочие span с классами f_* — просто текст
        for cls in classes:
            if cls.startswith('f_'):
                return inner

        return None

    # ========================================================================
    # ОБРАБОТКА ССЫЛОК
    # ========================================================================

    def _process_link(self, a_tag: Tag) -> Optional[str]:
        """Обрабатывает элемент <a>."""
        classes = set(a_tag.get('class', []))

        # Якоря — удаляем
        if 'hmanchor' in classes:
            return None

        # Ссылки на всплывающие окна (dropdown-toggle)
        if 'dropdown-toggle' in classes:
            return self._process_dropdown_toggle(a_tag)

        # Ссылки на изображения (imagetogglelink)
        if 'imagetogglelink' in classes:
            return self._process_image_toggle(a_tag)

        href = a_tag.get('href', '')
        text = self._process_inline_content(a_tag)

        if not text:
            return None

        # Внешние ссылки
        if 'weblink' in classes:
            target = a_tag.get('target', '')
            return f'[{text}]({href})'

        # Внутренние ссылки (topiclink)
        if 'topiclink' in classes or href:
            # Конвертируем .html в .md для внутренних ссылок
            if href and not href.startswith(('http://', 'https://', '#', 'mailto:')):
                href = re.sub(r'\.html?$', '.md', href)
            return f'[{text}]({href})' if href else text

        # Ссылки из локального оглавления
        if 'hmlinklistitem' in classes:
            return f'[{text}]({href})' if href else text

        return text

    def _process_dropdown_toggle(self, a_tag: Tag) -> Optional[str]:
        """Обрабатывает dropdown-toggle (раскрывающийся блок)."""
        text = self._process_inline_content(a_tag)

        if not self.config.include_dropdowns:
            return f'**{text}**' if text else None

        # Находим связанный dropdown-toggle-body
        # Он обычно идёт рядом в DOM
        return f'**{text}**' if text else None

    def _process_image_toggle(self, a_tag: Tag) -> Optional[str]:
        """Обрабатывает переключатель изображений."""
        img = a_tag.find('img', class_='image-toggle')
        if img:
            return self._process_image(img)
        return None

    # ========================================================================
    # ОБРАБОТКА ИЗОБРАЖЕНИЙ
    # ========================================================================

    def _process_image(self, img_tag: Tag) -> Optional[str]:
        """Обрабатывает элемент <img>."""
        if not self.config.include_images:
            return None

        classes = set(img_tag.get('class', []))

        # dropdown-toggle-icon — служебная иконка, пропускаем
        if 'dropdown-toggle-icon' in classes:
            return None

        src = img_tag.get('src', '')
        alt = img_tag.get('alt', '')
        title = img_tag.get('title', '')

        if not src:
            # Проверяем data-src0 для image-toggle
            src = img_tag.get('data-src0', '')

        if not src:
            return None

        # Определяем тип изображения
        # Маленькие иконки (кнопки, пиктограммы) — инлайн
        try:
            style = img_tag.get('style', '')
            width_match = re.search(r'width:\s*([\d.]+)rem', style)
            height_match = re.search(r'height:\s*([\d.]+)rem', style)

            if width_match and height_match:
                width = float(width_match.group(1))
                height = float(height_match.group(1))

                # Маленькие иконки (< 2rem) — вставляем как inline
                if width <= 2.0 and height <= 2.0:
                    return f'![{alt}]({src})'
        except (ValueError, AttributeError):
            pass

        # Полноразмерное изображение
        caption = title or alt or ''
        if caption:
            return f'![{caption}]({src})'
        return f'![]({src})'

    # ========================================================================
    # ОБРАБОТКА ТАБЛИЦ
    # ========================================================================


    def _is_image_wrapper_table(self, table_tag: Tag) -> bool:
        """
        Детектит таблицу, используемую как обёртка для картинки + подписи.
        Признаки: 1-2 строки, 1 столбец, содержит только img и/или подпись.
        """
        rows = table_tag.find_all('tr', recursive=True)
        if len(rows) > 3:
            return False

        for row in rows:
            cells = row.find_all(['td', 'th'], recursive=False)
            if len(cells) != 1:
                return False

        # Проверяем что содержимое — только картинки и/или подписи
        all_content = table_tag.get_text(strip=True)
        has_img = table_tag.find('img') is not None

        # Если есть картинка и мало текста — это обёртка
        if has_img and len(all_content) < 200:
            return True

        return False
    

    def _unwrap_image_table(self, table_tag: Tag) -> Optional[str]:
        """Извлекает содержимое таблицы-обёртки для картинки."""
        parts = []

        for row in table_tag.find_all('tr', recursive=True):
            for cell in row.find_all(['td', 'th'], recursive=False):
                # Ищем картинки
                for img in cell.find_all('img'):
                    img_md = self._process_image(img)
                    if img_md:
                        parts.append(img_md)

                # Ищем текст подписи (если не только картинка)
                text = self._get_clean_text(cell)
                # Убираем текст который уже в alt картинки
                if text and not cell.find('img'):
                    parts.append(f'*{text}*')

        return '\n\n'.join(parts) if parts else None

    def _process_table(self, table_tag: Tag) -> Optional[str]:
        note_result = self._check_note_table(table_tag)
        if note_result is not None:
            return note_result

        # Проверяем, является ли таблица контейнером для dropdown-toggle
        if self._is_dropdown_table(table_tag):
            return self._process_dropdown_table(table_tag)

        # *** НОВАЯ ПРОВЕРКА: таблица-обёртка для картинки ***
        if self._is_image_wrapper_table(table_tag):
            return self._unwrap_image_table(table_tag)

        # Обычная таблица данных
        if self.config.tables_as_markdown:
            return self._table_to_markdown(table_tag)
        else:
            return self._table_to_text(table_tag)

    def _check_note_table(self, table_tag: Tag) -> Optional[str]:
        """Проверяет, является ли таблица блоком примечания/предупреждения."""
        # Ищем характерную структуру: иконка (note.png, warning.png и т.д.)
        # в первой ячейке, текст во второй
        rows = table_tag.find_all('tr', recursive=False)
        if not rows:
            rows = []
            tbody = table_tag.find('tbody')
            if tbody:
                rows = tbody.find_all('tr', recursive=False)

        if len(rows) != 1:
            return None

        cells = rows[0].find_all('td', recursive=False)
        if len(cells) != 2:
            return None

        # Первая ячейка — иконка?
        first_cell = cells[0]
        img = first_cell.find('img')
        if not img:
            return None

        src = img.get('src', '').lower()
        note_types = {
            'note': '📝',
            'warning': '⚠️',
            'caution': '⛔',
            'tip': '💡',
            'info': 'ℹ️',
            'important': '❗',
        }

        note_icon = None
        for note_type, icon in note_types.items():
            if note_type in src:
                note_icon = icon
                break

        if note_icon is None:
            return None

        # Вторая ячейка — текст
        text_content = self._process_cell_content(cells[1])
        if text_content:
            # Форматируем как blockquote
            lines = text_content.split('\n')
            quoted = '\n'.join(f'> {line}' for line in lines)
            return f'> {note_icon} **Примечание**\n{quoted}'

        return None

    def _is_dropdown_table(self, table_tag: Tag) -> bool:
        """Проверяет, содержит ли таблица dropdown-toggle."""
        return table_tag.find('a', class_='dropdown-toggle') is not None

    def _process_dropdown_table(self, table_tag: Tag) -> Optional[str]:
        """Обрабатывает таблицу с dropdown-toggle (способы вызова)."""
        if not self.config.include_dropdowns:
            # Находим заголовок
            title_span = table_tag.find('span', class_='f_Section_Title')
            if title_span:
                return f'**{self._get_clean_text(title_span)}**'
            return None

        parts = []

        # Заголовок
        toggle_link = table_tag.find('a', class_='dropdown-toggle')
        if toggle_link:
            title = self._process_inline_content(toggle_link)
            if title:
                parts.append(f'**{title}**')

        # Содержимое dropdown-body
        body_div = table_tag.find('div', class_='dropdown-toggle-body')
        if body_div:
            body_content = self._process_content_block(body_div)
            if body_content:
                parts.append(body_content)

        return '\n\n'.join(parts) if parts else None

    def _table_to_markdown(self, table_tag: Tag) -> Optional[str]:
        """Конвертирует HTML-таблицу в Markdown-таблицу."""
        rows_data = []

        # Собираем все строки
        rows = table_tag.find_all('tr', recursive=False)
        if not rows:
            tbody = table_tag.find('tbody')
            if tbody:
                rows = tbody.find_all('tr', recursive=False)
            thead = table_tag.find('thead')
            if thead:
                header_rows = thead.find_all('tr', recursive=False)
                rows = header_rows + rows

        if not rows:
            return None

        max_cols = 0

        for row in rows:
            cells = row.find_all(['td', 'th'], recursive=False)
            row_data = []

            for cell in cells:
                content = self._process_cell_content(cell)
                # Обработка colspan
                colspan = int(cell.get('colspan', 1))
                row_data.append(content)
                for _ in range(colspan - 1):
                    row_data.append('')

            rows_data.append(row_data)
            max_cols = max(max_cols, len(row_data))

        if not rows_data or max_cols == 0:
            return None

        # Нормализуем количество столбцов
        for row in rows_data:
            while len(row) < max_cols:
                row.append('')

        # Очищаем содержимое ячеек от переводов строк (для markdown-таблицы)
        for i, row in enumerate(rows_data):
            for j, cell in enumerate(row):
                # Заменяем переводы строк на <br> для совместимости
                rows_data[i][j] = cell.replace('\n', ' ').strip()

        # Формируем markdown-таблицу
        lines = []

        # Заголовок (первая строка)
        header = rows_data[0]
        lines.append('| ' + ' | '.join(
            self._escape_table_cell(c) for c in header
        ) + ' |')

        # Разделитель
        lines.append('| ' + ' | '.join(
            '---' for _ in range(max_cols)
        ) + ' |')

        # Данные
        for row in rows_data[1:]:
            lines.append('| ' + ' | '.join(
                self._escape_table_cell(c) for c in row
            ) + ' |')

        return '\n'.join(lines)

    def _table_to_text(self, table_tag: Tag) -> Optional[str]:
        """Конвертирует таблицу в текстовый формат (для сложных таблиц)."""
        parts = []
        rows = table_tag.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            cell_texts = []
            for cell in cells:
                content = self._process_cell_content(cell)
                if content:
                    cell_texts.append(content)

            if cell_texts:
                parts.append(' | '.join(cell_texts))

        return '\n'.join(parts) if parts else None

    def _process_cell_content(self, cell: Tag) -> str:
        """Обрабатывает содержимое ячейки таблицы."""
        # Ячейка может содержать параграфы, списки, изображения
        blocks = []

        for child in cell.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    blocks.append(text)
                continue

            if isinstance(child, Comment):
                continue

            if not isinstance(child, Tag):
                continue

            if child.name == 'p':
                p_result = self._process_paragraph(child)
                if p_result:
                    blocks.append(p_result)
            elif child.name == 'div':
                div_result = self._process_div(child)
                if div_result:
                    blocks.append(div_result)
            elif child.name == 'table':
                # Вложенная таблица — рекурсия
                table_result = self._process_table(child)
                if table_result:
                    blocks.append(table_result)
            elif child.name == 'img':
                img_result = self._process_image(child)
                if img_result:
                    blocks.append(img_result)
            elif child.name == 'br':
                blocks.append('')
            else:
                text = self._process_inline_content(child)
                if text:
                    blocks.append(text)

        return '\n'.join(blocks)

    def _escape_table_cell(self, text: str) -> str:
        """Экранирует содержимое ячейки таблицы для markdown."""
        # Склеиваем смежные bold: **a****b** → **a b**
        for _ in range(3):
            text = re.sub(r'\*\*([^*]+)\*\*\s*\*\*([^*]+)\*\*', r'**\1 \2**', text)
        # Схлопываем оставшиеся 4+ звёздочки
        text = re.sub(r'\*{4,}', '**', text)
        # Пустой bold
        text = re.sub(r'\*{2,}\s*\*{2,}', '', text)
        # Заменяем | на \|
        text = text.replace('|', '\\|')
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        return text if text else ' '

    # ========================================================================
    # ОБРАБОТКА DIV
    # ========================================================================

    def _process_div(self, div_tag: Tag) -> Optional[str]:
        """Обрабатывает элемент <div>."""
        classes = set(div_tag.get('class', []))

        # Футер
        if 'topicfooter' in classes:
            if self.config.include_footer:
                return f"---\n\n{self._get_clean_text(div_tag)}"
            return None

        # Dropdown body
        if 'dropdown-toggle-body' in classes:
            if self.config.include_dropdowns:
                return self._process_content_block(div_tag)
            return None

        # div-обёртки для подписей к картинкам
        if 'p_Picture_Title' in classes:
            inner_p = div_tag.find('p')
            if inner_p:
                return self._process_paragraph(inner_p)
            # Может содержать span напрямую
            text = self._process_inline_content(div_tag)
            if text:
                return f'*{text}*'
            return None

        if 'p_Table_Para_NWA' in classes:
            return self._process_content_block(div_tag)

        # Обычные div-обёртки — обрабатываем содержимое
        result = self._process_content_block(div_tag)
        return result if result else None

    # ========================================================================
    # УТИЛИТЫ
    # ========================================================================

    def _get_clean_text(self, element: Tag) -> str:
        """Получает чистый текст элемента без тегов."""
        text = element.get_text()
        text = text.replace('\xa0', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _postprocess(self, md_text: str) -> str:
        """Финальная очистка markdown."""
        # Множественные пустые строки
        md_text = re.sub(r'\n{3,}', '\n\n', md_text)

        # ===== BOLD/ITALIC НОРМАЛИЗАЦИЯ =====
        
        # 1. Склеиваем смежные bold: **text1****text2** → **text1 text2**
        #    и **text1** **text2** → **text1 text2**
        for _ in range(3):  # несколько проходов для вложенных случаев
            md_text = re.sub(r'\*\*([^*]+)\*\*\s*\*\*([^*]+)\*\*', r'**\1 \2**', md_text)
        
        # 2. Пустое форматирование ** ** или **** → убираем
        md_text = re.sub(r'\*{2,}\s*\*{2,}', '', md_text)
        
        # 3. Схлопываем 4+ звёздочки: ****text**** → **text**
        md_text = re.sub(r'\*{4,}', '**', md_text)
        
        # 4. Тройные → двойные: ***text*** → **text**
        md_text = re.sub(r'\*{3}([^*]+)\*{3}', r'**\1**', md_text)
        
        # 5. Пробел перед закрывающим **: "текст **" → "текст**"
        md_text = re.sub(r'(\S)\s+\*\*', r'\1**', md_text)
        
        # 6. Пробел после открывающего **: "** текст" → "**текст"
        md_text = re.sub(r'\*\*\s+(\S)', r'**\1', md_text)
        
        # 7. Восстанавливаем пробел ПЕРЕД открывающим **
        md_text = re.sub(
            r'([а-яА-ЯёЁa-zA-Z0-9,.])\*\*([а-яА-ЯёЁa-zA-Z0-9])',
            r'\1 **\2',
            md_text
        )
        
        # 8. Восстанавливаем пробел ПОСЛЕ закрывающего **
        md_text = re.sub(
            r'([а-яА-ЯёЁa-zA-Z0-9])\*\*([а-яА-ЯёЁa-zA-Z(])',
            r'\1** \2',
            md_text
        )

        # ===== ПРОБЕЛЫ =====
        
        # Лишний пробел перед пунктуацией
        md_text = re.sub(r' +([.,:;!?)])', r'\1', md_text)
        
        # Множественные пробелы → один (кроме начала строки)
        md_text = re.sub(r'(?<=\S) {2,}(?=\S)', ' ', md_text)
        
        # Пробелы в конце строк
        md_text = re.sub(r'[ \t]+$', '', md_text, flags=re.MULTILINE)

        return md_text.strip()

# ============================================================================
# BATCH-КОНВЕРТАЦИЯ
# ============================================================================

def batch_convert(
    input_dir: Path,
    output_dir: Path,
    config: Optional[ConversionConfig] = None,
    toc_tree: Optional[dict] = None,
    max_files: int = 0,
) -> dict:
    """
    Пакетная конвертация всех HTML-файлов.

    Returns:
        Статистика: {total, converted, skipped, errors}
    """
    config = config or ConversionConfig()
    converter = HTMLToMarkdownConverter(config)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Собираем HTML-файлы
    html_files = sorted(input_dir.rglob('*.html'))

    # Исключаем служебные
    exclude_patterns = [
        'hmcontent', 'hmftsearch', 'hmkwindex', 'hmcontextids',
        'zoomimage', 'open.html'
    ]
    html_files = [
        f for f in html_files
        if not any(pat in f.name.lower() for pat in exclude_patterns)
    ]

    if max_files > 0:
        html_files = html_files[:max_files]

    stats = {'total': len(html_files), 'converted': 0, 'skipped': 0, 'errors': 0}

    logger.info(f"Найдено {len(html_files)} HTML-файлов")

    for i, html_path in enumerate(html_files, 1):
        if i % 100 == 0:
            logger.info(f"  Обработано {i}/{len(html_files)}...")

        try:
            md_content = converter.convert_file(html_path)

            if md_content is None:
                stats['skipped'] += 1
                continue

            # Добавляем метаданные из дерева если доступны
            if toc_tree:
                rel_name = html_path.name
                if rel_name in toc_tree:
                    toc_info = toc_tree[rel_name]
                    # Добавляем breadcrumbs из дерева в начало
                    toc_header = f'> **Раздел:** {toc_info["breadcrumbs"]}\n\n'
                    if not md_content.startswith('> **Путь:**'):
                        md_content = toc_header + md_content

            # Сохраняем
            # Сохраняем структуру директорий
            rel_path = html_path.relative_to(input_dir)
            md_path = output_dir / rel_path.with_suffix('.md')
            md_path.parent.mkdir(parents=True, exist_ok=True)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)

            stats['converted'] += 1

        except Exception as e:
            logger.error(f"Ошибка при обработке {html_path}: {e}")
            stats['errors'] += 1

    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Конвертер HTML-документации КОМПАС-3D в Markdown для RAG'
    )
    parser.add_argument(
        'input_dir',
        help='Директория с HTML-файлами документации'
    )
    parser.add_argument(
        '-o', '--output',
        default='./md_output',
        help='Директория для сохранения Markdown-файлов'
    )
    parser.add_argument(
        '--tree',
        default=None,
        help='Путь к файлу hmcontent.js с деревом документации'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=0,
        help='Максимальное число файлов (0 = все)'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Не включать изображения'
    )
    parser.add_argument(
        '--no-tables-md',
        action='store_true',
        help='Таблицы как текст, не как markdown-таблицы'
    )
    parser.add_argument(
        '--include-toc',
        action='store_true',
        help='Включать локальное оглавление'
    )
    parser.add_argument(
        '--no-dropdowns',
        action='store_true',
        help='Не включать содержимое раскрывающихся блоков'
    )
    parser.add_argument(
        '--include-footer',
        action='store_true',
        help='Включать футер'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Конфигурация
    config = ConversionConfig(
        include_images=not args.no_images,
        tables_as_markdown=not args.no_tables_md,
        include_local_toc=args.include_toc,
        include_dropdowns=not args.no_dropdowns,
        include_footer=args.include_footer,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Директория не найдена: {input_dir}")
        return

    # Парсим дерево документации если указано
    toc_tree = None
    if args.tree:
        tree_path = Path(args.tree)
        if tree_path.exists():
            logger.info(f"Парсинг дерева документации: {tree_path}")
            toc_tree = parse_toc_tree(tree_path)
            logger.info(f"  Найдено {len(toc_tree)} записей в дереве")
        else:
            logger.warning(f"Файл дерева не найден: {tree_path}")

    # Конвертация
    logger.info(f"Входная директория: {input_dir}")
    logger.info(f"Выходная директория: {output_dir}")
    logger.info(f"Конфигурация: images={config.include_images}, "
                f"tables_md={config.tables_as_markdown}, "
                f"dropdowns={config.include_dropdowns}")
    print()

    stats = batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        toc_tree=toc_tree,
        max_files=args.max_files,
    )

    print()
    print("=" * 50)
    print(f"Результат конвертации:")
    print(f"  Всего файлов:       {stats['total']}")
    print(f"  Сконвертировано:    {stats['converted']}")
    print(f"  Пропущено (пустые): {stats['skipped']}")
    print(f"  Ошибки:             {stats['errors']}")
    print("=" * 50)


if __name__ == '__main__':
    main()