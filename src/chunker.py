#!/usr/bin/env python3
"""
Семантический чанкер HTML-документации КОМПАС-3D для RAG.

Стратегия:
1. Разбиваем страницу на семантические секции по заголовкам (p_Head4, p_Section_Title, p_Procedure)
2. Внутри секций группируем связанные блоки (шаг + его пояснения, таблица + её название)
3. Контролируем размер чанков: слишком большие — дробим, слишком маленькие — склеиваем
4. Каждый чанк получает метаданные: breadcrumbs, заголовок секции, тип контента

Результат: список чанков с метаданными, готовых к эмбеддингу.
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
from bs4 import BeautifulSoup, Tag, NavigableString, Comment

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ТИПЫ И КОНФИГУРАЦИЯ
# ============================================================================

class BlockType(Enum):
    """Тип семантического блока."""
    # --- Структурные разделители (начинают новую секцию) ---
    PAGE_TITLE = auto()          # Заголовок страницы
    SECTION_HEADING = auto()     # p_Head4, p_Section_Title
    PROCEDURE_HEADING = auto()   # p_Procedure ("Порядок действий")

    # --- Контентные блоки ---
    PARAGRAPH = auto()           # p_bodytext, p_noNum, p_noNum2, p_Normal
    NUMBERED_STEP = auto()       # p_Numbered (шаг процедуры)
    NUMBERED_SUBSTEP = auto()    # p_Numbered2 (подшаг)
    BULLET_ITEM = auto()         # p_bulleted
    BULLET_ITEM_L2 = auto()      # p_bulleted2
    BULLET_ITEM_L3 = auto()      # p_bulleted3
    TABLE = auto()               # table (данные)
    TABLE_TITLE = auto()         # p_Table_Title
    NOTE = auto()                # table с иконкой note/warning/caution
    IMAGE = auto()               # p_Image_Center
    IMAGE_CAPTION = auto()       # p_Picture_Title, p_ImageCaption
    COMMAND_INVOCATION = auto()  # dropdown-toggle-body (способы вызова)

    # --- Навигационные / служебные ---
    LOCAL_TOC = auto()           # p_Z_LOC_TOC
    SEE_ALSO = auto()            # p_SeeAlso
    FOOTER = auto()              # topicfooter
    TERM_DEFINITION = auto()     # p_Term_Name + p_Term_Litera
    ANCHOR = auto()              # только hmanchor — удаляем

    # --- Прочее ---
    UNKNOWN = auto()

def _postprocess_markdown(md_text: str) -> str:
    """Финальная очистка markdown текста чанка."""
    
    # ===== BOLD НОРМАЛИЗАЦИЯ (надёжный подход) =====
    
    # Шаг 1: Все **** заменяем на маркер-разделитель
    md_text = md_text.replace('****', '★')
    
    # Шаг 2: Маркер между словами → пробел (было: **word1****word2** = word1★word2)
    md_text = re.sub(r'(\w)★(\w)', r'\1** **\2', md_text)
    
    # Шаг 3: Маркер в начале строки/после пунктуации → ** (было: \n****Word = \n★Word)
    md_text = md_text.replace('★', '**')
    
    # Шаг 4: Склеиваем смежные bold: **text1** **text2** → **text1 text2**
    for _ in range(5):
        md_text = re.sub(r'\*\*([^*]+)\*\*\s*\*\*([^*]+)\*\*', r'**\1 \2**', md_text)
    
    # Шаг 5: Пустое форматирование ** **
    md_text = re.sub(r'\*\*\s*\*\*', ' ', md_text)
    
    # Шаг 6: Тройные звёздочки  
    md_text = re.sub(r'\*{3,}', '**', md_text)
    
    # ===== ПРОБЕЛЫ =====
    
    # Лишний пробел перед пунктуацией
    md_text = re.sub(r' +([.,:;!?)])', r'\1', md_text)
    
    # Множественные пробелы → один
    md_text = re.sub(r'(?<=\S) {2,}(?=\S)', ' ', md_text)
    
    # Пробелы в конце строк
    md_text = re.sub(r'[ \t]+$', '', md_text, flags=re.MULTILINE)
    
    # Множественные пустые строки
    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    
    return md_text.strip()

# Какие типы блоков НАЧИНАЮТ новую секцию
SECTION_STARTERS = {
    BlockType.PAGE_TITLE,
    BlockType.SECTION_HEADING,
    BlockType.PROCEDURE_HEADING,
}

# Какие блоки ПРИЛИПАЮТ к предыдущему блоку (не отрываются)
STICKY_TO_PREVIOUS = {
    BlockType.IMAGE_CAPTION,     # подпись идёт после картинки
    BlockType.NOTE,              # примечание относится к предыдущему тексту
}

# Какие блоки ПРИЛИПАЮТ к следующему
STICKY_TO_NEXT = {
    BlockType.TABLE_TITLE,       # название таблицы → перед таблицей
}

# Блоки которые полностью исключаем из чанков
EXCLUDED_BLOCKS = {
    BlockType.FOOTER,
    BlockType.ANCHOR,
    BlockType.LOCAL_TOC,         # навигация — не контент для поиска
}

# Маппинг CSS-классов → BlockType
PARAGRAPH_CLASS_TO_BLOCK_TYPE = {
    'p_bodytext':           BlockType.PARAGRAPH,
    'p_noNum':              BlockType.PARAGRAPH,
    'p_noNum2':             BlockType.PARAGRAPH,
    'p_Normal':             BlockType.PARAGRAPH,

    'p_Head4':              BlockType.SECTION_HEADING,
    'p_Section_Title':      BlockType.SECTION_HEADING,

    'p_Procedure':          BlockType.PROCEDURE_HEADING,

    'p_Numbered':           BlockType.NUMBERED_STEP,
    'p_Numbered2':          BlockType.NUMBERED_SUBSTEP,

    'p_bulleted':           BlockType.BULLET_ITEM,
    'p_bulleted2':          BlockType.BULLET_ITEM_L2,
    'p_bulleted3':          BlockType.BULLET_ITEM_L3,

    'p_Table_Title':        BlockType.TABLE_TITLE,

    'p_Image_Center':       BlockType.IMAGE,
    'p_Image_Left':         BlockType.IMAGE,
    'p_Picture_Title':      BlockType.IMAGE_CAPTION,
    'p_ImageCaption':       BlockType.IMAGE_CAPTION,

    'p_SeeAlso':            BlockType.SEE_ALSO,
    'p_Z_LOC_TOC':          BlockType.LOCAL_TOC,

    'p_underway':           BlockType.PARAGRAPH,  # включаем как обычный текст
    'p_Z_New':              BlockType.PARAGRAPH,

    'p_StartPage_Link':     BlockType.PARAGRAPH,
    'p_StartPage_Sect':     BlockType.SECTION_HEADING,

    'p_Term_Name':          BlockType.TERM_DEFINITION,
    'p_Term_Litera':        BlockType.SECTION_HEADING,

    # Табличные параграфы — обычно внутри table, не на верхнем уровне
    'p_Table_Cell':         BlockType.PARAGRAPH,
    'p_Table_Cell_Center':  BlockType.PARAGRAPH,
    'p_Table_Col_Name':     BlockType.PARAGRAPH,
    'p_Table_Col_Name_Center': BlockType.PARAGRAPH,
    'p_Table_List':         BlockType.BULLET_ITEM,
    'p_Table_Para_NWA':     BlockType.PARAGRAPH,
    'p_Table_Para_0':       BlockType.PARAGRAPH,
    'p_Table_Section':      BlockType.PARAGRAPH,
}


@dataclass
class ChunkerConfig:
    """Настройки чанкера."""
    # Целевой размер чанка (в символах). Подбирается под модель эмбеддингов.
    target_chunk_size: int = 1500
    # Минимальный размер чанка — меньше склеиваем с соседним
    min_chunk_size: int = 200
    # Максимальный размер чанка — больше дробим
    max_chunk_size: int = 3000
    # Перекрытие при дроблении больших чанков (в символах)
    overlap_size: int = 200

    # Включать ли "Смотрите также" в отдельный чанк
    include_see_also: bool = True
    # Включать ли способы вызова команды
    include_command_invocation: bool = True
    # Включать ли изображения (ссылки на них)
    include_images: bool = True


@dataclass
class SemanticBlock:
    """Один семантический блок из HTML."""
    block_type: BlockType
    markdown: str                        # markdown-содержимое
    heading_text: str = ''               # текст заголовка (для SECTION_HEADING)
    source_classes: list[str] = field(default_factory=list)
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.markdown)


@dataclass
class Chunk:
    """Готовый чанк для индексации."""
    chunk_id: str                        # уникальный ID
    text: str                            # markdown-текст чанка
    char_count: int = 0
    token_count_approx: int = 0          # приблизительная оценка токенов

    # --- Метаданные ---
    source_file: str = ''                # имя исходного HTML-файла
    page_title: str = ''                 # заголовок страницы
    breadcrumbs: str = ''                # путь в дереве документации
    section_title: str = ''              # заголовок текущей секции (p_Head4)
    section_path: str = ''               # полный путь: page > section
    block_types: list[str] = field(default_factory=list)  # типы включённых блоков
    has_table: bool = False
    has_image: bool = False
    has_procedure: bool = False
    chunk_index: int = 0                 # порядковый номер чанка в странице

    def __post_init__(self):
        self.char_count = len(self.text)
        # Грубая оценка: ~4 символа на токен для русского текста
        self.token_count_approx = self.char_count // 4
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.source_file}:{self.chunk_index}:{self.text[:100]}".encode()
            ).hexdigest()[:12]


# ============================================================================
# ПАРСЕР: HTML → СЕМАНТИЧЕСКИЕ БЛОКИ
# ============================================================================

class SemanticBlockParser:
    """
    Парсит HTML-страницу документации в последовательность
    семантических блоков (SemanticBlock).
    """

    def __init__(self, md_converter):
        """
        Args:
            md_converter: экземпляр HTMLToMarkdownConverter из основного скрипта
        """
        self.md = md_converter

    def parse_page(self, soup: BeautifulSoup) -> tuple[dict, list[SemanticBlock]]:
        """
        Парсит страницу.

        Returns:
            (page_metadata, list_of_blocks)
        """
        metadata = self._extract_page_metadata(soup)
        blocks = []

        content_div = soup.find('div', id='hmpagebody_scroller')
        if content_div is None:
            return metadata, blocks

        for child in content_div.children:
            if isinstance(child, (NavigableString, Comment)):
                continue
            if not isinstance(child, Tag):
                continue

            child_blocks = self._classify_element(child)
            blocks.extend(child_blocks)

        return metadata, blocks


    def _extract_page_metadata(self, soup: BeautifulSoup) -> dict:
        """Извлекает метаданные страницы."""
        meta = {
            'title': '',
            'breadcrumbs': '',
        }

        title_el = soup.find('p', class_='topictitle')
        if title_el:
            meta['title'] = self._clean_text(title_el)

        bc_el = soup.find('p', id='ptopic_breadcrumbs')
        if bc_el:
            links = bc_el.find_all('a', href=True)
            if links:
                meta['breadcrumbs'] = ' → '.join(
                    self._clean_text(a) for a in links if self._clean_text(a)
                )

        return meta

    def _classify_element(self, element: Tag) -> list[SemanticBlock]:
        """Классифицирует HTML-элемент и возвращает список семантических блоков."""
        tag = element.name
        classes = set(element.get('class', []))

        # --- Футер ---
        if 'topicfooter' in classes:
            return [SemanticBlock(BlockType.FOOTER, '', source_classes=list(classes))]

        # --- Таблица ---
        if tag == 'table':
            return self._classify_table(element)

        # --- div ---
        if tag == 'div':
            return self._classify_div(element)

        # --- Параграф ---
        if tag == 'p':
            return self._classify_paragraph(element)

        # --- img на верхнем уровне ---
        if tag == 'img':
            md = self.md._process_image(element)
            if md:
                return [SemanticBlock(BlockType.IMAGE, md)]
            return []

        # --- hr ---
        if tag == 'hr':
            return []  # разделители не несут контента

        # --- Прочее ---
        md = self.md._process_inline_content(element)
        if md and md.strip():
            return [SemanticBlock(BlockType.UNKNOWN, md)]

        return []

    def _classify_paragraph(self, p_tag: Tag) -> list[SemanticBlock]:
        """Классифицирует параграф."""
        classes = set(p_tag.get('class', []))

        # Определяем тип
        block_type = BlockType.UNKNOWN
        for cls in classes:
            if cls in PARAGRAPH_CLASS_TO_BLOCK_TYPE:
                block_type = PARAGRAPH_CLASS_TO_BLOCK_TYPE[cls]
                break

        # Только якоря — пропускаем
        if self._is_anchor_only(p_tag):
            return [SemanticBlock(BlockType.ANCHOR, '')]

        # Конвертируем в markdown
        md = self.md._process_paragraph(p_tag)
        if not md or not md.strip():
            return []

        heading_text = ''
        if block_type in SECTION_STARTERS:
            heading_text = self._clean_text(p_tag)

        return [SemanticBlock(
            block_type=block_type,
            markdown=md,
            heading_text=heading_text,
            source_classes=list(classes),
        )]

    def _classify_table(self, table_tag: Tag) -> list[SemanticBlock]:
        """Классифицирует таблицу."""
        # Примечание / предупреждение?
        note_md = self.md._check_note_table(table_tag)
        if note_md:
            return [SemanticBlock(BlockType.NOTE, note_md)]

        # Dropdown (способы вызова)?
        if self.md._is_dropdown_table(table_tag):
            md = self.md._process_dropdown_table(table_tag)
            if md:
                return [SemanticBlock(BlockType.COMMAND_INVOCATION, md)]
            return []

        # Обычная таблица данных
        md = self.md._process_table(table_tag)
        if md:
            return [SemanticBlock(BlockType.TABLE, md)]
        return []

    def _classify_div(self, div_tag: Tag) -> list[SemanticBlock]:
        """Классифицирует div."""
        classes = set(div_tag.get('class', []))

        if 'topicfooter' in classes:
            return [SemanticBlock(BlockType.FOOTER, '')]

        if 'dropdown-toggle-body' in classes:
            md = self.md._process_content_block(div_tag)
            if md:
                return [SemanticBlock(BlockType.COMMAND_INVOCATION, md)]
            return []

        if 'p_Picture_Title' in classes:
            md = self.md._process_div(div_tag)
            if md:
                return [SemanticBlock(BlockType.IMAGE_CAPTION, md)]
            return []

        # Обычный div — рекурсивно разбираем содержимое
        blocks = []
        for child in div_tag.children:
            if isinstance(child, Tag):
                blocks.extend(self._classify_element(child))
        return blocks

    def _is_anchor_only(self, p_tag: Tag) -> bool:
        for child in p_tag.children:
            if isinstance(child, NavigableString) and str(child).strip():
                return False
            if isinstance(child, Tag) and 'hmanchor' not in child.get('class', []):
                return False
        return True

    def _clean_text(self, el: Tag) -> str:
        text = el.get_text().replace('\xa0', ' ')
        return re.sub(r'\s+', ' ', text).strip()


# ============================================================================
# ЧАНКЕР: СЕМАНТИЧЕСКИЕ БЛОКИ → ЧАНКИ
# ============================================================================

class SemanticChunker:
    """
    Группирует семантические блоки в чанки оптимального размера,
    соблюдая семантические границы.
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()

    def _make_id(self, source_file: str, index: int, text: str) -> str:
        """Генерирует уникальный ID чанка."""
        raw = f"{source_file}::{index}::{text[:200]}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()[:12]

    def _filter_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Убирает слишком маленькие и пустые чанки."""
        result = []
        for chunk in chunks:
            text = chunk.text.strip()
            # Убираем пустое форматирование
            clean = re.sub(r'[*#\-|_\s]', '', text)
            if len(clean) < 10:
                continue
            # Убираем чанки из одного заголовка без контента
            lines = [l for l in text.split('\n') if l.strip()]
            if len(lines) == 1 and lines[0].startswith('#'):
                continue
            # Минимальный размер
            if chunk.char_count < self.config.min_chunk_size:
                # Сохраняем если это таблица или процедура
                if chunk.has_table or chunk.has_procedure:
                    result.append(chunk)
                    continue
                # Сохраняем если это единственный чанк страницы
                if len(chunks) == 1:
                    result.append(chunk)
                    continue
                # Иначе пропускаем
                continue
            result.append(chunk)
        return result


    def chunk_page(
        self,
        page_metadata: dict,
        blocks: list[SemanticBlock],
        source_file: str = '',
    ) -> list[Chunk]:
        """
        Разбивает страницу на чанки.

        Алгоритм:
        1. Фильтруем исключённые блоки
        2. Разбиваем на секции по заголовкам
        3. Внутри секций группируем связанные блоки
        4. Контролируем размер: дробим большие, склеиваем маленькие
        """

        # 1. Фильтрация
        filtered = [b for b in blocks if b.block_type not in EXCLUDED_BLOCKS]

        if not self.config.include_see_also:
            filtered = [b for b in filtered if b.block_type != BlockType.SEE_ALSO]
        if not self.config.include_command_invocation:
            filtered = [b for b in filtered if b.block_type != BlockType.COMMAND_INVOCATION]
        if not self.config.include_images:
            filtered = [
                b for b in filtered
                if b.block_type not in (BlockType.IMAGE, BlockType.IMAGE_CAPTION)
            ]

        # Убираем пустые
        filtered = [b for b in filtered if b.markdown.strip()]

        if not filtered:
            return []

        # 2. Разбиваем на секции
        sections = self._split_into_sections(filtered)

        # 3. Формируем чанки
        chunks = []
        chunk_index = 0
        page_title = page_metadata.get('title', '')
        breadcrumbs = page_metadata.get('breadcrumbs', '')

        for section_title, section_blocks in sections:
            section_chunks = self._create_chunks_from_section(
                section_blocks,
                page_title=page_title,
                section_title=section_title,
                breadcrumbs=breadcrumbs,
                source_file=source_file,
                start_index=chunk_index,
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

            # Фильтрация мусорных чанков
        chunks = self._filter_chunks(chunks)

        # Назначаем ID
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.chunk_id = self._make_id(source_file, i, chunk.text)

        return chunks

    def _split_into_sections(
        self, blocks: list[SemanticBlock]
    ) -> list[tuple[str, list[SemanticBlock]]]:
        """
        Разбивает блоки на секции по заголовкам.

        Returns:
            Список (section_title, [blocks])
        """
        sections: list[tuple[str, list[SemanticBlock]]] = []
        current_title = ''
        current_blocks: list[SemanticBlock] = []

        for block in blocks:
            if block.block_type in SECTION_STARTERS and block.heading_text:
                # Сохраняем предыдущую секцию
                if current_blocks:
                    sections.append((current_title, current_blocks))

                current_title = block.heading_text
                current_blocks = [block]
            else:
                current_blocks.append(block)

        # Последняя секция
        if current_blocks:
            sections.append((current_title, current_blocks))

        return sections

    def _classify_paragraph(self, p_tag: Tag) -> list[SemanticBlock]:
        classes = set(p_tag.get('class', []))

        block_type = BlockType.UNKNOWN
        for cls in classes:
            if cls in PARAGRAPH_CLASS_TO_BLOCK_TYPE:
                block_type = PARAGRAPH_CLASS_TO_BLOCK_TYPE[cls]
                break

        if self._is_anchor_only(p_tag):
            return [SemanticBlock(BlockType.ANCHOR, '')]

        md = self.md._process_paragraph(p_tag)
        if not md or not md.strip():
            return []

        heading_text = ''
        if block_type in SECTION_STARTERS:
            # *** Чистый текст заголовка — без markdown-маркеров ***
            heading_text = self._clean_text(p_tag)
            # Убираем маркеры которые могли попасть из inline
            heading_text = heading_text.strip('*#_ ')

        return [SemanticBlock(
            block_type=block_type,
            markdown=md,
            heading_text=heading_text,
            source_classes=list(classes),
        )]

    def _split_large_block(self, text: str) -> list[str]:
        """Разбивает большой текстовый блок по абзацам."""
        paragraphs = text.split('\n\n')
        
        parts = []
        current = []
        current_size = 0
        target = self.config.target_chunk_size
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > target and current:
                parts.append('\n\n'.join(current))
                current = []
                current_size = 0
            
            current.append(para)
            current_size += para_size
        
        if current:
            parts.append('\n\n'.join(current))
        
        return parts if parts else [text]

    def _split_large_table(self, block: 'SemanticBlock') -> list[str]:
        """
        Разбивает большую markdown-таблицу на части по строкам.
        Каждая часть получает заголовок таблицы (первая строка + разделитель).
        """
        lines = block.markdown.split('\n')
        
        # Находим заголовок таблицы и разделитель
        header_lines = []
        data_lines = []
        in_header = True
        
        for line in lines:
            if in_header:
                header_lines.append(line)
                if re.match(r'^\|[\s\-:|]+\|$', line.strip()):
                    in_header = False
            else:
                if line.strip():
                    data_lines.append(line)
        
        if not data_lines:
            return [block.markdown]
        
        header = '\n'.join(header_lines)
        header_size = len(header)
        
        # Разбиваем строки таблицы на группы
        chunks_text = []
        current_rows = []
        current_size = header_size
        
        target = self.config.target_chunk_size
        
        for row in data_lines:
            row_size = len(row) + 1  # +1 для \n
            
            if current_size + row_size > target and current_rows:
                # Flush текущую группу
                chunk_text = header + '\n' + '\n'.join(current_rows)
                chunks_text.append(chunk_text)
                current_rows = []
                current_size = header_size
            
            current_rows.append(row)
            current_size += row_size
        
        # Последняя группа
        if current_rows:
            chunk_text = header + '\n' + '\n'.join(current_rows)
            chunks_text.append(chunk_text)
        
        return chunks_text

    def _create_chunks_from_section(
        self,
        blocks: list,                # ← первый позиционный (как в вызове)
        page_title: str = '',
        section_title: str = '',
        breadcrumbs: str = '',
        source_file: str = '',
        start_index: int = 0,
    ) -> list[Chunk]:
        """Создаёт чанки из блоков одной секции."""
        
        chunks = []
        current_texts = []
        current_size = 0
        current_types = set()
        

        def flush_chunk():
            nonlocal current_texts, current_size, current_types
            if not current_texts:
                return
            text = '\n\n'.join(current_texts)
            
            # Контекстный заголовок для продолжающих чанков
            if (section_title and
                section_title != page_title and
                len(chunks) > 0):
                text = f'## {section_title}\n\n' + text
            
            # Постобработка markdown
            text = _postprocess_markdown(text)
            
            chunk = Chunk(
                chunk_id='',
                text=text,
                source_file=source_file,
                page_title=page_title,
                breadcrumbs=breadcrumbs,
                section_title=section_title,
                section_path=(
                    f'{page_title} > {section_title}'
                    if section_title else page_title
                ),
                block_types=[t.name for t in current_types],
                has_table=BlockType.TABLE in current_types,
                has_image=BlockType.IMAGE in current_types,
                has_procedure=(
                    BlockType.NUMBERED_STEP in current_types
                    or BlockType.PROCEDURE_HEADING in current_types
                ),
                chunk_index=start_index + len(chunks),
            )
            chunks.append(chunk)
            
            current_texts = []
            current_size = 0
            current_types = set()
        
        for block in blocks:
            md = block.markdown.strip()
            if not md:
                continue
            
            block_size = len(md)
            
            # === БОЛЬШАЯ ТАБЛИЦА — дробим ===
            if (block.block_type == BlockType.TABLE and 
                block_size > self.config.max_chunk_size):
                
                # Сначала flush текущий накопленный чанк
                flush_chunk()
                
                # Разбиваем таблицу на части
                table_parts = self._split_large_table(block)
                
                for part_text in table_parts:
                    # Каждая часть — отдельный чанк
                    current_texts = [part_text]
                    current_size = len(part_text)
                    current_types = {BlockType.TABLE}
                    flush_chunk()
                
                continue
            
            # === БОЛЬШОЙ БЛОК (не таблица) — дробим по абзацам ===
            if block_size > self.config.max_chunk_size:
                flush_chunk()
                
                sub_parts = self._split_large_block(md)
                for part in sub_parts:
                    current_texts = [part]
                    current_size = len(part)
                    current_types = {block.block_type}
                    flush_chunk()
                
                continue
            
            # === Обычная логика: проверяем поместится ли ===
            if (current_size + block_size > self.config.target_chunk_size 
                and current_texts):
                flush_chunk()
            
            current_texts.append(md)
            current_size += block_size
            current_types.add(block.block_type)
        
        # Flush остаток
        flush_chunk()
        
        return chunks

    def _group_sticky_blocks(
        self, blocks: list[SemanticBlock]
    ) -> list[list[SemanticBlock]]:
        """
        Группирует «прилипающие» блоки:
        - TABLE_TITLE + TABLE → одна группа
        - IMAGE + IMAGE_CAPTION → одна группа
        - NUMBERED_STEP + последующие пояснения (PARAGRAPH, BULLET_*) → одна группа
        """
        groups: list[list[SemanticBlock]] = []
        current_group: list[SemanticBlock] = []

        i = 0
        while i < len(blocks):
            block = blocks[i]

            # TABLE_TITLE прилипает к следующей TABLE
            if block.block_type == BlockType.TABLE_TITLE:
                current_group.append(block)
                if i + 1 < len(blocks) and blocks[i + 1].block_type == BlockType.TABLE:
                    current_group.append(blocks[i + 1])
                    i += 1
                if current_group:
                    groups.append(current_group)
                    current_group = []
                i += 1
                continue

            # IMAGE_CAPTION прилипает к предыдущему IMAGE
            if block.block_type == BlockType.IMAGE_CAPTION:
                if groups and groups[-1][-1].block_type == BlockType.IMAGE:
                    groups[-1].append(block)
                else:
                    current_group.append(block)
                    groups.append(current_group)
                    current_group = []
                i += 1
                continue

            # NOTE прилипает к предыдущему блоку
            if block.block_type == BlockType.NOTE:
                if groups:
                    groups[-1].append(block)
                else:
                    groups.append([block])
                i += 1
                continue

            # NUMBERED_STEP: собираем шаг + все его подблоки до следующего шага/секции
            if block.block_type == BlockType.NUMBERED_STEP:
                step_group = [block]
                j = i + 1
                while j < len(blocks):
                    next_block = blocks[j]
                    # Подблоки шага: подшаги, пояснения, списки, картинки, примечания
                    if next_block.block_type in (
                        BlockType.NUMBERED_SUBSTEP,
                        BlockType.PARAGRAPH,
                        BlockType.BULLET_ITEM,
                        BlockType.BULLET_ITEM_L2,
                        BlockType.BULLET_ITEM_L3,
                        BlockType.IMAGE,
                        BlockType.IMAGE_CAPTION,
                        BlockType.NOTE,
                        BlockType.TABLE,
                        BlockType.TABLE_TITLE,
                    ):
                        step_group.append(next_block)
                        j += 1
                    else:
                        break

                groups.append(step_group)
                i = j
                continue

            # Обычный блок
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([block])
            i += 1

        if current_group:
            groups.append(current_group)

        return groups

    def _split_large_group(
        self,
        group: list[SemanticBlock],
        page_title: str,
        section_title: str,
        breadcrumbs: str,
        source_file: str,
        start_index: int,
    ) -> list[Chunk]:
        """Дробит слишком большую группу блоков на чанки."""
        chunks = []
        current_texts = []
        current_size = 0

        for block in group:
            block_size = block.char_count

            if block_size > self.config.max_chunk_size:
                # Одиночный блок слишком большой (например, огромная таблица)
                # Дробим по строкам с перекрытием
                if current_texts:
                    chunks.append(self._make_chunk(
                        '\n\n'.join(current_texts),
                        page_title, section_title, breadcrumbs,
                        source_file, start_index + len(chunks),
                        [block.block_type.name]
                    ))
                    current_texts = []
                    current_size = 0

                text_chunks = self._split_text_with_overlap(
                    block.markdown,
                    self.config.target_chunk_size,
                    self.config.overlap_size,
                )
                for tc in text_chunks:
                    chunks.append(self._make_chunk(
                        tc, page_title, section_title, breadcrumbs,
                        source_file, start_index + len(chunks),
                        [block.block_type.name]
                    ))
                continue

            if current_size + block_size > self.config.target_chunk_size and current_texts:
                chunks.append(self._make_chunk(
                    '\n\n'.join(current_texts),
                    page_title, section_title, breadcrumbs,
                    source_file, start_index + len(chunks),
                    []
                ))
                current_texts = []
                current_size = 0

            current_texts.append(block.markdown)
            current_size += block_size

        if current_texts:
            chunks.append(self._make_chunk(
                '\n\n'.join(current_texts),
                page_title, section_title, breadcrumbs,
                source_file, start_index + len(chunks),
                []
            ))

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Склеивает слишком маленькие соседние чанки."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]

            if chunk.char_count < self.config.min_chunk_size and i + 1 < len(chunks):
                # Склеиваем с следующим
                next_chunk = chunks[i + 1]
                combined_text = chunk.text + '\n\n' + next_chunk.text

                merged_chunk = Chunk(
                    chunk_id='',
                    text=combined_text,
                    source_file=chunk.source_file,
                    page_title=chunk.page_title,
                    breadcrumbs=chunk.breadcrumbs,
                    section_title=chunk.section_title or next_chunk.section_title,
                    section_path=chunk.section_path or next_chunk.section_path,
                    block_types=list(set(chunk.block_types + next_chunk.block_types)),
                    has_table=chunk.has_table or next_chunk.has_table,
                    has_image=chunk.has_image or next_chunk.has_image,
                    has_procedure=chunk.has_procedure or next_chunk.has_procedure,
                    chunk_index=chunk.chunk_index,
                )
                merged.append(merged_chunk)
                i += 2
            else:
                merged.append(chunk)
                i += 1

        return merged

    def _split_text_with_overlap(
        self, text: str, target_size: int, overlap: int
    ) -> list[str]:
        """Дробит большой текст на части с перекрытием по границам абзацев."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_parts = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > target_size and current_parts:
                chunks.append('\n\n'.join(current_parts))

                # Перекрытие: берём последние N символов
                overlap_parts = []
                overlap_size = 0
                for p in reversed(current_parts):
                    if overlap_size + len(p) > overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_size += len(p)

                current_parts = overlap_parts
                current_size = overlap_size

            current_parts.append(para)
            current_size += para_size

        if current_parts:
            chunks.append('\n\n'.join(current_parts))

        return chunks

    def _make_chunk(
        self, text: str, page_title: str, section_title: str,
        breadcrumbs: str, source_file: str, index: int,
        block_types: list[str],
    ) -> Chunk:
        """Создаёт объект Chunk."""
        return Chunk(
            chunk_id='',
            text=text,
            source_file=source_file,
            page_title=page_title,
            breadcrumbs=breadcrumbs,
            section_title=section_title,
            section_path=f'{page_title} > {section_title}' if section_title else page_title,
            block_types=block_types,
            has_table='TABLE' in block_types,
            has_image='IMAGE' in block_types,
            has_procedure='NUMBERED_STEP' in block_types or 'PROCEDURE_HEADING' in block_types,
            chunk_index=index,
        )


# ============================================================================
# ПАЙПЛАЙН: HTML → ЧАНКИ
# ============================================================================

class KompasDocChunker:
    """
    Полный пайплайн: HTML-файл → список чанков с метаданными.
    Объединяет конвертер, парсер и чанкер.
    """

    def __init__(
        self,
        chunker_config: Optional[ChunkerConfig] = None,
        converter_config=None,  # ConversionConfig из основного скрипта
    ):
        # Импортируем конвертер (предполагается, что он в том же пакете)
        from convert_kompas_html_to_md import HTMLToMarkdownConverter, ConversionConfig

        self.converter = HTMLToMarkdownConverter(converter_config or ConversionConfig())
        self.parser = SemanticBlockParser(self.converter)
        self.chunker = SemanticChunker(chunker_config)

    def process_file(self, html_path: Path) -> list[Chunk]:
        """Обрабатывает один HTML-файл, возвращает список чанков."""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Ошибка чтения {html_path}: {e}")
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # Парсим в семантические блоки
        metadata, blocks = self.parser.parse_page(soup)

        if not blocks:
            return []

        # Формируем чанки
        chunks = self.chunker.chunk_page(
            page_metadata=metadata,
            blocks=blocks,
            source_file=html_path.name,
        )

        return chunks

    def process_directory(
        self,
        input_dir: Path,
        toc_tree: Optional[dict] = None,
    ) -> list[Chunk]:
        """Обрабатывает все HTML-файлы в директории."""
        all_chunks = []

        html_files = sorted(input_dir.rglob('*.html'))
        exclude = ['hmcontent', 'hmftsearch', 'hmkwindex', 'hmcontextids', 'zoomimage']
        html_files = [f for f in html_files if not any(e in f.name.lower() for e in exclude)]

        logger.info(f"Обработка {len(html_files)} файлов...")

        for i, path in enumerate(html_files, 1):
            if i % 100 == 0:
                logger.info(f"  {i}/{len(html_files)}...")

            chunks = self.process_file(path)

            # Обогащаем breadcrumbs из дерева
            if toc_tree and path.name in toc_tree:
                for chunk in chunks:
                    if not chunk.breadcrumbs:
                        chunk.breadcrumbs = toc_tree[path.name].get('breadcrumbs', '')

            all_chunks.extend(chunks)

        logger.info(f"Итого: {len(all_chunks)} чанков из {len(html_files)} файлов")
        return all_chunks


# ============================================================================
# ЭКСПОРТ ЧАНКОВ
# ============================================================================

def export_chunks_jsonl(chunks: list[Chunk], output_path: Path):
    """Экспорт чанков в JSONL (по одному JSON-объекту на строку)."""
    import json

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            record = {
                'id': chunk.chunk_id,
                'text': chunk.text,
                'metadata': {
                    'source_file': chunk.source_file,
                    'page_title': chunk.page_title,
                    'breadcrumbs': chunk.breadcrumbs,
                    'section_title': chunk.section_title,
                    'section_path': chunk.section_path,
                    'block_types': chunk.block_types,
                    'has_table': chunk.has_table,
                    'has_image': chunk.has_image,
                    'has_procedure': chunk.has_procedure,
                    'chunk_index': chunk.chunk_index,
                    'char_count': chunk.char_count,
                    'token_count_approx': chunk.token_count_approx,
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Экспортировано {len(chunks)} чанков в {output_path}")


def export_chunks_for_embedding(chunks: list[Chunk]) -> list[dict]:
    """Возвращает список словарей, готовых для отправки в embedding API."""
    results = []
    for chunk in chunks:
        # Формируем текст для эмбеддинга: breadcrumbs + section + content
        embedding_parts = []
        if chunk.breadcrumbs:
            embedding_parts.append(chunk.breadcrumbs)
        if chunk.page_title:
            embedding_parts.append(chunk.page_title)
        if chunk.section_title and chunk.section_title != chunk.page_title:
            embedding_parts.append(chunk.section_title)
        embedding_parts.append(chunk.text)

        results.append({
            'id': chunk.chunk_id,
            'text_for_embedding': '\n\n'.join(embedding_parts),
            'text_for_display': chunk.text,
            'metadata': {
                'source_file': chunk.source_file,
                'page_title': chunk.page_title,
                'section_path': chunk.section_path,
                'breadcrumbs': chunk.breadcrumbs,
                'has_table': chunk.has_table,
                'has_procedure': chunk.has_procedure,
            }
        })
    return results
