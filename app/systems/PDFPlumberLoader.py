import os
import re
from typing import List, Iterator, Optional, Callable, Any
import pdfplumber
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class PDFPlumberLoader(BaseLoader):
    """
    Загрузчик PDF-файлов с помощью pdfplumber.

    Аргументы:
        file_path (str): Путь к PDF-файлу.
        password (str, optional): Пароль для зашифрованного PDF.
        clean_function (Callable, optional): Внешняя функция для очистки текста.
            Если не указана, используется встроенная приватная очистка.
    """

    # ---------- Константы (паттерны для очистки) ----------
    _COPYRIGHT_PATTERNS = [
        r'Ни одна из частей этого документа.*?письменного разрешения АО «ИнфоТеКС»\.',
        r'ViPNet является зарегистрированным товарным знаком АО «ИнфоТеКС»\.',
        r'Все названия компаний.*?принадлежат соответствующим владельцам\.',
        r'Copyright \(c\) InfoTeCS',
        r'®',
        r'™'
    ]

    _CONTACT_PATTERNS = [
        r'АО «ИнфоТеКС»[\s\S]*?тел\.?[:\s]*[+]?[\d\s()-]{10,}',
        r'Телефон:.*?(?=\n|$)',
        r'Сайт:.*?(?=\n|$)',
        r'Служба поддержки:.*?(?=\n|$)',
        r'125167, г\. Москва,.*?(?=\n|$)',
        r'8 \(800\) 250-0260.*?(?=\n|$)',
        r'\+7 \(495\) 737-6192.*?(?=\n|$)',
        r'infotecs\.ru',
        r'hotline@infotecs\.ru'
    ]

    _TOC_MID_PATTERNS = [
        r'(?i)^\s*(приложение\s+[a-zа-я]\.?\s*.*?)(?:\d+\s*$)?',
        r'(?i)^\s*(неполадки в работе.*?)(?:\d+\s*$)?',
        r'(?i)^\s*(ошибка создания.*?)(?:\d+\s*$)?',
        r'(?i)^\s*(низкая скорость.*?)(?:\d+\s*$)?',
        r'(?i)^\s*(особенности.*?)(?:\d+\s*$)?',
        r'(?i)^\s*после настройки.*?(?:\d+\s*$)?',
        r'(?i)^\s*не отображаются.*?(?:\d+\s*$)?',
        r'(?i)^\s*блокирование.*?(?:\d+\s*$)?',
    ]

    _HEADER_WITH_PAGE_PATTERNS = [
        r'(?i)^\s*(о документе|соглашения документа|обратная связь|введение)\s+\d+\s*$',
        r'^\s*[А-Я][а-я]+\s+\d+\s*$',
    ]

    _TOC_START_PATTERNS = [
        r'(?i)^\s*содержан(ие|ия)\s*[\n\r]',
        r'(?i)^\s*оглавлен(ие|ия)\s*[\n\r]',
        r'(?i)^\s*table\s+of\s+contents\s*[\n\r]',
        r'(?i)^\s*contents?\s*[\n\r]',
    ]

    def __init__(
        self,
        file_path: str,
        password: Optional[str] = None,
        clean_function: Optional[Callable[[str], str]] = None
    ) -> None:
        self.file_path = file_path
        self.password = password
        self.clean_function = clean_function

    def load(self) -> List[Document]:
        """Загружает PDF и возвращает список документов (один документ на файл)."""
        with pdfplumber.open(self.file_path, password=self.password) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if self.clean_function is not None:
                text = self.clean_function(text)
            else:
                text = self._clean_text(text)

            metadata = {
                "source": self.file_path,
                "file_name": os.path.basename(self.file_path),
                "page_count": len(pdf.pages),
            }
            return [Document(page_content=text, metadata=metadata)]

    def lazy_load(self) -> Iterator[Document]:
        """Ленивая загрузка – возвращает итератор (экономия памяти)."""
        with pdfplumber.open(self.file_path, password=self.password) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if self.clean_function is not None:
                text = self.clean_function(text)
            else:
                text = self._clean_text(text)

            metadata = {
                "source": self.file_path,
                "file_name": os.path.basename(self.file_path),
                "page_count": len(pdf.pages),
            }
            yield Document(page_content=text, metadata=metadata)

    # ---------- Приватные методы очистки ----------
    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Основной метод очистки текста (вызывает все этапы)."""
        text = cls._remove_table_of_contents(text)
        text = cls._remove_toc_leftovers(text)
        text = cls._clean_basic_footers_and_headers(text)
        text = cls._remove_copyright_and_contacts(text)
        text = cls._filter_device_lines(text)
        text = cls._normalize_whitespace(text)
        return text

    @classmethod
    def _remove_table_of_contents(cls, text: str) -> str:
        """Удаляет оглавление целиком."""
        lines = text.split('\n')
        toc_start_index = -1

        # Ищем начало оглавления в первых 150 строках
        for i, line in enumerate(lines[:150]):
            for pattern in cls._TOC_START_PATTERNS:
                if re.match(pattern, line, re.IGNORECASE):
                    toc_start_index = i
                    break
            if toc_start_index != -1:
                break

        if toc_start_index == -1:
            return text

        # Ищем конец оглавления (5 последовательных строк, не похожих на строки оглавления)
        consecutive_non_toc = 0
        toc_end_index = -1

        for i in range(toc_start_index + 1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            if cls._is_toc_line(line):
                consecutive_non_toc = 0
                toc_end_index = i
            else:
                consecutive_non_toc += 1
                if consecutive_non_toc >= 5:
                    break

        if toc_end_index == -1:
            return text

        # Формируем результат: до оглавления и после оглавления
        before_toc = lines[:toc_start_index]
        after_toc = lines[toc_end_index + 1:]

        # Если последняя строка перед оглавлением — это заголовок "Содержание", удаляем её
        if before_toc and re.match(r'(?i)^\s*содержан|оглавлен', before_toc[-1]):
            before_toc = before_toc[:-1]

        return '\n'.join(before_toc + after_toc)

    @classmethod
    def _remove_toc_leftovers(cls, text: str) -> str:
        """Удаляет остатки оглавления, которые не были удалены на первом этапе."""
        lines = text.split('\n')
        filtered_lines = []
        skip_next = False

        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue

            original_line = line
            line_stripped = line.strip()

            if not line_stripped:
                continue

            # Пропускаем заголовки с номерами страниц (например, "О документе 13")
            if cls._is_header_with_page(line_stripped):
                # Пропускаем следующую строку, если она содержит номер в формате "| 14"
                if i + 1 < len(lines) and re.match(r'^\s*\|\s*\d+\s*$', lines[i + 1]):
                    skip_next = True
                continue

            # Пропускаем строки, похожие на элементы оглавления в середине документа
            if cls._is_mid_toc_line(line_stripped):
                continue

            # Пропускаем строки с точками-заполнителями и номерами страниц
            if cls._is_dotted_page_line(line_stripped):
                continue

            # Пропускаем строки, содержащие только номер страницы или точки
            if cls._is_just_number_or_dots(line_stripped):
                continue

            # Пропускаем явные заголовки оглавления
            if re.search(r'(?i)(содержание|оглавление|contents?|table\s+of\s+contents)', line_stripped):
                continue

            # Пропускаем строки с диапазонами страниц
            if cls._is_page_range_line(line_stripped):
                continue

            # Пропускаем строки вида "ViPNet Coordinator HW 5. Настройка ... | 13"
            if re.search(r'^.*\|\s*\d+\s*$', line_stripped):
                continue

            filtered_lines.append(original_line.rstrip())

        return '\n'.join(filtered_lines)

    @classmethod
    def _clean_basic_footers_and_headers(cls, text: str) -> str:
        """Удаляет колонтитулы, номера страниц, версии продукта и т.п."""
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'© АО «ИнфоТеКС», \d{4}', '', text)
        text = re.sub(r'ФРКЕ\.\d+\.\d+ИС\d', '', text)
        text = re.sub(r'Версия продукта [\d\.]+, документ обновлен [\d\.]+', '', text)
        text = re.sub(r'\|\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\|\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text

    @classmethod
    def _remove_copyright_and_contacts(cls, text: str) -> str:
        """Удаляет информацию о копирайте и контактные данные."""
        for pattern in cls._COPYRIGHT_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.DOTALL)

        for pattern in cls._CONTACT_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text

    @classmethod
    def _filter_device_lines(cls, text: str) -> str:
        """Удаляет строки, содержащие названия моделей устройств и версии продукта."""
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                continue

            if re.match(r'^ViPNet Coordinator.*(HW\d+|VA)$', line_stripped):
                continue

            if re.match(r'^Версия продукта:?.*$', line_stripped, re.IGNORECASE):
                continue

            filtered_lines.append(line_stripped)

        return ' '.join(filtered_lines)

    @classmethod
    def _normalize_whitespace(cls, text: str) -> str:
        """Убирает лишние пробелы, маркеры списков и нормализует переносы строк."""
        text = re.sub(r'[•·]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        text = re.sub(r'^[.\s]+|[.\s]+$', '', text)
        return text

    # ---------- Вспомогательные методы для определения строк оглавления ----------
    @classmethod
    def _is_toc_line(cls, line: str) -> bool:
        """Проверяет, является ли строка частью оглавления."""
        line = line.strip()
        if not line:
            return False

        if re.search(r'[\.\-\s]{8,}.*?\d+', line):
            return True

        if re.search(r'^\s*(\d+\.?\d*|[IVXLCDM]+\.?)\s+\D+\d+\s*$', line):
            return True

        if re.search(r'(?i)(глав[аы]|раздел|часть|section|chapter).*\d+\s*$', line):
            return True

        if re.search(r'(?i)^\s*приложение\s+[a-zа-я]\.?\s+.*?\d+\s*$', line):
            return True

        if re.search(r'(?i)(неполадки|ошибка|низкая скорость|особенности|не отображаются|блокирование).*\d+\s*$', line):
            return True

        return False

    @classmethod
    def _is_header_with_page(cls, line: str) -> bool:
        """Проверяет, является ли строка заголовком, за которым сразу следует номер страницы."""
        for pattern in cls._HEADER_WITH_PAGE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _is_mid_toc_line(cls, line: str) -> bool:
        """Проверяет, похожа ли строка на элемент оглавления, встречающийся в середине документа."""
        for pattern in cls._TOC_MID_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Дополнительная проверка: не является ли это реальным заголовком раздела
                if not line.endswith('.') and len(line) < 80:
                    return True
        return False

    @classmethod
    def _is_dotted_page_line(cls, line: str) -> bool:
        """Проверяет, содержит ли строка точки/тире в качестве заполнителя и номер страницы."""
        return bool(re.search(r'[\.\-\s]{10,}.*?\d+\s*$', line))

    @classmethod
    def _is_just_number_or_dots(cls, line: str) -> bool:
        """Проверяет, состоит ли строка преимущественно из цифр, точек или тире."""
        if re.match(r'^\s*[\d\.\-\s]+$', line):
            return True

        dots_and_dashes = len(re.findall(r'[\.\-]', line))
        if dots_and_dashes > len(line) * 0.3:
            return True

        digits = sum(c.isdigit() for c in line)
        letters = sum(c.isalpha() for c in line)
        if digits > letters * 0.3:
            return True

        return False

    @classmethod
    def _is_page_range_line(cls, line: str) -> bool:
        """Проверяет, является ли строка диапазоном страниц (например, 12-15)."""
        return bool(re.search(r'\d+\s*[-–—]\s*\d+\s*$', line)) and len(line) < 80