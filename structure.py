import os
from pathlib import Path


def print_tree(start_path, ignore_dirs=None, ignore_files=None, max_depth=5):
    if ignore_dirs is None:
        ignore_dirs = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            "venv",
            "env"}
    if ignore_files is None:
        ignore_files = {".pyc", ".pyo", ".pyd", ".DS_Store", ".coverage"}

    prefix = ""

    def _print_tree(path, prefix, depth):
        if depth > max_depth:
            return

        # Получаем содержимое
        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return

        # Фильтруем
        items = [i for i in items if i not in ignore_dirs]

        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1

            if os.path.isdir(item_path):
                # Пропускаем игнорируемые директории
                if item in ignore_dirs:
                    continue

                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")

                # Рекурсивно обходим
                extension = "    " if is_last else "│   "
                _print_tree(item_path, prefix + extension, depth + 1)
            else:
                # Пропускаем игнорируемые файлы
                if any(item.endswith(ext) for ext in ignore_files):
                    continue

                print(f"{prefix}{'└── ' if is_last else '├── '}{item}")

    print(f"{start_path}/")
    _print_tree(start_path, "", 0)


# Использование
if __name__ == "__main__":
    project_root = Path(__file__).parent
    print_tree(str(project_root))
