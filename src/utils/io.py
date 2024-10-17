"""utilities for read and save to json"""
import json
from pathlib import Path
from typing import Any

import jsonlines


def read_json(path: str = None) -> Any:
    """
    Reads a JSON file from the given path.
    """
    if (not path) or (Path(path).suffix not in [".json"]):
        raise ValueError("Please check Path is not None or Path is ended with (.json)")

    json_data = None
    with open(path, encoding="utf8") as file:
        json_data = json.load(file)
    return json_data


def save_to_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Saves data to a JSON file at the specified path.
    """
    if (not file_path) or (Path(file_path).suffix not in [".json"]):
        raise ValueError("Please check Path is not None or Path is ended with (.json)")

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=indent, ensure_ascii=False)

    # print(f"Save data successfully to path: {file_path}")


def save_to_jsonl(data: Any, file_path: str) -> None:
    """
    Saves a list of dictionaries to a JSON Lines file at the specified path.
    """
    with jsonlines.open(file_path, mode="w") as writer:
        writer.write_all(data)


def read_txt(path: str = None) -> Any:
    """
    Reads a text file from the given path and returns its lines as a list of strings.
    """
    txt_data = []
    with open(path, encoding="utf-8") as file:  # type: ignore
        for line in file:
            txt_data.append(line.strip())
    return txt_data

def save_to_txt(data: Any, file_path: str) -> None:
    """save .txt."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            if isinstance(data, list):
                for item in data:
                    print(item)
                    file.writelines(" ".join(map(str, item)))
            elif isinstance(data, str):
                file.write(data)

        print(f"Nội dung đã được lưu vào {file_path}.")
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")