import json
import yaml
from pathlib import Path


def load_file(file_input: str) -> dict:
    """Load file for different format and return a dict

    Args:
        file_input (str): path of the file

    Raises:
        yaml.parser.ParserError: Invalid yaml file. 
        FileExistsError: File does not exits.

    Returns:
        dict: Dict of data.
    """
    if file_input is None:
        kw_file={}
    else:
        if isinstance(file_input, dict):
            kw_file = file_input
        elif Path(file_input).exists():
            try:
                with open(file_input, "r", encoding="utf-8") as f:
                    kw_file = json.load(f)
                    # print(f"Load json file {file_input}")
            except json.JSONDecodeError:
                try:
                    with open(file_input, "r", encoding="utf-8") as f:
                        kw_file = yaml.safe_load(f)
                        # print(f"Load yaml file {file_input}")
                except yaml.parser.ParserError as e:
                    raise yaml.parser.ParserError(f"{file_input} is an invalid file") from e
        else:
            raise FileExistsError(f"{file_input} does not exist")
    return kw_file