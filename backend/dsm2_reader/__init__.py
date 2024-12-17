from hecdss import HecDss
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import os

from typing import Dict

from pandas.core.series import validate_bool_kwarg


@dataclass
class SDGScenario:
    path: str
    name: str

    def __locate_files(self):
        output_files = os.listdir("output")
        echo_file = [
            f"output/{f}" for f in output_files if "echo" in f and self.name in f
        ]
        dss_files = [
            f"output/{f}" for f in output_files if "dss" in f and self.name in f
        ]

        return {"echo": echo_file, "dss": dss_files}

    def __read_echo_file(self, filepath: str):
        with open(filepath, "r") as f:
            lines = f.readlines()

        start_index = next(
            (i for i, line in enumerate(lines) if "GATE_WEIR_DEVICE" in line), None
        )
        end_index = next(
            (i for i, line in enumerate(lines) if "END" in line and i > start_index),
            None,
        )

        if start_index is not None and end_index is not None:
            data_rows = lines[start_index + 1 : end_index]
            col_names = [
                "GATE_NAME",
                "DEVICE",
                "NDUPLICATE",
                "WIDTH",
                "ELEV",
                "HEIGHT",
                "CF_FROM_NODE",
                "CF_TO_NODE",
                "DEFAULT_OP",
            ]

            data = [line.split(maxsplit=len(col_names) - 1) for line in data_rows[1:]]
            df = pd.DataFrame(data, columns=col_names)
            df = df[df["DEVICE"] == "fish_passage"]
            return df

    def __post_init__(self):
        self.files = self.__locate_files()
        self.echo = self.__read_echo_file(self.files.get("echo")[0])

    def read_dss(
        self,
        dss: HecDss,
        part_names: Dict[str, str] | None = None,
        concat: bool = False,
    ) -> pd.DataFrame | Dict[str, pd.DataFrame]:
        out = {}
        cat = dss.get_catalog()
        paths = list(cat.recordTypeDict.keys())
        for i, path in enumerate(paths):
            path_data = dss.get(path)
            out[path] = pd.DataFrame(
                {
                    "datetime": path_data.get_dates(),  # type: ignore
                    "value": path_data.get_values(),  # type: ignore
                    "unit": path_data.units,  # type: ignore
                }
            )
            if part_names is not None:
                if "A" in part_names:
                    out[path][part_names.get("A")] = cat.items[i].A
                if "B" in part_names:
                    out[path][part_names.get("B")] = cat.items[i].B
                if "C" in part_names:
                    out[path][part_names.get("C")] = cat.items[i].C
                if "D" in part_names:
                    out[path][part_names.get("D")] = cat.items[i].D
                if "E" in part_names:
                    out[path][part_names.get("E")] = cat.items[i].E
                if "F" in part_names:
                    out[path][part_names.get("F")] = cat.items[i].F

        if concat:
            out = pd.concat(out.values())
        return out


def read_scenario(path):
    hydo_dss = ""
    sdg_dss = ""
    echo_file = ""


def read_echo_file(filepath: str):
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_index = next(
        (i for i, line in enumerate(lines) if "GATE_WEIR_DEVICE" in line), None
    )
    end_index = next(
        (i for i, line in enumerate(lines) if "END" in line and i > start_index), None
    )

    if start_index is not None and end_index is not None:
        data_rows = lines[start_index + 1 : end_index]
        col_names = [
            "GATE_NAME",
            "DEVICE",
            "NDUPLICATE",
            "WIDTH",
            "ELEV",
            "HEIGHT",
            "CF_FROM_NODE",
            "CF_TO_NODE",
            "DEFAULT_OP",
        ]

        data = [line.split(maxsplit=len(col_names) - 1) for line in data_rows[1:]]
        df = pd.DataFrame(data, columns=col_names)
        df = df[df["DEVICE"] == "fish_passage"]
        return df


def get_all_data_from_dsm2_dss(
    dss: HecDss,
    part_names: Dict[str, str] | None = None,
    concat: bool = False,
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    out = {}
    cat = dss.get_catalog()
    paths = list(cat.recordTypeDict.keys())
    for i, path in enumerate(paths):
        path_data = dss.get(path)
        out[path] = pd.DataFrame(
            {
                "datetime": path_data.get_dates(),  # type: ignore
                "value": path_data.get_values(),  # type: ignore
                "unit": path_data.units,  # type: ignore
            }
        )
        if part_names is not None:
            if "A" in part_names:
                out[path][part_names.get("A")] = cat.items[i].A
            if "B" in part_names:
                out[path][part_names.get("B")] = cat.items[i].B
            if "C" in part_names:
                out[path][part_names.get("C")] = cat.items[i].C
            if "D" in part_names:
                out[path][part_names.get("D")] = cat.items[i].D
            if "E" in part_names:
                out[path][part_names.get("E")] = cat.items[i].E
            if "F" in part_names:
                out[path][part_names.get("F")] = cat.items[i].F

    if concat:
        out = pd.concat(out.values())
    return out


def read_hydro_dss(filepath):
    dss = HecDss(filepath)
    part_names = {"B": "gate", "C": "parameter", "F": "scenario"}
    data = get_all_data_from_dsm2_dss(dss, part_names=part_names, concat=True)
    return data


def read_gates_dss(filepath):
    dss = HecDss(filepath)
    part_names = {"B": "gate_op", "F": "scenario"}
    data = get_all_data_from_dsm2_dss(dss, part_names=part_names, concat=True)
    return data


def read_scenario_dir(
    dir: str, v7_filter: str | None = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Reads and processes hydrological scenario data from a directory containing DSS files.
    Matches corresponding hydro and SDG files based on naming patterns and returns their processed data.

    The function expects a specific directory structure where DSS files are stored in an 'output'
    subdirectory. It matches hydro and SDG files that share the same base name (e.g., 'scenario1_hydro.dss'
    with 'scenario1_SDG.dss').

    Parameters
    ----------
    dir : str
        Path to the main scenario directory. The function will look for files in a subdirectory
        named 'output'.
    v7_filter : str | None, optional
        If provided, only processes files containing this string (case-insensitive).
        Useful for filtering specific versions of files. Defaults to None.

    Returns
    -------
    dict
        A dictionary where:
        - Keys are the paths to hydro DSS files (as strings)
        - Values are dictionaries containing:
            - 'hydro': Processed data from the hydro DSS file
            - 'sdg': Processed data from the corresponding SDG file

    Raises
    ------
    ValueError
        If the number of hydro files doesn't match the number of SDG files in the directory.

    Examples
    --------
    >>> # Read all DSS files in the directory
    >>> data = read_scenario_dir("scenarios/FPV2Ma")

    >>> # Read only version 7 files
    >>> data = read_scenario_dir("scenarios/FPV2Ma", v7_filter="V7")

    Notes
    -----
    - Files must have '.dss' extension for hydro and SDG files
    - Files must contain 'hydro' or 'sdg' in their names (case-insensitive)
    - Matching is based on the first part of the filename when split by underscores
    - The function will also identify '.inp' echo files, though these are not currently processed
    """
    # first need to get a list of all scnarios in this folder
    scenario_path = Path(dir)
    output_path = scenario_path / "output"
    output_path_files = (
        list(output_path.iterdir())
        if v7_filter is None
        else [f for f in output_path.iterdir() if v7_filter.lower() in f.name.lower()]
    )
    hydro_files = [
        f for f in output_path_files if "hydro" in f.name.lower() and f.suffix == ".dss"
    ]

    sdg_files = [
        f for f in output_path_files if "sdg" in f.name.lower() and f.suffix == ".dss"
    ]

    echo_files_in_path = [
        f for f in output_path_files if "echo" in f.name.lower() and f.suffix == ".inp"
    ]

    if len(hydro_files) != len(sdg_files):
        raise ValueError(f"the number of hydro files and sdg must be equal")

    matches = list()
    for f in hydro_files:
        for s in sdg_files:
            f_split = f.stem.split("_")
            s_split = s.stem.split("_")
            if len(f_split) == len(s_split) and f_split[0] == s_split[0]:
                print(f"mathing:\t\t{f} <----> {s}")
                matches.append((f, s))

    data = {}
    try:
        for match in matches:
            data[str(match[0])] = {
                "hydro": read_hydro_dss(str(match[0])),
                "sdg": read_gates_dss(str(match[1])),
            }

    except Exception as e:
        print(e)

    return data
