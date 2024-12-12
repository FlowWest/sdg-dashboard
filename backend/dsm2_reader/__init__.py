from hecdss import HecDss
import pandas as pd

from typing import Dict


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
        return df


def get_all_data_from_dsm2_dss(
    dss: HecDss, part_names: Dict[str, str] | None = None, concat: bool = False
) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    out = {}
    cat = dss.get_catalog()
    paths = list(cat.recordTypeDict.keys())
    for i, path in enumerate(paths):
        path_data = dss.get(path)
        out[path] = pd.DataFrame(
            {
                "datetime": path_data.get_dates(),
                "value": path_data.get_values(),
                "unit": path_data.units,
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
