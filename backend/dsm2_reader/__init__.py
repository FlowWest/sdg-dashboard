from hecdss import HecDss
import pandas as pd


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
    dss, part_b="gate", part_c="parameter", part_f="scenario"
):
    out = {}
    cat = dss.get_catalog()
    paths = list(cat.recordtypedict.keys())
    for i, path in enumerate(paths):
        path_data = dss.get(path)
        out[path] = pd.dataframe(
            {
                part_b: cat.items[i].b,
                part_c: cat.items[i].c,
                part_f: cat.items[i].f,
                "datetime": path_data.get_dates(),
                "value": path_data.get_values(),
                "unit": path_data.units,
            }
        )

    return out
