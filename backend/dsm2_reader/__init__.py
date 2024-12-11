from hecdss import HecDss
import pandas as pd


def get_all_data(dss):
    out = {}
    parts_letters = ["A", "B", "C", "D", "E", "F"]
    cat = dss.get_catalog()
    paths = list(cat.recordTypeDict.keys())
    for path in paths:
        path_data = dss.get(path)
        out[path] = pd.DataFrame(
            {
                "datetime": path_data.get_dates(),
                "value": path_data.get_values(),
                "unit": path_data.units,
            }
        )

    return out
