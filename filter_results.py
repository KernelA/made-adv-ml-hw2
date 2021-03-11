import zipfile
import pathlib
import pickle
import argparse

import pandas as pd

from rating_model import PICKLE_PROTOCOL, TeamResults


def unpickle_zip(zip_file, filename: str):
    with zip_file.open(filename, "r") as file_info:
        return pickle.load(file_info)


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    tours = pd.read_pickle(args.tour_path)

    with zipfile.ZipFile(args.input, "r") as zip_file:
        with zip_file.open("results.pkl", "r") as file_info:
            results = TeamResults.load_pickle(file_info)

    selected_tour_ids = tuple(filter(lambda x: x not in tours.index, results.results.keys()))

    for del_id in selected_tour_ids:
        results.results.pop(del_id)

    with open(out_dir / "team_results.pickle", "wb") as dump_file:
        pickle.dump(results, dump_file, protocol=PICKLE_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="A path to zip file")
    parser.add_argument("-t", "--tournaments", dest="tour_path", type=str, required=True,
                        help="A path to pandas dataframe with tournamnets")
    parser.add_argument("-o", dest="out_dir", type=str, required=True, help="A path to out dir")

    args = parser.parse_args()

    main(args)
