from typing import Union
import zipfile
import pathlib
import pickle
import argparse

import pandas as pd

from rating_model import PICKLE_PROTOCOL, TeamResults


def unpickle_zip(zip_file, filename: str):
    with zip_file.open(filename, "r") as file_info:
        return pickle.load(file_info)


def save_filtered_data(all_results: TeamResults, selected_tour_ids, path_to_save: Union[str, pathlib.Path]):
    data = TeamResults()
    for id in selected_tour_ids:
        if id in all_results.tours:
            data.add_result(id, all_results[id])

    with open(path_to_save, "wb") as file_dump:
        pickle.dump(data, file_dump, protocol=PICKLE_PROTOCOL)


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    tours = pd.read_pickle(args.tour_path)

    with zipfile.ZipFile(args.input, "r") as zip_file:
        with zip_file.open("results.pkl", "r") as file_info:
            results = TeamResults.load_pickle(file_info)

    results.filter_incorrect_questions_tours()
    train_tour_ids = tours.index[tours["dateStart"].dt.year == args.train_year]
    test_tour_ids = tours.index[tours["dateStart"].dt.year == args.test_year]

    save_filtered_data(results, train_tour_ids, out_dir / "train_team_results.pickle")
    save_filtered_data(results, test_tour_ids, out_dir / "test_team_results.pickle")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="A path to zip file")
    parser.add_argument("-t", "--tournaments", dest="tour_path", type=str, required=True,
                        help="A path to pandas dataframe with tournamnets")
    parser.add_argument("-o", dest="out_dir", type=str, required=True, help="A path to output dir")
    parser.add_argument("--train_year", type=int, default=2019,
                        help="A dateStart year for train data")
    parser.add_argument("--test_year", type=int, default=2020,
                        help="A dateStart year for test data")

    args = parser.parse_args()

    main(args)
