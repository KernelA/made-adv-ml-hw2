import zipfile
import pathlib
import pickle
import argparse

import pandas as pd

from rating_model import PICKLE_PROTOCOL


def unpickle_zip(zip_file, filename: str):
    with zip_file.open(filename, "r") as file_info:
        return pickle.load(file_info)


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.years:
        args.years = list(map(int, args.years))

    with zipfile.ZipFile(args.input, "r") as zip_file:
        players = unpickle_zip(zip_file, "players.pkl")
        players = pd.DataFrame.from_dict(players, orient="index")
        players.index.name = "id"
        players.drop("id", axis="columns", inplace=True)
        players.to_pickle(out_dir / "players-dt.pickle", protocol=PICKLE_PROTOCOL)

        del players

        tournaments = unpickle_zip(zip_file, "tournaments.pkl")
        tournaments = pd.DataFrame.from_dict(tournaments, orient="index")
        tournaments.index.name = "id"
        tournaments.drop("id", axis="columns", inplace=True)
        tournaments["dateStart"] = pd.to_datetime(tournaments["dateStart"], utc=True)
        tournaments["dateEnd"] = pd.to_datetime(tournaments["dateEnd"], utc=True)
        if args.years:
            tournaments = tournaments[tournaments["dateStart"].dt.year.isin(args.years)]
        tournaments.to_pickle(out_dir / "tournaments-dt.pickle", protocol=PICKLE_PROTOCOL)

        del tournaments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        required=True, help="A path to zip file")
    parser.add_argument("-o", dest="out_dir", type=str, required=True, help="A path to out dir")
    parser.add_argument("--years",  nargs="+", default=None, help="Save only selected years")

    args = parser.parse_args()

    main(args)
