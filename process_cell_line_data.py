import argparse
import json
import pandas as pd
import numpy as np
import os
import json


def get_args():
    parser = argparse.ArgumentParser(description="Pathway and Omics Data Processing")

    parser.add_argument('--in_path', type=str, default="data",
                        help='The input directory. All the data should be in this directory.')

    parser.add_argument('--pathway', type=str, required=True,
                        help='File name for the pathway JSON file, key is pathway name, value is gene list')
    parser.add_argument('--cell_line_meta', type=str, required=True,
                        help='File name for the cell line metadata file')
    parser.add_argument('--rna', type=str, required=True,
                        help='File name for RNA expression data file')
    parser.add_argument('--cnv', type=str, required=True,
                        help='File name for the CNV data file')
    parser.add_argument('--mutation', type=str, required=True,
                        help='File name for the mutation data file')

    parser.add_argument('--out_path', type=str, default="out",
                        help='The output directory. Generated file(s) will be stored in this directory.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.in_path):
        raise ValueError("The input directory does not exist.")

    os.makedirs(args.out_path, exist_ok=True)

    with open(os.path.join(args.in_path, args.pathway), "r") as f:
        pid_pathways = json.load(f)

    genes = []
    for gene in pid_pathways.values():
        genes.extend(gene)

    genes = set(genes)

    cell_line_mapping = pd.read_csv("data/Cell_listMon Jan 13 01_52_56 2025.csv")
    cell_line_mapping.columns = cell_line_mapping.columns.str.strip()
    cell_line_mapping = cell_line_mapping[cell_line_mapping.Datasets.eq("GDSC2")].drop_duplicates(["Model ID", "COSMIC ID"])

