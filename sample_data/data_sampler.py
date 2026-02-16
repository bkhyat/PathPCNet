import argparse
import json
import os
import random

import pandas as pd


def create_sample_data(input_path: str = "", output_path: str = "sample_data"):
    with open(os.path.join(input_path, "pathways.json"), "r") as f:
        pathways = json.load(f)
    # Keep 10 shortest pathways
    pathways = dict(list(sorted(pathways.items(), key=lambda x: len(x[1])))[:10])
    genes = set()
    for gene_list in pathways.values():
        genes = genes.union(gene_list)
    genes = list(genes)

    cnv_mat = pd.read_csv(os.path.join(input_path, "cnv_mat.csv"), index_col=0)
    exp_mat = pd.read_csv(os.path.join(input_path, "exp_mat.csv"), index_col=0)
    mut_mat = pd.read_csv(os.path.join(input_path, "mut_mat.csv"), index_col=0)

    common_cell_lines = set(cnv_mat.index).intersection(exp_mat.index).intersection(mut_mat.index)

    drug_response = pd.read_csv(os.path.join(input_path, "drug_response.csv"))

    common_cell_lines = common_cell_lines.intersection(drug_response["COSMIC_ID"].unique())

    # Randomly sample 50 cell lines:
    common_cell_lines = list(common_cell_lines)
    filtered_cell_lines = random.choices(common_cell_lines, k=50)

    drug_response = drug_response[drug_response["COSMIC_ID"].isin(filtered_cell_lines)]

    smiles = pd.read_csv(os.path.join(input_path, "drug_smiles.csv"), index_col=0)

    common_drugs = list(set(smiles.index).intersection(drug_response["DRUG_ID"].unique()))

    smiles = smiles.loc[common_drugs].drop_duplicates("SMILES")

    # Randomly select 10 drugs
    smiles = smiles.sample(10)
    drug_response = drug_response[drug_response["DRUG_ID"].isin(smiles.index)]

    # Write sample data files to the output folder
    cnv_mat.loc[filtered_cell_lines, list(genes)].to_csv(os.path.join(output_path, "cnv_mat.csv"))
    exp_mat.loc[filtered_cell_lines, list(genes)].to_csv(os.path.join(output_path, "exp_mat.csv"))
    mut_mat.loc[filtered_cell_lines, list(genes)].to_csv(os.path.join(output_path, "mut_mat.csv"))

    drug_response.to_csv(os.path.join(output_path, "drug_response.csv"))
    smiles.to_csv(os.path.join(output_path, "drug_smiles.csv"))

    with open(os.path.join(output_path, "pathways.json"), "w") as f:
        json.dump(pathways, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="processed_data")
    parser.add_argument("--out_path", type=str, default="./")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    create_sample_data(args.in_path, args.out_path)
