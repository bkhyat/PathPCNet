import argparse
import json
import os

import numpy as np
import pandas as pd

from process_drug_data import pull_smiles_from_pubchem
from utils import parse_pathway


def get_rna_data(file_path, cell_line_mapping, gene_subset=None):
    df = pd.read_csv(file_path)
    mask = (df["dataset_name"].eq('Sanger & Broad Cell Lines RNASeq')
            & (df["data_source"].str.lower().str.strip() == "sanger")
            )
    if gene_subset is not None:
        mask &= (df["gene_symbol"].isin(genes))

    return (
        df[mask].pivot(index=["model_id"], columns="gene_symbol", values="tpm")
        .apply(lambda tpm: np.log2(tpm + 1))
        .merge(cell_line_mapping[["Model ID", "COSMIC ID"]],
               left_index=True,
               right_on="Model ID")
        .set_index("COSMIC ID")
        .drop(columns="Model ID")
        .sort_index(level=[0, 1])
    )


def get_mut_data(file_path, cell_line_mapping, gene_subset=None, vaf=True):
    df = pd.read_csv(file_path)
    mask = df["source"].str.lower().eq("sanger")
    if gene_subset is not None:
        mask &= (df["gene_symbol"].isin(genes))

    df = df[mask].drop_duplicates(["model_id", "gene_symbol"])

    if vaf:
        df = df.pivot(index=["model_id"], columns="gene_symbol", values="vaf").fillna(0)
    else:
        df = pd.crosstab(df["model_id"], df["gene_symbol"])
    return (
        df.merge(cell_line_mapping[["Model ID", "COSMIC ID"]], left_index=True, right_on="Model ID")
        .set_index("COSMIC ID")
        .drop(columns="Model ID")
        .sort_index(level=[0, 1])
    )


def get_cnv_data(file_path, cell_line_mapping, gene_subset=None, seg_mean=True):
    df = pd.read_csv(file_path)
    mask = df["source"].str.lower().str.strip() == "sanger"
    if gene_subset is not None:
        mask &= (df["symbol"].isin(genes))

    df = df[mask]

    if seg_mean:
        return df.pivot(columns=["symbol"], index="model_id", values="seg_mean")
    else:
        return (df.pivot(columns=["symbol"], index="model_id", values="cn_category")
                .replace({"Deletion": -2, "Loss": -1, "Neutral": 0, "Gain": 1, "Amplification": 2})
                .merge(cell_line_mapping[["Model ID", "COSMIC ID"]], left_index=True, right_on="Model ID")
                .set_index("COSMIC ID")
                .drop(columns="Model ID")
                .sort_index(level=[0, 1]))


def get_args():
    parser = argparse.ArgumentParser(description="Pathway and Omics Data Processing")

    parser.add_argument('--in_path', type=str, default="data",
                        help='The input directory. All the data should be in this directory.')

    parser.add_argument('--pathway', type=str, required=True,
                        help='File name for the pathway JSON file, key is pathway name, value is gene list')
    parser.add_argument('--cell_line_list', type=str, required=True,
                        help='File name for the cell line metadata file')
    parser.add_argument('--drugs', type=str, required=True,
                        help='File name for the GDSC drug data file')

    parser.add_argument('--rna', type=str, required=True,
                        help='File name for RNA expression data file')
    parser.add_argument('--cnv', type=str, required=True,
                        help='File name for the CNV data file in provided input directory --in_path')
    parser.add_argument('--mutation', type=str, required=True,
                        help='File name for the mutation data file')
    parser.add_argument('--response', type=str, required=True,
                        help='File name for the fitted drug response file')
    parser.add_argument('--out_path', type=str, default="processed_data",
                        help='The output directory. Generated files will be stored in this directory.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.in_path):
        raise ValueError("The input directory does not exist.")

    os.makedirs(args.out_path, exist_ok=True)

    pathways, genes = parse_pathway(os.path.join(args.in_path, args.pathway))

    cell_line_mapping = pd.read_csv(os.path.join(args.in_path, args.cell_line_list))
    cell_line_mapping.columns = cell_line_mapping.columns.str.strip()
    cell_line_mapping = cell_line_mapping[cell_line_mapping.Datasets.eq("GDSC2")].drop_duplicates(
        ["Model ID", "COSMIC ID"])

    drug_smiles = pull_smiles_from_pubchem(os.path.join(args.in_path, args.drugs))
    drug_response = pd.read_excel(os.path.join(args.in_path, args.response))
    drug_response = drug_response.loc[drug_response["DRUG_ID"].isin(drug_smiles.explode("smiles").dropna().index)]
    drug_response.set_index(["COSMIC_ID", "DRUG_ID"], inplace=True)
    cell_line_mapping = cell_line_mapping[
        cell_line_mapping["COSMIC ID"].isin(drug_response.index.get_level_values(level=0).unique())]

    rna_mat = get_rna_data(os.path.join(args.in_path, args.rna), cell_line_mapping, genes)
    cnv_mat = get_cnv_data(os.path.join(args.in_path, args.cnv), cell_line_mapping, genes, seg_mean=False)
    mut_mat = get_mut_data(os.path.join(args.in_path, args.mutation), cell_line_mapping, genes)

    common_cell_lines = sorted(
        list(set(rna_mat.index).intersection(mut_mat.index, cnv_mat.index)))

    with open(os.path.join(args.out_path, "pathways.json"), "w") as f:
        json.dump(pathways, f)

    cnv_mat.loc[common_cell_lines].to_csv(os.path.join(args.out_path, "cnv_mat.csv"))
    mut_mat.loc[common_cell_lines].to_csv(os.path.join(args.out_path, "mut_mat.csv"))
    rna_mat.loc[common_cell_lines].to_csv(os.path.join(args.out_path, "rna_mat.csv"))
    (
        drug_response[drug_response.index.get_level_values(0).isin(common_cell_lines)][["LN_IC50"]]
        .to_csv(os.path.join(args.out_path, "drug_response.csv"))
    )
    drug_smiles.to_csv(os.path.join(args.out_path, "drug_smiles.csv"))
