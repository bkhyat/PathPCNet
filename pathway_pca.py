import argparse
import os

import pandas as pd
from sklearn.decomposition import PCA

from utils import get_pathway_genes


def get_pathway_pcs(data: pd.DataFrame, pathways, prefix: str, n_components: int) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)

    for pathway, genes in pathways.items():
        common_genes = [gene for gene in genes if gene in data]

        n_components = min(n_components, len(common_genes))

        if n_components < 5:
            print(f"Data has only {n_components} genes for {pathway}.")

        pca = PCA(n_components=n_components)

        if n_components:
            pcs = pca.fit_transform(data[common_genes])
            result = result.assign(**{f"{prefix}_{pathway}.PC{i + 1}": pcs[:, i].tolist() for i in range(n_components)})

    return result


def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", required=True)
    parser.add_argument("-o", "--out_path", default="filtered_data")
    parser.add_argument("--n_pcs", type=int, default=4)

    return parser.parse_args()


def get_pca_from_raw_data_read(in_path: str, n_pcs=4):
    pathways, genes = get_pathway_genes(os.path.join(in_path, "pathways.json"))
    cnv_mat = pd.read_csv(os.path.join(in_path, "cnv_mat.csv"), index_col=0)
    mut_mat = pd.read_csv(os.path.join(in_path, "mut_mat.csv"), index_col=0)
    exp_mat = pd.read_csv(os.path.join(in_path, "exp_mat.csv"), index_col=0)

    for c in cnv_mat:
        rang = cnv_mat[c].max() - cnv_mat[c].min()
        cnv_mat[c] = (cnv_mat[c] - cnv_mat[c].min()) / rang

    cnv_mat = cnv_mat.fillna(0)  # Indicates normal copy
    mut_mat = mut_mat.fillna(0)  # Indicates absence of mutation

    for c in exp_mat:
        rang = exp_mat[c].max() - exp_mat[c].min()
        exp_mat[c] = (exp_mat[c] - exp_mat[c].min()) / rang

    exp_mat = exp_mat[exp_mat.columns[exp_mat.notna().all()]]

    cnv_pcs = get_pathway_pcs(cnv_mat, pathways, "CNV", n_pcs)
    mut_pcs = get_pathway_pcs(mut_mat, pathways, "MUT", n_pcs)
    rna_pcs = get_pathway_pcs(exp_mat, pathways, "EXP", n_pcs)

    return pd.concat([cnv_pcs, mut_pcs, rna_pcs], axis=1)


if __name__ == "__main__":
    args = parse_parameter()

    os.makedirs(args.out_path, exist_ok=True)
    cell_line_feat_mat = get_pca_from_raw_data_read(args.in_path)

    cell_line_feat_mat.to_csv(os.path.join(args.out_path, "cell_line_feat_mat.csv"))
