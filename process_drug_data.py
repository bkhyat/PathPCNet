import pandas as pd
import pubchempy as pcp
import functools
import argparse
import os
import time
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def get_morgan_fingerprint(smiles_list, n_bits):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    mfps = []
    for smiles in smiles_list:
        fp = morgan_gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        mfps.append({f"BIT_{i}": bit for i, bit in enumerate(fp)})
    return mfps


def retry(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        count = 0
        while count<10:
            try:
                return func(*args, **kwargs)
            except pcp.ServerBusyError as e:
                count += 1
                time.sleep(1)

        raise e
    return wrapper


@retry
def pull_smiles(row):
    compounds = []
    pubchemids = str(row["pubchem"]).split(",")
    for pubchemid in pubchemids:
        if all(c.isdigit() for c in pubchemid.strip()):
            compounds.append(pcp.Compound.from_cid(int(pubchemid.strip())))

    smiles = set()
    for compound in compounds:
        smile = compound.smiles
        if smile not in smiles:
            smiles.add(smile)

    return list(smiles)


def pull_smiles_from_pubchem(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df = df[df["pubchem"].apply(lambda x: isinstance(x, str) and all(c.isdigit() for c in x))].set_index(
        "drug_id").sort_index()
    smiles = []
    for _, row in df.iterrows():
        smiles.append(pull_smiles(row))

    df["smiles"] = smiles

    return df

def parse_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--drug_list_file_path", required=True)
    parser.add_argument("-o", "--out_dir", default="filtered_data")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parameter()
    os.makedirs(args.out_dir, exist_ok=True)
    drugs_df = pd.read_csv(args.drug_list_file_path)
    drugs_df.columns = drugs_df.columns.str.strip().str.lower()
    drugs_df = drugs_df.loc[drugs_df["datasets"].eq("GDSC2")]
    drugs_df.reset_index(drop=True)
    smiles = drugs_df.apply(pull_smiles, axis=1)
    drugs_df["smiles"] = smiles
    drugs_df = drugs_df[drugs_df.smiles.str.len() == 1].reset_index(drop=True)
    drugs_df = drugs_df.explode("smiles")
    drugs_df=drugs_df.loc[drugs_df.groupby("smiles")["number of cell lines"].idxmax()].reset_index(drop=True)
    file_path = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.drug_list_file_path))[0] + "SMILES.csv")
    drugs_df.to_csv(file_path, index=False)

    print(f"COMPLETE: SMILES pulled in {file_path}")

