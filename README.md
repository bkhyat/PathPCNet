# PathPCNet: Pathway Principal Component-Based Interpretable Framework for Drug Sensitivity Prediction

Bikhyat Adhikari, Masrur Sobhan, Ananda Sutradhar, Giri Narasimhan, Ananda Mohan Mondal
bioRxiv 2025.08.20.668802; doi: https://doi.org/10.1101/2025.08.20.668802

![End-to-end Pipeline for PathPCNet](images/workflow.png)

## Set up

Create Python 3.10 Virtual Environment and install the necessary package:

```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the evaluation on sample data

### 1. Traditional Machine Learning Methods on PathwayPC SAMPLE data

```shell
python traditional_regression.py -i sample_data -o traditional_output --n_pcs 2 --n_mfp_bits 128
```
You can additionally pass one or more values to `--models` argument: `--models xgboost lightgbm et rf ridge`. The
default behavior is to run all these five models if models argument is not passed.

### 1. The deep learning network on PathwayPC SAMPLE data
```shell
python run.py -i sample_data -o output --n_pcs 1 --n_mfp_bits 128 -cv 5 --cuda -1 -e 20 -lr 0.1 -b 32
```

## Running Full Experiment

Follow the steps below to run full experiment on whole data.

### 1. Preprocessing Raw Data

Create a new folder and download the necessary files:

- PID Pathway: https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
- GDSC2 Drug, Cell lines, and IC50 data:
    - Drug Screening IC50s: https://www.cancerrxgene.org/downloads/bulk_download
    - Drug List: https://www.cancerrxgene.org/compounds
    - Cell line List: https://www.cancerrxgene.org/celllines
- Download Cell line data from https://cellmodelpassports.sanger.ac.uk/downloads:
    - Expression Data: rnaseq_all_20220624/rnaseq_all_data_20220624.csv
    - Mutation Data: mutations_all_20230202.csv
    - Copy Number Data: WES_pureCN_CNV_genes_20221213 2/WES_pureCN_CNV_genes_20221213.csv

  Download the specified raw data files from the mentioned sources, extract if zip files, and move the specified files
  to a single folder, and provide the folder path as
  in_path, and specify the file names for each of the below. The script will convert the data into specific matrix
  format,
  apply transformation, and filter out common cell lines, pull SMILES.

**Note**: Update the file names for -cell_line_list and --drugs arguments below:

```shell
python process_raw_input_data.py --in_path data --pathway c2.cp.pid.v2026.1.Hs.symbols.gmt --cell_line_list "Cell_listMon Feb 16 23_10_40 2026.csv"  --drugs "Drug_listMon Feb 16 23_09_54 2026.csv"  --rna rnaseq_all_data_20220624.csv  --cnv WES_pureCN_CNV_genes_20221213.csv  --mutation mutations_all_20230202.csv  --response GDSC2_fitted_dose_response_27Oct23.xlsx  --out_path processed_data
```

### 2. Generating Pathway PCA from processed data

Before running the script, ensure the following files exist in the specified in_path folder:

```shell
python pathway_pca.py --in_path processed_data --out_path processed_data
```

### 3. Running experiments for traditional ML models

To run this code, it requires the three files from step 2(cell_line_feat.csv) and step 3(drug_smiles.csv, response.csv)
The script runs five traditional ML methods on pathway PCA and drug morgan fingerprint features.

```shell
python traditional_regression.py --in_path processed_data --out_path ./
```

### 4. Hyperparameter Tuning

### 5. Evaluating Model for Original Gene Feature

### 6. Evaluating Model for Pathway PCA feature

#### 7. Running Parallel Experiment:

Objective: To evaluate the model for different set of features and varying number of principal components. The script
trains and evaluates the model across different subset of features and number of principal components in parallel across
different cuda devices available in the system.

**PS**: You would get an assertion error if you run this code without Cuda, the code is specifically written to run in
parallel in GPU because the experiment is very time-consuming.

```shell
python parallel_experiments.py -i path_to_input_CSV_file.csv -o output_directory
```

The input data should have the following format:

- Index: Cosmic ID, Drug ID
- Drug Response: LN_IC50
- Gene Expression Features: EXP_pathway_1_name.PC1, EXP_pathway_1_name.PC2..., EXP_pathway_4_name.PC4
- CNV Features: CNV_pathway_1_name.PC1, CNV_pathway_1_name_pathway_1_name.PC2..., CNV_pathway_1_name_pathway_4_name.PC4
- Mutation Features: MUT_pathway_1_name.PC1, MUT_pathway_1_name_pathway_1_name.PC2...,
  MUT_pathway_1_name_pathway_4_name.PC4
- Morgan Fingerprint: bit_0, bit_1, bit_2, ...., bit_255
