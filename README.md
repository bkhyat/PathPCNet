# PathPCNet: Pathway Principal Component-Based Interpretable Framework for Drug Sensitivity Prediction
Bikhyat Adhikari, Masrur Sobhan, Ananda Sutradhar, Giri Narasimhan, Ananda Mohan Mondal
bioRxiv 2025.08.20.668802; doi: https://doi.org/10.1101/2025.08.20.668802

![End-to-end Pipeline for PathPCNet](images/workflow.png)

## Set up
Create Python 3.8 Virtual Environment and install the necessary package:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Running the evaluation on sample data
```shell
python run.py 
```

## Running Full Experiment
Follow the steps below to run full experiment on whole data.
### 1. Preprocessing Raw Data

Download the specified raw data files from the mentioned sources to a single folder, and provide the folder path as
in_path, and specify the file names for each of the below. The script will convert the data into specific matrix format,
apply transformation, and filter out common cell lines, pull SMILES

```shell
python process_raw_input_data.py --in_path data --pathway pathway.gmt --cell_line_list "Cell_listMon Jan 13 01_52_56 2025.csv"  --drugs drugs.csv  --rna rnaseq_all_data_20220624.csv  --cnv cnv_WES.csv  --mutation mutation.csv  --response drug_screening.csv  --out_path processed_data
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
