# PathPCNet

The code base for PathPCNet. From Pathways to Principal Components: An Interpretable Framework for Drug Sensitivity
Prediction
Authors: Bikhyat Adhikari, Ananda Mohan Mondal

#### Running Parallel Experiment:

Objective: To evaluate the model for different set of features and varying number of principal components. The script
trains and evaluates the model across different subset of features and number of principal components in parallel across
different cuda devices available in the system.
**PS**: You would get an assertion error if you run this code without Cuda, the code is specifically written to run in
parallel in GPU because the experiment is very time-consuming.

```python
python
parallel_experiments.py - i
path_to_input_CSV_file.csv - o
output_directory
```

The input data should have the following format:

- Index: Cosmic ID, Drug ID
- Drug Response: LN_IC50
- Gene Expression Features: EXP_pathway_1_name.PC1, EXP_pathway_1_name.PC2..., EXP_pathway_4_name.PC4
- CNV Features: CNV_pathway_1_name.PC1, CNV_pathway_1_name_pathway_1_name.PC2..., CNV_pathway_1_name_pathway_4_name.PC4
- Mutation Features: MUT_pathway_1_name.PC1, MUT_pathway_1_name_pathway_1_name.PC2...,
  MUT_pathway_1_name_pathway_4_name.PC4
- Morgan Fingerprint: bit_0, bit_1, bit_2, ...., bit_255
