# pdb2img
Informative 2D maps from 3D protein models.

## Installation
1. Clone this repository.
```
git clone https://github.com/gtaghon/pdb2img/
cd pdb2img
```
2. Install requirements. Use a virtual enviroment for cleanliness:
```
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

3. Build Protein Data Compressor (PDC) inside the pdb2img directory (Ensure that you have a C++ compiler installed):
```
cd pdc
make
```

## Usage
### Creating Datasets
#### Acquire a list of proteins from uniprot.org in csv format: 
1. Perform a search for your desired protein set.
2. Click 'Customize columns'. Ensure that at least columns 'Names & Taxonomy>Entry Name' and 'Gene Ontology (GO)>Gene Ontology IDs' are selected. These are required for image creation.
3. Download the results as excel or TSV. Open and export a copy as csv.
4. Place the csv file into a dataset folder, i.e. datasets/xx/

#### Run pipeline-eff (efficient)
Specify input csv file and output directory to save downloaded pdbs and their created images.
```
python pipeline-eff.py datasets/xx/input.csv datasets/xx/images
```
The pipeline will use all available cores to download all existing alphafold models based on the list of entries in the csv, and convert them to png images. The pdbs will be saved to the images directory. You may remove them manually if they are not needed.

### Training a classifier (ResNet-50)
The following command will train ResNet-50 for multilabel classification of the png images. The labels are the top 100 (--max-classes) most frequently observed GO annotations provided in the csv file (--data-path). The input dataset is automatically partitioned into randomly shuffled 70/15/15 train/validation/test sets. The classifier will report live validation loss as well as validation Label Ranking Average Precision (LRAP) score.
```
accelerate launch classifier-distributed.py --epochs 150 --lr 0.001 --max-classes 100 --data-path datasets/xx/file.csv --img-dir datasets/xx/images
```
