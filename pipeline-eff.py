import os
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image
from Bio import PDB
from create_map_opt import create_adjacency_matrix, add_transparent_padding
from rdh_rgba import reversible_data_hiding_rgba
from pdc import PDCompressor
import csv
import logging
from tqdm import tqdm
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_alphafold_pdb(uniprot_id, output_dir):
    base_url = "https://alphafold.ebi.ac.uk/files/"
    pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
    pdb_url = f"{base_url}{pdb_filename}"
    output_path = os.path.join(output_dir, pdb_filename)

    if os.path.exists(output_path):
        #logging.info(f"Skipping existing AlphaFold PDB: {pdb_filename}")
        return output_path

    try:
        response = requests.get(pdb_url)
        response.raise_for_status()

        with open(output_path, 'wb') as pdb_file:
            pdb_file.write(response.content)

        #logging.info(f"Successfully downloaded AlphaFold PDB: {pdb_filename}")
        return output_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download AlphaFold PDB {pdb_filename}: {str(e)}")
        return None

def process_single_protein(args):
    uniprot_id, output_dir = args
    try:
        #logging.info(f"Processing {uniprot_id}")

        # Step 1: Download PDB file if not exists
        pdb_file = download_alphafold_pdb(uniprot_id, output_dir)
        if not pdb_file:
            return None

        # Step 2: Process PDB file
        parser = PDB.PDBParser(QUIET=True)
        modelname = Path(pdb_file).stem
        img_size = 128

        # Get length of model
        try:
            pdb = parser.get_structure("input", pdb_file)
            reslen = sum(len([_ for _ in chain.get_residues() if PDB.is_aa(_)]) for chain in pdb.get_chains())
            distance_threshold = np.sqrt(reslen)
        except Exception as e:
            logging.error(f"Error parsing PDB file {pdb_file}: {str(e)}")
            return None

        # Adjust image size if necessary
        if reslen > img_size:
            img_size = reslen

        # Create adjacency matrix
        try:
            adjacency_matrix = create_adjacency_matrix(pdb_file, distance_threshold)
            adjacency_matrix = add_transparent_padding(adjacency_matrix, (img_size, img_size))
        except Exception as e:
            logging.error(f"Error creating adjacency matrix for {pdb_file}: {str(e)}")
            return None

        # Compress PDB
        try:
            compressor = PDCompressor()
            pdc_data = compressor.compress_lossless_to_string(pdb_file)
        except Exception as e:
            logging.error(f"Error compressing PDB file {pdb_file}: {str(e)}")
            return None

        # Hide compressed PDB in adjacency matrix
        try:
            marked_image = reversible_data_hiding_rgba(pdc_data, adjacency_matrix)
        except Exception as e:
            logging.error(f"Error hiding data in image for {pdb_file}: {str(e)}")
            return None
        
        # Save the final output
        output_path = os.path.join(output_dir, f"{modelname}_matrix_with_pdc.png")
        try:
            marked_image.save(output_path)
            #logging.info(f"Successfully processed {pdb_file}")
            return output_path
        except Exception as e:
            logging.error(f"Error saving output image for {pdb_file}: {str(e)}")
            return None

    except Exception as e:
        logging.error(f"Unexpected error processing {uniprot_id}: {str(e)}")
        return None

def main(input_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read UniProt IDs from CSV
    uniprot_ids = []
    with open(input_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            uniprot_id = row.get('Entry')
            if uniprot_id:
                uniprot_ids.append(uniprot_id)
            else:
                logging.warning(f"Missing 'Entry' in row: {row}")

    # Set up multiprocessing
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Prepare arguments for multiprocessing
    process_args = [(id, output_dir) for id in uniprot_ids]

    # Process proteins in parallel with progress bar
    results = list(tqdm(pool.imap(process_single_protein, process_args), total=len(uniprot_ids), desc="Processing proteins"))

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Print results
    successful = [r for r in results if r is not None]
    logging.info(f"Successfully processed {len(successful)} out of {len(uniprot_ids)} proteins")
    logging.info("Processing complete")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <input_csv_file> <output_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_file, output_dir)