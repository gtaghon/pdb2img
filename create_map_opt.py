### Create RGBA adjacency matrix maps from PDB models
"""
Geoffrey Taghon
Aug 1, 2024
NIST-MML
"""

import numpy as np
from Bio.PDB import Select, PDBIO, PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1 as three_to_one
from Bio.SeqUtils import ProtParam
from PIL import Image

class StandardAminoAcidSelect(Select):
    def accept_residue(self, residue):
        return is_aa(residue, standard=True) and all(atom.is_disordered() == 0 for atom in residue)
    
def clean_pdb(pdb_file):
    """
    Clean the PDB file by removing nonstandard residues, ions, cofactors, etc.
    Rebuild incomplete residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    
    # Remove hetero atoms and nonstandard residues
    io = PDBIO()
    io.set_structure(structure)
    clean_pdb_file = pdb_file.replace(".pdb", "_clean.pdb")
    io.save(clean_pdb_file, StandardAminoAcidSelect())
    
    # Rebuild incomplete residues
    structure = parser.get_structure("structure", clean_pdb_file)
    for model in structure:
        for chain in model:
            for residue in chain:
                if not all(atom in residue for atom in ['N', 'CA', 'C', 'O']):
                    rebuild_residue(residue)
    
    io.set_structure(structure)
    io.save(clean_pdb_file)
    
    return clean_pdb_file

def rebuild_residue(residue):
    """
    Rebuild incomplete residues by adding missing backbone atoms.
    """
    backbone_atoms = {'N': [1.32, 0, 0], 'CA': [0, 0, 0], 'C': [-1.32, 0, 0], 'O': [-2.24, 1, 0]}
    
    for atom_name, coords in backbone_atoms.items():
        if atom_name not in residue:
            new_atom = Atom(atom_name, coords, 0, 1, ' ', atom_name, None, element=atom_name[0])
            residue.add(new_atom)

def calculate_centroid(residue):
    """
    Calculate the centroid of a residue by averaging the coordinates of its atoms.
    Handle cases where the residue has no atoms.
    """
    coord_sum = np.zeros(3)
    atom_count = 0
    for atom in residue:
        coord_sum += atom.get_coord()
        atom_count += 1
    if atom_count == 0:
        return np.zeros(3)  # Return [0, 0, 0] if there are no atoms
    return coord_sum / atom_count

def calculate_biophysical_params(aa1, aa2):
    """
    Calculate biophysical parameters for a pair of amino acids using the Biopython ProtParam module.
    """
    pp = ProtParam.ProteinAnalysis(aa1 + aa2)
    
    hydrophobicity = pp.gravy()
    isoelectric_point = pp.isoelectric_point()
    mol_weight = pp.molecular_weight()
    aromaticity = pp.aromaticity()
    instability_index = pp.instability_index()
    
    return hydrophobicity, isoelectric_point, mol_weight, aromaticity, instability_index

def normalize_params(params, param_ranges):
    """
    Normalize biophysical parameters to the range [0, 255] based on the maximum and minimum possible values.
    Handle cases where max_val equals min_val.
    """
    normalized_params = []
    for param, param_range in zip(params, param_ranges):
        min_val, max_val = param_range
        if max_val == min_val:
            normalized_param = 0
        else:
            normalized_param = int(((param - min_val) / (max_val - min_val)) * 255)
        normalized_params.append(normalized_param)
    return normalized_params

def encode_rgba(normalized_params):
    """
    Encode normalized biophysical parameters into RGBA color values.
    """
    hydrophobicity, isoelectric_point, mol_weight, aromaticity, instability_index = normalized_params
    r = hydrophobicity
    g = isoelectric_point 
    b = mol_weight
    a = (aromaticity + instability_index) // 2
    return r, g, b, a

def create_adjacency_matrix(pdb_file, distance_threshold):
    """
    Create a protein residue adjacency matrix with biophysical parameters encoded as RGBA values.
    """
    #if not pdb_file.startswith("AF-"):
    #    cleaned_pdb_file = clean_pdb(pdb_file)
    #else:
    cleaned_pdb_file = pdb_file

    structure = PDBParser(QUIET=True).get_structure("structure", cleaned_pdb_file)
    residues = [res for res in structure.get_residues() if isinstance(res, Residue)]
    
    matrix_size = len(residues)
    matrix = np.zeros((matrix_size, matrix_size, 4), dtype=np.uint8)
    
    atom_list = [atom for res in residues for atom in res]
    neighbor_search = NeighborSearch(atom_list)
    
    # Precalculate biophysical parameters for all amino acid pairs
    aa_pairs = set([(three_to_one(res1.resname, undef_code="W"), three_to_one(res2.resname, undef_code="W")) 
                    for res1 in residues for res2 in residues if res1 != res2])
    aa_pair_params = {pair: calculate_biophysical_params(*pair) for pair in aa_pairs}
    
    # Collect all biophysical parameters
    all_params = list(aa_pair_params.values())
    
    # Calculate parameter ranges
    param_ranges = [(min(p), max(p)) for p in zip(*all_params)]
    
    # Normalize parameters and populate the matrix
    for i, res1 in enumerate(residues):
        centroid1 = calculate_centroid(res1)
        neighbors = neighbor_search.search(centroid1, distance_threshold)
        neighbor_residues = {atom.get_parent() for atom in neighbors}
        for j, res2 in enumerate(residues):
            if i >= j or res2 not in neighbor_residues:
                continue
            aa_pair = (three_to_one(res1.resname, undef_code="W"), three_to_one(res2.resname, undef_code="W"))
            params = aa_pair_params[aa_pair]
            normalized_params = normalize_params(params, param_ranges)
            matrix[i, j] = encode_rgba(normalized_params)
            matrix[j, i] = matrix[i, j]  # Make the matrix symmetric
    
    image = Image.fromarray(matrix, mode="RGBA")
    return image

def add_transparent_padding(matrix_image, target_size):
    """
    Add transparent padding around the matrix image to achieve the desired final dimensions.
    """
    matrix_width, matrix_height = matrix_image.size
    target_width, target_height = target_size

    # Calculate the padding dimensions
    left_pad = (target_width - matrix_width) // 2
    top_pad = (target_height - matrix_height) // 2

    # Create a new transparent image with the target size
    padded_image = Image.new("RGBA", target_size, (0, 0, 0, 0))

    # Paste the matrix image into the center of the padded image
    padded_image.paste(matrix_image, (left_pad, top_pad))

    return padded_image
