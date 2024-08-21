### Python Bindings for PDC: Protein Data Compressor
"""
This work presents the highly compact PDC format to store the coordinate
and temperature factor information of protein structures in binary format.
A large-scale benchmark shows that delta encoding at lossless mode and combination
of Cartesian and torsion space representations at lossy mode enables more effective
compression.

The PDC format was originally designed to parse AlphaFold-predicted protein 
structure models.
To generalize PDC on other macromolecular structures in the future, several 
modifications will be needed, including the marking of missing atoms and addition 
of dedicated fields for heteroatoms.

Chengxin Zhang, Anna Marie Pyle (2023)
"PDC: a highly compact file format to store protein 3D coordinates."
Database: The Journal of Biological Databases and Curation. baad018.

Bindings by:
Geoffrey Taghon
June 7, 2024 // Update: July 18
NIST-MML
"""

import subprocess
import tempfile
import os

class PDCompressor:
    def __init__(self, pdc_path="pdc/pdc", pdd_path="pdc/pdd"):
        self.pdc_path = pdc_path
        self.pdd_path = pdd_path

    def compress_lossless(self, input_file, output_file):
        subprocess.run([self.pdc_path, input_file, output_file], check=True)

    def compress_lossy(self, input_file, output_file):
        subprocess.run([self.pdc_path, input_file, output_file, "-l=2"], check=True)

    def compress_lossless_ca_only(self, input_file, output_file):
        subprocess.run([self.pdc_path, input_file, output_file, "-l=3"], check=True)

    def compress_lossy_ca_only(self, input_file, output_file):
        subprocess.run([self.pdc_path, input_file, output_file, "-l=4"], check=True)

    def decompress(self, input_file, output_file):
        subprocess.run([self.pdd_path, input_file, output_file], check=True)

    def compress_lossless_to_string(self, input_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdc') as temp_file:
            temp_filename = temp_file.name

        try:
            # Compress the input file to a temporary file
            self.compress_lossless(input_file, temp_filename)

            # Read the temporary file as binary data
            with open(temp_filename, 'rb') as file:
                compressed_data = file.read()

            # Return the binary data as a string
            return compressed_data.decode('latin-1')

        finally:
            # Clean up the temporary file
            os.unlink(temp_filename)