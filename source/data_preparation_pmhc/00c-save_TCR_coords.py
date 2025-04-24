import os
import numpy as np
import sys
from Bio.PDB import *

#### ---------------------------------------
##   本脚本：储存TCR（所有原子的 ##可按需改动）坐标&类型：
##         后面需要依据TCR-cutoff筛选可训练的pMHC-patch
#### ---------------------------------------

## from SBI.structure import PDB
## 通过sys导入自定义的default_config/input_output模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")
from default_config.masif_opts import masif_opts

in_fields = sys.argv[1].split("_") ## PDBID_pmhc_tcr_label
pdb_id = in_fields[0]
tcr_chain = in_fields[2]
tcr_type = in_fields[3]

if not os.path.exists(masif_opts["pmhc_fingerprint"]["tcr_coords_dir"]):
    os.mkdir(masif_opts["pmhc_fingerprint"]["tcr_coords_dir"])

'''
# Ligands of interest
ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]

structure_ligands_type = []
structure_ligands_coords = []
try:
    structure = PDB(
        os.path.join(masif_opts["ligand"]["assembly_dir"], "{}.pdb".format(pdb_id))
    )
except:
    print("Problem with opening structure", pdb)
for chain in structure.chains:
    for het in chain.heteroatoms:
        # Check all ligands in structure and save coordinates if they are of interest
        if het.type in ligands:
            structure_ligands_type.append(het.type)
            structure_ligands_coords.append(het.all_coordinates)
'''

structure_tcr_type = []
structure_tcr_coords = []
parser = PDBParser()
pdb_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
struct = parser.get_structure(pdb_id, pdb_file)
## 拆分TCR双链
tcr = list(tcr_chain)
tcr_a = tcr[0]
tcr_b = tcr[1]
### 选出TCR上的所有原子，并存储其坐标
for atom in struct.get_atoms():
    residue = atom.get_parent()
    chain = residue.get_parent()
    if chain.get_id() in tcr:
        coords = "{:.06f} {:.06f} {:.06f}".format(
            atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            ) ## 需要这么高的精度吗？？
        structure_tcr_coords.append(coords)

### 存储TCR类型
structure_tcr_type.append(tcr_type)

'''
np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_types.npy".format(pdb_id)
    ),
    structure_ligands_type,
)
np.save(
    os.path.join(
        masif_opts["ligand"]["ligand_coords_dir"], "{}_ligand_coords.npy".format(pdb_id)
    ),
    structure_ligands_coords,
)
'''

np.save(
    os.path.join(
        masif_opts["pmhc_fingerprint"]["tcr_coords_dir"], "{}_tcr_type.npy".format(pdb_id)
    ),
    structure_tcr_type,
)
np.save(
    os.path.join(
        masif_opts["pmhc_fingerprint"]["tcr_coords_dir"], "{}_tcr_coords.npy".format(pdb_id)
    ),
    structure_tcr_coords,
)


