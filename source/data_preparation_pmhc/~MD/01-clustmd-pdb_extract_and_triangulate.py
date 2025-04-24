#! /usr/bin/env python3
#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

## 通过sys导入自定义的模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")
# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS

from triangulation.fixmesh import fix_mesh
'''
import pymesh
''' ## SC
## 导入可作为.egg文件使用的pymesh模块
pymesh_egg_path='/usr/local/lib/python3.8/dist-packages/pymesh2-0.3-py3.8-linux-x86_64.egg'
sys.path.append(pymesh_egg_path)
import pymesh

from input_output.extractPDB import extractPDB

from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh

from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree

## 忽略警告
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) <= 1: 
    '''
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    ''' ## SC
    print("Usage: {config} "+sys.argv[0]+" PDBID_AC")
    print("AC are the chains to include in pMHC surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1] ## pMHC chains:AC

pdb_info = sys.argv[1] ## PDBID_pmhc_tcr_label
nf = sys.argv[2] ## 一共均匀抽取几张“脸”(frames/faces)
f = sys.argv[3] ## 目前处理第几张“脸”

'''
if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
    pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
else:
    pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
''' ## SC

### pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
pdb_filename = masif_opts['raw_pdb_dir']+"Clust_MD/"+pdb_info+"/"+pdb_id+"_"+nf+"_"+f+".pdb"  ## 原始PDB位置

## 注意：中间步骤的文件（即存放在临时文件夹/tmp下的文件），尽量不改动其名称（以免对调用函数造成影响）
## 因此，由于处理过程是单线程的，我们对同一结构的不同帧在文件名上不做区分 （每次会覆盖掉重写）

## 为什么又质子化一遍？
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb" ## 质子化PDB位置
### protonated_file = tmp_dir+"/"+pdb_id+"_"+nf+"_"+f+".pdb" ## 质子化PDB位置
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file


# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1 ## 只存pMHC的PDB位置
### out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1+"_"+nf+"_"+f  ## 只存pMHC的PDB位置
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

# Compute MSMS of surface w/hydrogens, 
try:
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb",\
        protonate=True)
except:
    set_trace()




# Compute "charged" vertices
if masif_opts['use_hbond']:
    vertex_hbond = computeCharges(out_filename1, vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid. 
if masif_opts['use_hphob']:
    vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

# Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)
## SC, For test  print(regular_mesh.vertices,out_filename1+".pdb", out_filename1)


if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

iface = np.zeros(len(regular_mesh.vertices))
if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS(pdb_filename,\
        protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface)

else:
    # Convert to ply and save.
    save_ply(out_filename1+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
## 注意：out_filename1为绝对路径 ## 这里都没有变
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 


