#!/usr/bin/python
import Bio
from Bio.PDB import * 
import sys
import importlib
import os

#### -----------------
##   本脚本：PDB下载和质子化 (PS. 由于需要下载PDB，本步骤在Mac Pro上完成：)
#### -----------------

## 通过sys导入自定义的default_config/input_output模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")

from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate


if len(sys.argv) <= 1: 
    '''
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    ''' ## SC
    print("Usage: "+sys.argv[0]+" PDBID_pmhc_tcr_label")

    sys.exit(1)

if not os.path.exists(masif_opts['raw_pdb_dir']):
    os.makedirs(masif_opts['raw_pdb_dir'])  ## 储存质子化后的PDB文件

if not os.path.exists(masif_opts['tmp_dir']):
    os.mkdir(masif_opts['tmp_dir'])

## 确认初始文件夹存在
if not os.path.exists(masif_opts['init_pdb_dir']):
    os.mkdir(masif_opts['init_pdb_dir'])  ## 储存直接下载得到的PDB文件

'''
in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]
''' ## SC
in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]


'''
# Download pdb 
pdbl = PDBList(server='http://ftp.wwpdb.org')
pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masif_opts['tmp_dir'],file_format='pdb')
## 下载到指定文件夹：pdir
''' ## SC
pdbl = PDBList(server='http://ftp.wwpdb.org')
pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masif_opts['init_pdb_dir'],file_format='pdb')
## pdb_filename = masif_opts['init_pdb_dir']+"/"+pdb_id+".pdb"


##### Protonate with reduce, if hydrogens included.
# - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

