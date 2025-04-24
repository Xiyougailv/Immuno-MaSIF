import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

## 本程序用于：Decompose proteins into patches for input into the neural network.

## 通过sys导入自定义的模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source") ## For Linux

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

# Load training data (From many files)
from masif_modules.read_data_from_surface import read_data_from_surface, extract_patch, output_patch_feat, compute_shape_complementarity

print(sys.argv[2])

if len(sys.argv) <= 2:
    print("Usage: {config} "+sys.argv[0]+" PDBID_pmhc_tcr_label {masif_pmhc}")
    print("For example: 1AO7_AC_DE_A6.")
    ## argv[2]的格式：1AO7_AC
    sys.exit(1)

masif_app = sys.argv[2]

nf = sys.argv[3] ## 一共均匀抽取几张“脸”(frames/faces)
f = sys.argv[4] ## 目前处理第几张“脸”

if masif_app == 'masif_pmhc': 
    params = masif_opts['pmhc_fingerprint']

ppi_pair_list = [sys.argv[1]]

total_shapes = 0
total_ppi_pairs = 0
np.random.seed(0)
print('Reading data from input ply surface files.')
for ppi_pair_id in ppi_pair_list:

    all_list_desc = []
    all_list_coords = []
    all_list_shape_idx = []
    all_list_names = []
    idx_positives = []

    ## 这里需要修改（要确保创建新文件夹）
    my_precomp_dir = params['masif_precomputation_dir_md']+nf+'_'+f+'/'+ppi_pair_id+'/'
    if not os.path.exists(my_precomp_dir):
        os.makedirs(my_precomp_dir)
    
    # Read directly from the ply file.
    fields = ppi_pair_id.split('_')
    ply_file = {}
    ply_file['p1'] = masif_opts['ply_file_template'].format(fields[0], fields[1])

    '''
    if len (fields) == 2 or fields[2] == '':
        pids = ['p1'] 
    ''' ## SC

    if len (fields) == 4 or fields[4] == '':
        pids = ['p1']
    else:
        ply_file['p2']  = masif_opts['ply_file_template'].format(fields[0], fields[2])
        pids = ['p1', 'p2']
        
    # Compute shape complementarity between the two proteins. 
    rho = {}
    neigh_indices = {}
    mask = {}
    input_feat = {}
    theta = {}
    iface_labels = {}
    ## 补充定义，SC
    verts = {}
    faces = {}
    norms = {}

    for pid in pids:
        ## 为实现patch特征可视化，补充拷贝mesh
        ##input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)
        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid], faces[pid], norms[pid] = read_data_from_surface(ply_file[pid], params)

        ## 参数格式示例："data_preparation/01-benchmark_surfaces/1AO7_AC.ply"; masif_opts['ppi_search/site/ligand']
        ##================================================##
        ##         添加功能：Patch特征可视化 By 尚醇          ##
        ##================================================##
        ''' ## 尚醇，批量处理
        for i in [100,1000,1100,1200,1300,1400,1500,1700]:
            neigh_i = np.array(neigh_indices[pid][i]) ## 取得对应patch中的neigh索引
            feature_matrix = input_feat[pid]  ##  取得patch neigh的特征矩阵
            subv, subn, subf = extract_patch(verts[pid], faces[pid], norms[pid], neigh_i, i)
            output_patch_feat(subv, subf, subn, i, neigh_i, feature_matrix)
        '''




    
    if len(pids) > 1 and masif_app == 'masif_ppi_search':
        start_time = time.time()
        p1_sc_labels, p2_sc_labels = compute_shape_complementarity(ply_file['p1'], ply_file['p2'], neigh_indices['p1'],neigh_indices['p2'], rho['p1'], rho['p2'], mask['p1'], mask['p2'], params)
        np.save(my_precomp_dir+'p1_sc_labels', p1_sc_labels)
        np.save(my_precomp_dir+'p2_sc_labels', p2_sc_labels)
        end_time = time.time()
        print("Computing shape complementarity took {:.2f}".format(end_time-start_time))

    # Save data only if everything went well. 
    for pid in pids: 
        np.save(my_precomp_dir+pid+'_rho_wrt_center', rho[pid])
        np.save(my_precomp_dir+pid+'_theta_wrt_center', theta[pid])
        np.save(my_precomp_dir+pid+'_input_feat', input_feat[pid])
        np.save(my_precomp_dir+pid+'_mask', mask[pid])
        np.save(my_precomp_dir+pid+'_list_indices', neigh_indices[pid])
        ### 注意：80上上一行代码会报错：数组的形状不规则：
        ##  解决：降级了numpy版本：1.24.3 --> 1.21.6(!pip install numpy==1.21.6)，得到了warning
        np.save(my_precomp_dir+pid+'_iface_labels', iface_labels[pid])
        # Save x, y, z
        np.save(my_precomp_dir+pid+'_X.npy', verts[pid][:,0])
        np.save(my_precomp_dir+pid+'_Y.npy', verts[pid][:,1])
        np.save(my_precomp_dir+pid+'_Z.npy', verts[pid][:,2])




