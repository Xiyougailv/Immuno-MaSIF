import sys
import time
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)

####################################
##  本程序用于： 数据后处理
##      MaSIF_pMHC解释性：Trace patches on surface（在pMHC表面上对verts的重要程度可视化）
##      本脚本：产生.ply文件（与原04sc文件地位相当）
###   SC，2023/10/24

## 更新， 24.Dec
####################################


## 通过sys导入自定义的模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source") ## For Linux

# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0) 

# Load training data (From many files)
## 溯源：从新编写的脚本中引入功能函数
from masif_modules.read_pool_from_surface import read_pool_from_surface, extract_pool, output_pool_feat

print(sys.argv[2]) ## masif_pmhc


if len(sys.argv) <= 2:
    print("Usage: {config} "+sys.argv[0]+" PDBID_pmhc_tcr_label {masif_pmhc}")
    print("For example: 1AO7_AC_DE_A6.")
    ## argv[2]的格式：1AO7_AC
    sys.exit(1)

masif_app = sys.argv[2] 

if masif_app == 'masif_pmhc': 
    params = masif_opts['pmhc_fingerprint']

## 溯源：csv来自：结果文件夹
test_set_out_dir = params["predict_trace_40-7_pepcut4_all-test"] ## 24.Dec
## 溯源：保存pool.ply文件：确保创建ply文件夹
ply_set_out_dir = params["ply_trace_40-7_pepcut4_all-test"] ## 0305修改
if not os.path.exists(ply_set_out_dir):
    os.makedirs(ply_set_out_dir)


ppi_pair_list = [sys.argv[1]] ## 格式：1AO7_AC_DE_A6

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

    my_precomp_dir = params['masif_precomputation_dir']+ppi_pair_id+'/'
    if not os.path.exists(my_precomp_dir):
        os.makedirs(my_precomp_dir)
    
    # Read directly from the ply file.
    fields = ppi_pair_id.split('_')
    ply_file = {}
    ply_file['p1'] = masif_opts['ply_file_template'].format(fields[0], fields[1]) ## 我们使用的：1AO7_AC.ply
    print(ply_file['p1'])
    #######     （'p1'代表我们只处理pMHC表面即chainAC，未处理TCR一边）

    ## 溯源
    csv_file = {}
    csv_file['p1'] = test_set_out_dir + "/pool_{}.csv".format(fields[0])  ## 本标签：pool_5C07.csv
    #csv_file['p1'] = test_set_out_dir + "/m2pred_pool_{}.csv".format(fields[0])  ## 10.30 非本标签：maxpred_pool_5C07.csv
    ### 添加权重，23/10/27
    weigh_csv_file = {}
    weigh_csv_file['p1'] = test_set_out_dir + "/weigh_pool_{}.csv".format(fields[0])  ## 本标签：weigh_pool_5C07.csv
    #weigh_csv_file['p1'] = test_set_out_dir + "/m2pred_weigh_pool_{}.csv".format(fields[0])  ## 10.30 非本标签：maxpred_weigh_pool_5C07.csv

    out_ply_file = {}
    out_ply_file['p1'] = ply_set_out_dir + "/pool_{}.ply".format(fields[0]) ## pool_5C07.ply
    #out_ply_file['p1'] = ply_set_out_dir + "/m2pred_pool_{}.ply".format(fields[0]) ## 10.30 非本标签：maxpred_pool_5C07.ply
    ### 添加权重，23/10/27
    weigh_out_ply_file = {}
    weigh_out_ply_file['p1'] = ply_set_out_dir + "/weigh_pool_{}.ply".format(fields[0]) ## pool_5C07.ply
    #weigh_out_ply_file['p1'] = ply_set_out_dir + "/m2pred_weigh_pool_{}.ply".format(fields[0]) ## 10.30 非本标签：maxpred_weigh_pool_5C07.ply

    '''
    if len (fields) == 2 or fields[2] == '':
        pids = ['p1'] 
    ''' ## SC

    if len (fields) == 4 or fields[4] == '':
        pids = ['p1'] ## 我们只考虑互作的pMHC一侧
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
    ## 溯源
    pool_index = {}

    for pid in pids:
        ## 为实现patch特征可视化，补充拷贝mesh
        ##input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)
        input_feat[pid], pool_index[pid], verts[pid], faces[pid], norms[pid] = read_pool_from_surface(ply_file[pid], params, csv_file[pid])
        ## 参数格式示例："data_preparation/01-benchmark_surfaces/1AO7_AC.ply"; masif_opts['ppi_search/site/ligand']
        
        ##================================================##
        ##         添加功能：Patch溯源 By 尚醇          ##
        ##                 2023/10/24
        ##================================================##
        neigh_i = pool_index[pid] ## 取得pool vert索引
        feature_matrix = input_feat[pid]
        print('------- WITHOUT weigh -------')
        print('Extracting pool sub-surf.')
        subv, subn, subf = extract_pool(verts[pid], faces[pid], norms[pid], neigh_i)
        print('Storing feature-riched pool.ply.')
        output_pool_feat(subv, subf, subn, neigh_i, feature_matrix, out_ply_file[pid])
        ## np.save(my_precomp_dir+pid+'_X_randomtest.npy', verts[pid][:,0])

        #######################################
        ####  添加权重：尚醇 2023/10/27
        ###################
        input_feat[pid], pool_index[pid], verts[pid], faces[pid], norms[pid] = read_pool_from_surface(ply_file[pid], params, weigh_csv_file[pid])
        neigh_i = pool_index[pid] ## 取得pool vert索引
        feature_matrix = input_feat[pid]
        print('--------- WITH weigh --------')
        print('Extracting weigh_pool sub-surf.')
        subv, subn, subf = extract_pool(verts[pid], faces[pid], norms[pid], neigh_i)
        print('Storing feature-riched weigh_pool.ply.')
        output_pool_feat(subv, subf, subn, neigh_i, feature_matrix, weigh_out_ply_file[pid])




