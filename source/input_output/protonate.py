"""
protonate.py: Wrapper method for the reduce program: protonate (i.e., add hydrogens) a pdb using reduce 
                and save to an output file.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

from subprocess import Popen, PIPE
from IPython.core.debugger import set_trace
import os

''' ## original version, SC, 240714
def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()
'''


#########################
## 240714 chatgpt改写，加入中转文件，使用结束后清理
##       由于Docking结构的PDB文件并未质子化
#########################

def protonate(in_pdb_file, out_pdb_file):
    """
    使用 Reduce 工具对 PDB 文件进行质子化，并保存到输出文件中。
    
    :param in_pdb_file: 要质子化的 PDB 文件。
    :param out_pdb_file: 保存质子化后 PDB 文件的输出文件。
    """
    ## 打印初始文件和输出文件
    print(f"Input file: {in_pdb_file}")
    print(f"Output file: {out_pdb_file}")



    # 中转文件
    temp_pdb_file = "reduce_temp.pdb"
    
    # 第一步：移除质子（如果结构已经质子化）
    args = ["reduce", "-Trim", in_pdb_file]
    #p2 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    
    # 将修剪后的输出写入中转文件
    with open(temp_pdb_file, "w") as outfile:
        outfile.write(stdout.decode('utf-8').rstrip())
    
    # 第二步：添加质子
    args = ["reduce", "-HIS", temp_pdb_file] ## 参考「reduce -BUILD 选项」
    #p2 = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    
    # 将添加质子后的输出写入最终输出文件
    with open(out_pdb_file, "w") as outfile:
        outfile.write(stdout.decode('utf-8'))
    

    # 删除中转文件
    if os.path.exists(temp_pdb_file):
        os.remove(temp_pdb_file)


# 示例调用
# protonate("input.pdb", "output.pdb")

