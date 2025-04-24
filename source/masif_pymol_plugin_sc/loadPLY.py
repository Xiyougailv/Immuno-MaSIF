from pymol import cmd, stored
import sys
import os, math, re
from pymol.cgo import *
import os.path
import numpy as np
"""
   loadPLY.py: This pymol function loads ply files into pymol.
    Pablo Gainza - LPDI STI EPFL 2016-2019
    This file is part of MaSIF.
    Released under an Apache License 2.0
"""
colorDict = {
    "sky": [COLOR, 0.0, 0.76, 1.0],
    "sea": [COLOR, 0.0, 0.90, 0.5],
    "yellowtint": [COLOR, 0.88, 0.97, 0.02],
    "hotpink": [COLOR, 0.90, 0.40, 0.70],
    "greentint": [COLOR, 0.50, 0.90, 0.40],
    "blue": [COLOR, 0.0, 0.0, 1.0],
    "green": [COLOR, 0.0, 1.0, 0.0],
    "yellow": [COLOR, 1.0, 1.0, 0.0],
    "orange": [COLOR, 1.0, 0.5, 0.0],
    "red": [COLOR, 1.0, 0.0, 0.0],
    "black": [COLOR, 0.0, 0.0, 0.0],
    "white": [COLOR, 1.0, 1.0, 1.0],
    "gray": [COLOR, 0.9, 0.9, 0.9],
    "greymesh": [COLOR, 0.5, 0.5, 0.5],
    "greyvert": [COLOR, 0.7, 0.7, 0.7],
    "dotcyan": [COLOR, 0.5, 1.0, 1.0],
    "edgeteal": [COLOR, 0, 0.75, 0.75],
}
### 加了greymesh和greyvert，把颜色变深^_^
### 24/01/02 加了dotcyan和edgeteal颜色，使vert-pool更加明显

# Create a gradient color from color 1 to whitish(白色), to color 2. val goes from 0 (color1) to 1 (color2).
#### 这个函数好像不能用 :|
def color_gradient(vals, color1, color2):
    c1 = Color("white")
    c2 = Color("orange")
    ix = np.floor(vals * 100).astype(int) ## np.floor：向下取整
    crange = list(c1.range_to(c2, 100))
    mycolor = []
    print(crange[0].get_rgb())
    for x in ix:
        myc = crange[x].get_rgb()
        mycolor.append([COLOR, myc[0], myc[1], myc[2]])
    return mycolor



def iface_color(iface):
    # max value is 1, min values is 0
    hp = iface.copy()
    hp = hp * 2 - 1
    mycolor = charge_color(-hp)
    return mycolor


# Returns the color of each vertex according to the charge.
# The most purple colors are the most hydrophilic values, and the most
# white colors are the most positive colors.
def hphob_color(hphob):
    # max value is 4.5(ILE), min values is -4.5(ARG)
    hp = hphob.copy()
    # normalize
    hp = hp + 4.5
    hp = hp / 9.0
    # mycolor = [ [COLOR, 1.0, hp[i], 1.0]  for i in range(len(hp)) ]
    mycolor = [[COLOR, 1.0, 1.0 - hp[i], 1.0] for i in range(len(hp))]
    ''' SC, 25 Apr
    因为红和蓝通道都恒定为1.0，而绿色随着 hp[i] 增大（越疏水）而减少，
    颜色会从 白色（R=G=B=1，对应最亲水）过渡到紫红（R=B=1, G=0，对应最疏水）

    白色 → 代表亲水
    亮粉色/浅紫色 → 中间性质
    品红/紫红 → 代表疏水
    '''
    return mycolor


# Returns the color of each vertex according to the charge.
# The most red colors are the most negative values, and the most
# blue colors are the most positive colors.
def charge_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    blue_charges = np.array(norm_charges)
    red_charges = np.array(norm_charges)
    blue_charges[blue_charges < 0] = 0 ## 蓝色正电
    red_charges[red_charges > 0] = 0 ## 红色负电
    red_charges = abs(red_charges)
    red_charges[red_charges > max_val] = max_val ## 负电从-1处削平
    #blue_charges[blue_charges < min_val] = min_val ## 这句在干嘛 删掉了？？？
    blue_charges[blue_charges > max_val] = max_val ## 我加上的，正电从1处削平
    red_charges = red_charges / max_val
    blue_charges = blue_charges / max_val
    # red_charges[red_charges>1.0] = 1.0
    # blue_charges[blue_charges>1.0] = 1.0
    green_color = np.array([0.0] * len(charges))
    mycolor = [
        [
            COLOR,
            0.9999 - blue_charges[i],
            0.9999 - (blue_charges[i] + red_charges[i]),
            0.9999 - red_charges[i],
        ]
        for i in range(len(charges))
    ]
    for i in range(len(mycolor)):
        for k in range(1, 4):
            if mycolor[i][k] < 0:
                mycolor[i][k] = 0

    return mycolor
'''
#################################################
######    定义红绿着色函数 For 几何特征   By SC ######
######    green: >0; red: <0                #####
#################################################
def geometrical_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    green_charges = np.array(norm_charges)
    red_charges = np.array(norm_charges)
    green_charges[green_charges < 0] = 0
    red_charges[red_charges > 0] = 0
    red_charges = abs(red_charges)
    red_charges[red_charges > max_val] = max_val
    green_charges[green_charges < min_val] = min_val
    red_charges = red_charges / max_val
    green_charges = green_charges / max_val
    # red_charges[red_charges>1.0] = 1.0
    # blue_charges[blue_charges>1.0] = 1.0
    blue_color = np.array([0.0] * len(charges))
    mycolor = [
        [
            COLOR,
            0.9999 - green_charges[i],
            0.9999 - red_charges[i],
            0.9999 - (green_charges[i] + red_charges[i]),
        ]
        for i in range(len(charges))
    ]
    for i in range(len(mycolor)):
        for k in range(1, 4):
            if mycolor[i][k] < 0:
                mycolor[i][k] = 0

    return mycolor
''' ## 不用红绿色

#################################################
######    定义cyan-magenta函数 For 几何特征         By SC ######
######    magenta(RGB: 1.0, 0.0, 1.0): >0; 
######    white  (RGB: 1.0, 1.0, 1.0): =0; 
######    cyan   (RGB: 0.0, 1.0, 1.0): <0 
#################################################
def geometrical_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    magenta_charges = np.array(norm_charges)
    cyan_charges = np.array(norm_charges)
    magenta_charges[magenta_charges < 0] = 0 ## 品红凸起
    magenta_charges[magenta_charges > max_val] = max_val ## 我加上的，正电从1处削平
    cyan_charges[cyan_charges > 0] = 0 ## 青色凹下
    cyan_charges = abs(cyan_charges)
    cyan_charges[cyan_charges > max_val] = max_val ## 负电从-1处削平
    mycolor = [
        [
            COLOR,
            1.0 - cyan_charges[i],
            1.0 - magenta_charges[i],
            1.0,
        ]
        for i in range(len(charges))
    ]
    return mycolor



def load_ply(
    filename, color="white", name="ply", dotSize=0.2, lineSize=2, doStatistics=False
):
    ## Pymesh should be faster and supports binary ply files. However it is difficult to install with pymol...
    #        import pymesh
    #        mesh = pymesh.load_mesh(filename)
    ## 我把lineSize 从0.5 --> 1.0
    ##   dotSize 从0.2 --> 0.3


    '''
    from simple_mesh import Simple_mesh
    ''' ## SC
    from .simple_mesh import Simple_mesh

    mesh = Simple_mesh()
    mesh.load_mesh(filename)

    ignore_normal = False
    with_normal = False
    with_color = False

    group_names = ""

    verts = mesh.vertices
    try:
        charge = mesh.get_attribute("vertex_charge")
        color_array = charge_color(charge)
    except:
        print("Could not load vertex charges.")
        color_array = [colorDict["green"]] * len(verts)
    if "vertex_nx" in mesh.get_attribute_names():
        nx = mesh.get_attribute("vertex_nx")
        ny = mesh.get_attribute("vertex_ny")
        nz = mesh.get_attribute("vertex_nz")
        normals = np.vstack([nx, ny, nz]).T
        print(normals.shape)

    # Draw vertices
    ########### 把verts按照charge特性着色：vert_
    obj = []
    color = "green"

    for v_ix in range(len(verts)):
        vert = verts[v_ix]
        ## colorToAdd = color_array[v_ix] ## vertice不显示带电情况
        colorToAdd = colorDict["gray"]  ## 24/03/06
        # Vertices
        obj.extend(colorToAdd)
        obj.extend([SPHERE, vert[0], vert[1], vert[2], dotSize])

    name = "vert_" + filename
    group_names = name
    cmd.load_cgo(obj, name, 1.0)
    obj = []

    faces = mesh.faces

    # Draw surface charges.
    ######### 画静电表面（但不明白颜色过渡是如何做到的）：pb_
    if (
        "vertex_charge" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        color_array_surf = color_array
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "pb_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []


    ##################################################
    ###  FOR溯源 添加可视化“特征”：poolvert的标签关联程度
    ###          SC   2023/10/25
    ##################################################
    if (
        "vertex_impot" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        impot = mesh.get_attribute("vertex_impot")
        norm_impot = impot
        importance = np.array(impot)
        ### 为了适应charge color方式：将[0,1]-->[0,2]-->[-1,1]-->[1,-1]
        ### 处理后，红色：重要程度高；蓝色：重要程度低
        for i, iv in enumerate(importance):
            importance[i] *= 2
            importance[i] -= 1
            importance[i] *= -1
        color_array_surf = charge_color(importance)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "impo_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []






    # Draw hydrophobicity
     ######### 画亲疏水表面：hphobic_
    if (
        "vertex_hphob" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        hphob = mesh.get_attribute("vertex_hphob")
        color_array_surf = hphob_color(hphob)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "hphobic_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    # Draw shape index
    ######### 画形状指数表面：si_
    if (
        "vertex_si" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        si = mesh.get_attribute("vertex_si")
        color_array_surf = geometrical_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []
    '''
    # Draw shape index
    if (
        "vertex_si" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        si = mesh.get_attribute("vertex_si")
        color_array_surf = geometrical_color(si)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "si_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []

    obj = []
    ''' ## 写重复了


    # Draw ddc
    ######### 画距离相关曲率表面：ddc_
    if (
        "vertex_ddc" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        ddc = mesh.get_attribute("vertex_ddc")
        # Scale to -1.0->1.0
        ddc = ddc * 1.4285
        color_array_surf = geometrical_color(ddc)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "ddc_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []

    # Draw iface
    if (
        "vertex_iface" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        iface = mesh.get_attribute("vertex_iface")
        color_array_surf = iface_color(iface)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.5])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "iface_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    obj = []

    # Draw hbond
    ######### 氢键供受体表面：hbond_
    if (
        "vertex_hbond" in mesh.get_attribute_names()
        and "vertex_nx" in mesh.get_attribute_names()
    ):
        hbond = mesh.get_attribute("vertex_hbond")
        color_array_surf = charge_color(hbond)
        for tri in faces:
            vert1 = verts[int(tri[0])]
            vert2 = verts[int(tri[1])]
            vert3 = verts[int(tri[2])]
            na = normals[int(tri[0])]
            nb = normals[int(tri[1])]
            nc = normals[int(tri[2])]
            obj.extend([BEGIN, TRIANGLES])
            # obj.extend([ALPHA, 0.6])
            obj.extend(color_array_surf[int(tri[0])])
            obj.extend([NORMAL, (na[0]), (na[1]), (na[2])])
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend(color_array_surf[int(tri[1])])
            obj.extend([NORMAL, (nb[0]), (nb[1]), (nb[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.extend(color_array_surf[int(tri[2])])
            obj.extend([NORMAL, (nc[0]), (nc[1]), (nc[2])])
            obj.extend([VERTEX, (vert3[0]), (vert3[1]), (vert3[2])])
            obj.append(END)
        name = "hbond_" + filename
        cmd.load_cgo(obj, name, 1.0)
        obj = []
        group_names = group_names + " " + name

    # Draw triangles (faces)
    ############ 画三角形（mesh edges）：mesh_
    for tri in faces:
        pairs = [[tri[0], tri[1]], [tri[0], tri[2]], [tri[1], tri[2]]]
        colorToAdd = colorDict["greyvert"] ## 把颜色调深了 ## 24/03/06
        for pair in pairs:
            vert1 = verts[pair[0]]
            vert2 = verts[pair[1]]
            obj.extend([LINEWIDTH, 2]) ## 24/03/06
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
    name = "mesh_" + filename
    cmd.load_cgo(obj, name, 1.0)
    group_names = group_names + " " + name

    # Draw normals
    if with_normal and not ignore_normal:
        for v_ix in range(len(verts)):
            colorToAdd = colorDict["white"]
            vert1 = verts[v_ix]
            vert2 = [
                verts[v_ix][0] + nx[v_ix],
                verts[v_ix][1] + ny[v_ix],
                verts[v_ix][2] + nz[v_ix],
            ]
            obj.extend([LINEWIDTH, 2.0])
            obj.extend([BEGIN, LINES])
            obj.extend(colorToAdd)
            obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
            obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
            obj.append(END)
        cmd.load_cgo(obj, "normal_" + filename, 1.0)

    print(group_names)
    cmd.group(filename, group_names)


# Load the sillouete of an iface.
def load_giface(filename, color="white", name="giface", dotSize=0.2, lineSize=1.0):
    mesh = pymesh.load_mesh(filename)
    if "vertex_iface" not in mesh.get_attribute_names():
        return
    iface = mesh.get_attribute("vertex_iface")
    # Color an edge only if:
    # iface > 0 for its two edges
    # iface is zero for at least one of its edges.
    # Go through each face.
    faces = mesh.faces
    verts = mesh.vertices
    obj = []
    visited = set()
    colorToAdd = colorDict["green"]
    obj.extend([BEGIN, LINES])
    obj.extend([LINEWIDTH, 5.0])
    obj.extend(colorToAdd)
    for tri in faces:
        pairs = [
            [tri[0], tri[1], tri[2]],
            [tri[0], tri[2], tri[1]],
            [tri[1], tri[2], tri[0]],
        ]
        for pair in pairs:
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([VERTEX, (vert1[0]), (vert1[1]), (vert1[2])])
                obj.extend([VERTEX, (vert2[0]), (vert2[1]), (vert2[2])])
    obj.append(END)
    name = "giface_" + filename
    cmd.load_cgo(obj, name, 1.0)
    colorToAdd = colorDict["green"]

    obj = []
    obj.extend(colorToAdd)
    for tri in faces:
        pairs = [
            [tri[0], tri[1], tri[2]],
            [tri[0], tri[2], tri[1]],
            [tri[1], tri[2], tri[0]],
        ]
        for pair in pairs:
            if iface[pair[0]] > 0 and iface[pair[1]] > 0 and iface[pair[2]] == 0:
                vert1 = verts[pair[0]]
                vert2 = verts[pair[1]]

                obj.extend([SPHERE, (vert1[0]), (vert1[1]), (vert1[2]), 0.4])
                obj.extend([SPHERE, (vert2[0]), (vert2[1]), (vert2[2]), 0.4])
    # obj.append(END)
    name = "giface_verts_" + filename
    cmd.load_cgo(obj, name, 1.0)

