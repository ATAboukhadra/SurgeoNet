import numpy as np
import pymeshlab
import os
import torch
import pandas as pd

df = pd.read_csv('data/kps.csv')

def write_obj(verts, faces, filename, pred=True):
    """Saves and obj file using vertices and faces"""
    # if 'gt' in filename:
    texture = np.zeros_like(verts)
    if pred:
        texture[:, 0] = 0
        texture[:, 1] = 0
        texture[:, 2] = 1
    else:
        texture[:, 0] = 0.7
        texture[:, 1] = 0.7
        texture[:, 2] = 0.7

    if texture is not None:
        alpha = np.ones((verts.shape[0], 1))
        v_color_matrix = np.append(texture, alpha, axis=1)
        m = pymeshlab.Mesh(verts, faces, v_color_matrix=v_color_matrix)
    else:
        m = pymeshlab.Mesh(verts, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, f'{filename}')
    ms.save_current_mesh(f'{filename}.obj', save_vertex_normal=True, save_vertex_color=True, save_polygonal=True)


def find_delimiter(path):
    with open(path) as f:
        lines = f.readlines()
        c = 0
        for i, line in enumerate(lines):
            component = line.split(' ')[0]
            if component == 'v':
                c += 1
            elif component == 'o' and ('Right' in line or 'right' in line or '001' in line): #('right' in line or 'bolt' in line or 'Bolt' in line or 'Right' in line):
                break
        delimiter = c
    return delimiter

def load_kps_sheet(name, nKps):
    # first row as column names
    row = df[df['Object'] == name]
    # all values except the first one
    kps = list(row.values[0][1:])
    
    return kps

class SurgeryTool():

    def __init__(self, verts, faces, xArticulatedList, yArticulatedList, name, nKps) -> None:
        self.verts = verts.unsqueeze(0)
        self.faces = faces.unsqueeze(0)
        self.isXArticulated = name in xArticulatedList
        self.isYArticulated = name in yArticulatedList
        self.delimiter = verts.shape[0] if (name not in xArticulatedList and name not in yArticulatedList) else find_delimiter(os.path.join('objects', name))
        self.bbox3d = self.find_bbox3d()
        if nKps > 12:
            _, indices = farthest_point_down_sample(verts, nKps)
        else:
            indices = load_kps_sheet(name[:-4], nKps)
        self.sampledPoints = indices
    
    def find_bbox3d(self):
        corners = ['000', '001', '010', '011', '100', '101', '110', '111'] # 0 is min and 1 is max
        bbox3d = []
        for corner in corners:
            corner_vertex = []
            for dim in range(3):
                x = corner[dim]
                x = torch.min(self.verts[0, :, dim]) if x == '0' else torch.max(self.verts[0, :, dim])
                corner_vertex.append(x)

            bbox3d.append(corner_vertex)

        bbox3d = torch.tensor(bbox3d).unsqueeze(0).to(self.verts.device)
        return bbox3d

def load_obj(path, device):
    # load a .obj file and return vertices and faces
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    m = ms.current_mesh()
    verts = torch.tensor(m.vertex_matrix(), device=device, dtype=torch.float32)
    faces = torch.tensor(m.face_matrix(), device=device)
    return verts, faces

def load_objects(device, labels, nKps=10):
    # labels are 1-indexed
    translationDict = {
        "":"",
        "Overholt Clamp": "KlemmeOverholt",
        "Metz. Scissor": "Metzenb",
        "Sur. Scissor":"Chir",
        "Needle Holder": "Nadelhalter",
        "Sur. Forceps": "PinzetteMittelbreit_Split",
        "Atr. Forceps": "PinzetteDeBakey_Split",
        "Scalpel": "Skalpellgriff",
        "Retractor": "Wundhaken",
        "Hook": "Langenbeck",
        "Lig. Clamp": "KlemmeDeBakey_03",
        "Peri. Clamp": "PeritoneumKlemme",
        "Bowl": "Metallschale",
        "Tong": "Kornzange"
    } 

    xArticulated = ['PinzetteMittelbreit_Split', 'PinzetteDeBakey_Split']
    yArticulated = ['KlemmeOverholt', 'Metzenb', 'Chir', 'Nadelhalter', 'KlemmeDeBakey_03', 'PeritoneumKlemme', 'Kornzange']
    
    xArticulated = [s+'.obj' for s in xArticulated]
    yArticulated = [s+'.obj' for s in yArticulated]
    meshes_path = [s+'.obj' for s in np.array(list(translationDict.values()))[labels]]

    names = list(translationDict.values())
    objects = []
    for i, path in enumerate(meshes_path):
        verts, faces = load_obj(os.path.join('objects', path), device=device)
        label = labels[i]
        tool = SurgeryTool(verts, faces, xArticulated, yArticulated, names[label]+'.obj', nKps)
        objects.append(tool)
    
    return objects, translationDict



def farthest_point_down_sample(point_cloud, num_points):

    """
    Down-sample a 3D point cloud using the farthest point sampling algorithm.
    
    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the input point cloud.
        num_points (int): The desired number of down-sampled points.

    Returns:
        torch.Tensor: A tensor of shape (num_points, 3) representing the down-sampled point cloud.
    """
    N, _ = point_cloud.shape
    sampled_indices = []

    # Choose a random starting point
    start_idx = torch.tensor(0)
    sampled_indices.append(start_idx.item())
    distances = torch.norm(point_cloud - point_cloud[start_idx], dim=1)

    for _ in range(num_points - 1):
        # Find the point farthest from the already sampled points
        farthest_idx = torch.argmax(distances)
        sampled_indices.append(farthest_idx.item())

        # Update distances based on the newly added point
        new_distances = torch.norm(point_cloud - point_cloud[farthest_idx], dim=1)
        distances = torch.min(distances, new_distances)

    # Create the down-sampled point cloud
    down_sampled_cloud = point_cloud[sampled_indices, :]

    return down_sampled_cloud, sampled_indices