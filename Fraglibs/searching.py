from rdkit import Chem

def search_fragments(smiles, atom_indices):
    # 将SMILES转换为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 获取所有原子的邻接原子
    atom_neighbors = {atom.GetIdx(): set(atom.GetNeighbors()) for atom in mol.GetAtoms()}

    # 初始化分类结果
    classified_atoms = []
    visited = set()

    # 遍历原子索引，进行分类
    for atom_idx in atom_indices:
        if atom_idx in visited:
            continue

        # 从当前原子出发，找到所有连接的原子
        stack = [atom_idx]
        connected_atoms = set()

        while stack:
            current_idx = stack.pop()
            if current_idx not in visited:
                visited.add(current_idx)
                connected_atoms.add(current_idx)
                # 将当前原子的邻接原子加入栈中
                for neighbor in atom_neighbors[current_idx]:
                    if neighbor.GetIdx() in atom_indices and neighbor.GetIdx() not in visited:
                        stack.append(neighbor.GetIdx())

        # 将连接的原子组作为一个分类加入结果
        classified_atoms.append(list(connected_atoms))

    return classified_atoms
    
    
def boundary_atoms(smiles, atoms_idx):
    # 将SMILES转换为分子对象
    mol = Chem.MolFromSmiles(smiles)

    # 获取所有原子的邻接原子
    atom_neighbors = {atom.GetIdx(): [neighbor.GetIdx() for neighbor in atom.GetNeighbors()] for atom in mol.GetAtoms()}
    boundary = []
    for atom_idx in atoms_idx:

        for neighbor_idx in atom_neighbors[atom_idx]:
            if neighbor_idx in atoms_idx:
                continue
            else:
                boundary.append([atom_idx, neighbor_idx])

    return boundary


def get_negative_substructures(smiles, indices):
    """

    Find the continued atom of linker with negative contribution.

    :param smiles: The target smiles.
    :param contributions: The atomic contribution values.
    :param indexes: The indices of linker.
    :return: A dict of branch and linker with negative contribution.
    """

    mol = Chem.MolFromSmiles(smiles)

    atom_neighbors = {atom.GetIdx(): [neighbor.GetIdx() for neighbor in atom.GetNeighbors()] for atom in mol.GetAtoms()}
    results = {"linker": [], "branch": []}
    
    for idxs in indices:
        records = []
        
        for idx in idxs:
            atom_neighbor = atom_neighbors[idx]
            for neighbor in atom_neighbor:
                if neighbor not in idxs:
                    records.append(idx)
                else:
                    continue
        
        records = set(records)
        if len(records) == 1:
            results["branch"].append(idxs)
        elif len(records) > 1:
            results["linker"].append(idxs)
        else:
            print("Something wrong!")
            exit()

    return results    