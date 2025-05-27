import re
import numpy as np
import concurrent.futures
from rdkit import Chem
from itertools import product
from rdkit.Chem.QED import qed
from rdkit.Chem import Crippen, Descriptors
from .sascores import calculateScore


def connect_mols(mol1, mol2, atom1, atom2):
    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()
    atom1_idx = atom1.GetIdx()
    atom2_idx = atom2.GetIdx()
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx,
                 neighbor2_idx + mol1.GetNumAtoms(),
                 order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()

    return mol


def get_star_indices(smiles):
    """
    获取 SMILES 中所有 `*` 号标记的原子索引。

    :param smiles: 输入的 SMILES 字符串
    :return: 包含 `*` 号标记的原子索引的列表
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms = mol.GetAtoms()
    # 获取所有原子及其对应的 Isotope 属性
    atom_indices = []
    for atom in atoms:
        if '*' in atom.GetSmarts():
            atom_indices.append(atom.GetIdx())

    return atom_indices


def generate_connections(smiles_combination):
    frag, bran = smiles_combination
    mol1 = Chem.MolFromSmiles(frag)
    mol2 = Chem.MolFromSmiles(bran)
    indices1 = get_star_indices(frag)
    indices2 = get_star_indices(bran)

    generation = []
    for point_combination in product(indices1, indices2):
        point1 = point_combination[0]
        point2 = point_combination[1]
        atom1 = mol1.GetAtomWithIdx(point1)
        atom2 = mol2.GetAtomWithIdx(point2)
        connected_mol = connect_mols(mol1, mol2, atom1, atom2)
        connected_smiles = Chem.MolToSmiles(connected_mol)
        generation.append(connected_smiles)
    
    return generation


def connection(fragments: list, branches: list, cpu_count: int):
    generation = []
    # 使用 ProcessPoolExecutor 来实现并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        # 将 fragments 和 branches 的所有组合传入生成器
        results = executor.map(generate_connections, product(fragments, branches))
        
        # 收集结果
        for result in results:
            generation.extend(result)
    
    return generation


def validation(mol):
    if mol is None:
        return False

    if Chem.rdmolops.Kekulize(mol, clearAromaticFlags=True) is False:
        return False

    try:
        Chem.MolToSmiles(mol)
    except:
        return False

    mw = Descriptors.MolWt(mol)
    if mw < 60 or mw > 1200:
        return False

    return True


def parallel_validation(gen_list):
    # 使用 ThreadPoolExecutor 或 ProcessPoolExecutor 并行化 validation 检查
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 执行并行化任务，返回符合要求的 SMILES 列表
        valid_mols = list(executor.map(validation, [Chem.MolFromSmiles(gen) for gen in gen_list]))
    # 只返回有效的 SMILES
    return [gen for gen, is_valid in zip(gen_list, valid_mols) if is_valid]


def long_rings_count(mol):
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    large_rings = [ring for ring in rings if len(ring) > 6]
    return len(large_rings)


def calculate_drug_likeness(mol):
    """计算 QED, MolLogP 和 SAScore 以及长环信息"""
    q = qed(mol)
    logp = Crippen.MolLogP(mol)
    sascore = -calculateScore(mol)
    long_ring = -long_rings_count(mol)
    return q, logp, sascore, long_ring


def scoring(smiles_list, qed_cutoff=0.5, plogp_cutoff=0.0, num_cpu=16):
    # 将 SMILES 转换为分子对象
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        results = list(executor.map(calculate_drug_likeness, mols))
    
    qeds, logps, sascores, long_rings = zip(*results)
    drug_likeness = np.array([[logp, sascore, long_ring] for logp, sascore, long_ring in zip(logps, sascores, long_rings)])
    plogps = list(np.sum(drug_likeness, axis=1))
    
    smi_list = []
    qed_list = []
    plogp_list = []
    
    # 过滤出符合条件的 SMILES
    for qv, pv, smiles in zip(qeds, plogps, smiles_list):
        if qv > qed_cutoff and pv > plogp_cutoff:
            smi_list.append(smiles)
            qed_list.append(qv)
            plogp_list.append(pv)
        else:
            continue
            
    # 计算分数
    scores = [q * p for q, p in zip(qed_list, plogp_list)]
    
    return smi_list, scores, qed_list, plogp_list

# smi_list = ['CC(C)O[C@H](O)CC(=O)O','CCC(CC)C[C@H](O)CC(=O)O','O=C(O)C[C@@H](O)OC(F)(F)Cl','CC(C)(O)[C@H](O)CC(=O)O']
# scoring(smi_list, num_cpu=32)

def remove_star(smiles):
    # 使用正则表达式去掉形如 [数字*] 或 [*] 的部分
    return re.sub(r'\[\d*\*\]', '', smiles).replace('*', '')
    