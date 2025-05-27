import os
import re
import subprocess
from openbabel import openbabel


def prepare_pdbqt(input_pdb: str, output_pdbqt: str):
    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")
        
    obConversion = openbabel.OBConversion()
    
    obConversion.OpenInAndOutFiles(input_pdb, output_pdbqt)
    obConversion.SetInAndOutFormats("pdb","pdbqt")
    obConversion.Convert()
    obConversion.CloseOutFile()


def run_qvina(vina_path: str, receptor_path: str, ligand_path: str, cpu_count: str, center_x: str, center_y: str, center_z: str,
              size_x: str, size_y: str, size_z: str, output: str, mode_num: str, seed: str = '42'):
    """

    :param output:
    :param receptor_path:
    :param ligand_path:
    :param cpu_count:
    :param center_x:
    :param center_y:
    :param center_z:
    :param size_x:
    :param size_y:
    :param size_z:
    :param seed:
    :return:
    """
    if not os.path.exists(vina_path):
        raise FileNotFoundError(f"Vina executable not found: {vina_path}")
    if not os.path.exists(receptor_path):
        raise FileNotFoundError(f"Receptor PDBQT file not found: {receptor_path}")
    if not os.path.exists(ligand_path):
        raise FileNotFoundError(f"Ligand PDBQT file not found: {ligand_path}")
        
    command = [vina_path, '--receptor', receptor_path, '--ligand', ligand_path, '--center_x', center_x,
               '--center_y', center_y, '--center_z', center_z, '--size_x', size_x, '--size_y', size_y,
               '--size_z', size_z, '--seed', seed, '--cpu', cpu_count, '--num_modes', mode_num, '--out', output]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Standard output: {e.stdout.decode('utf-8')}")
        print(f"Standard error: {e.stderr.decode('utf-8')}")
        

def extract_energy(file_path: str):
    with open(file_path, 'r') as file:
        content = file.read()

    energy_pattern = r"REMARK VINA RESULT:\s*(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    matches = re.findall(energy_pattern, content)

    if matches:
        energy = matches[0][0]
        return float(energy)
    else:
        print("Energy not found in the file.")
        return None


def prepare_pdb(input_pdbqt: str, output_pdb: str):
    if not os.path.exists(input_pdbqt):
        raise FileNotFoundError(f"Input PDBQT file not found: {input_pdbqt}")
        
    obConversion = openbabel.OBConversion()
    
    obConversion.OpenInAndOutFiles(input_pdbqt, output_pdb)
    obConversion.SetInAndOutFormats("pdbqt","pdb")
    obConversion.Convert()
    obConversion.CloseOutFile()


def prepare_multi_pdb(input_pdbqt: str, output_prefix: str):
    """
    将包含多个结构的PDBQT文件分割为单独的PDB文件。
    
    Args:
        input_pdbqt (str): 输入的PDBQT文件路径
        output_prefix (str): 输出文件前缀（如"output"会生成output_1.pdb, output_2.pdb）
    """
    if not os.path.exists(input_pdbqt):
        raise FileNotFoundError(f"Input PDBQT file not found: {input_pdbqt}")

    # 初始化Open Babel对象
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("pdbqt")
    
    # 读取所有结构
    mol = openbabel.OBMol()
    structures = []
    not_at_end = obConversion.ReadFile(mol, input_pdbqt)
    while not_at_end:
        structures.append(mol)
        mol = openbabel.OBMol()  # 创建新分子对象
        not_at_end = obConversion.Read(mol)  # 读取下一个结构

    # 确保找到至少一个结构
    if not structures:
        raise ValueError("No structures found in the input PDBQT file")

    # 保存每个结构到单独的PDB文件
    obConversion.SetOutFormat("pdb")
    for i, structure in enumerate(structures, 1):
        output_path = f"{output_prefix}_{i}.pdb"
        obConversion.WriteFile(structure, output_path)
        print(f"Saved structure {i} to {output_path}")
        
# input_pdb = '/home/ls/wuxiaoyan/FragOPT/output_folder_5n2f/5n2f_ligand.pdb'
# output_pdbqt1 = '/home/ls/wuxiaoyan/FragOPT/output_folder_5n2f/5n2f_ligand.pdbqt'
# vina_path = './libs/vina/qvina-w'
# receptor_pdbqt = './PDB/5N2F/Protein.pdbqt'
# ligand_pdbqt = './output_folder_5n2f/5n2f_ligand.pdbqt'
# output_pdbqt2 = './output_folder_5n2f/5n2f_ligand_out.pdbqt'
# 
# # step 1: convert PDB to PDBQT
# prepare_pdbqt(input_pdb, output_pdbqt1)
# 
# # step 2: run Quick Vina to dock
# run_qvina(vina_path, receptor_pdbqt, ligand_pdbqt, '32', '32.168', '11.761', '126.19', '30.0', '35.0', '39.0', output_pdbqt2)
# 
# # step 3: extract the binding energy
# energy = extract_energy(output_pdbqt2)
# 
# print(energy)
# prepare_pdb(output_pdbqt2, '5n2f_ligand_out.pdb')
