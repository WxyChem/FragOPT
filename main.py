import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore')

import json
import joblib
import keras
import argparse
from Fraglibs.utils import fingerprints_morgan, dataset_sklearn
from Fraglibs.voting import voting_predict, interpretation
from Fraglibs.importance import *
from Fraglibs.searching import get_negative_substructures, search_fragments, boundary_atoms
from Fraglibs.dock import prepare_pdbqt, run_qvina, prepare_multi_pdb
from rdkit.Chem import Descriptors
from Fraglibs.components.growing import run_deepfrag
from Fraglibs.opt import connection, parallel_validation, scoring, remove_star
from rdkit.Chem.QED import qed
from rdkit import DataStructs

VERSION = "1.0.0"

activity_labels = {0: 'Inactive', 1: 'Active'}


def main():
    global VERSION

    parser = argparse.ArgumentParser(description='Identification of small molecules')
    parser.add_argument('-j', '--json_file', type=str, required=True, help='Input all parameters of the identification process')
    args = parser.parse_args()

    print("\n[*] FragOPT " + VERSION)
    
    j_file = args.json_file
    
    with open(j_file, 'r') as pr_file:
        pr_data = json.load(pr_file)

    try:
        os.makedirs(pr_data["Output_path"])
    except FileExistsError:
        pass

    rf_path = pr_data["RF_model_path"]
    svr_path = pr_data["SVM_model_path"]
    lstm_path = pr_data["LSTM_model_path"]
    mlp_path = pr_data["MLP_model_path"]
    xgb_path = pr_data["XGB_model_path"]
    models_path = [rf_path, svr_path, lstm_path, mlp_path, xgb_path]
    models_type = pr_data["models_type"]
    explainers_type = pr_data["explainers_type"]
    
    training_path = pr_data["Training_set_path"]
    color = pr_data["Drawing_color"]

    radius = pr_data["ECFP_radius"]
    hash_size = pr_data["ECFP_hash_size"]

    df = pd.read_csv(pr_data["filename_of_target_molecule"], header=None)
    smiles_list = df.iloc[:, 0].tolist()
    
    pdb_file = pr_data["PDB_file_path"]
    atom_csv = pr_data["WEIGHT_file_path"]
    img_file = pr_data["IMG_file_path"]
    sdf_file = pr_data["SDF_file_path"]
    pdbqt_file = pr_data["PDBQT_file_path"]
    
    prepare_path = pr_data["DEEPFRAG_config"]
    
    vina_path = pr_data["quickvina_path"]
    receptor_pdbqt = pr_data["Receptor"]
    cpu_num = pr_data["cpu_number"]
    mode_num = pr_data["mode_number"]
    center_x = pr_data["center_x"] 
    center_y = pr_data["center_y"] 
    center_z = pr_data["center_z"] 
    size_x = pr_data["size_x"]
    size_y = pr_data["size_y"]
    size_z = pr_data["size_z"]
    output_pdbqt = pr_data["output_pdbqt"]
    output_pdb = pr_data["output_pdb"]
    
    top_k = pr_data["top_k"]
    num = len(smiles_list)
    
    if num == 1:
    
        print(f"\n[*] Input Molecule Smiles:{smiles_list[0]}")
        
        fps, _ = fingerprints_morgan(smiles_list, radius, hash_size)
        fps = np.array(fps)
        models_list = []
        
        for model_path, model_type in zip(models_path, models_type):
            if model_type == "sklearn":
                model = joblib.load(model_path)
                models_list.append(model)
            elif model_type == "keras":
                model = keras.models.load_model(model_path)
                models_list.append(model)
            else:
                print(f"The model type is not exist: {model_type}, please check your json file!")
                exit()
            
        pred, pred_probability = voting_predict(models_list, models_type, fps)
    
        for i, p in enumerate(pred):
            print(f"\n[*] The molecule {smiles_list[0]} was predicted as {activity_labels[p]}")
    
        original_data = pd.read_csv(training_path)
        original = original_data.iloc[:, 0].tolist()
        labels = np.array(original_data.iloc[:, 1].tolist())
        X_train, _, _, _, _, _ = dataset_sklearn(original, labels, normalization=False, val_set=True, random_state=pr_data["Random_seed"])
        
        # Interpretation module
        print("\n[*] Starting interpretation module ...")
        print("[*] Running weights calculation ...")
    
        weights = []
        X = np.array([fps[0]])
        for _ in range(0, 3):
            w = interpretation(models_list, models_type, explainers_type, X, X_train, num_sample=10)
            weights.append(w)
        
        w_matrix = (weights[0] + weights[1] + weights[2]) / 3
        w_matrix = list(w_matrix)
        
        bit_name = range(0, hash_size)
        
        # 3D conformors generation module
        print("\n[*] Processing the molecule to 3D structure...")
        
        num_atoms = get_smiles_atom_count(smiles_list[0])
        generate_optimized_3d_structure(smiles_list[0], pdb_file)
        atom_names_list = extract_atoms_from_pdb(pdb_file, num_atoms)
        prepare_pdbqt(pdb_file, pdbqt_file)
        run_qvina(vina_path, receptor_pdbqt, pdbqt_file, cpu_num, center_x, center_y, center_z, size_x, size_y, size_z, output_pdbqt, mode_num)
        prepare_multi_pdb(output_pdbqt, output_pdb)
        atom_nums, contributions_list = contribution_visualization(smiles=smiles_list[0], bit_numbers=bit_name, importance=w_matrix, 
                                                           morgan_radius=radius, morgan_hash_size=hash_size, color_map=color, 
                                                           weights_path=atom_csv, image_path=img_file, 
                                                           method=pr_data["Distribution"], fig_save=True, weights_save=False)
        
        # add_atom_names_to_csv(atom_csv, atom_csv, pdb_atom_names)
        # df1 = pd.read_csv(atom_csv)
        # contributions_list = df1['atom_weights'].tolist()
        # atom_names_list = df1['AtomName'].tolist()
        
        mol = Chem.MolFromSmiles(smiles_list[0])
        compose_index = [num for num, w in zip(atom_nums, contributions_list) if w > 0]
        decompose_index = [num for num, w in zip(atom_nums, contributions_list) if w <= 0]
        
        fragments_index = search_fragments(smiles_list[0], compose_index)
        fragments = set()
        for fragment_index in fragments_index:
            bound_atoms = boundary_atoms(smiles_list[0], fragment_index)
            bond_idxs = []
            for atoms in bound_atoms:
                bond = mol.GetBondBetweenAtoms(atoms[0], atoms[1])
                if bond:
                    bond_idxs.append(bond.GetIdx())
                else:
                    continue
                
                mol_f = Chem.FragmentOnBonds(mol, bond_idxs)
                frags = Chem.MolToSmiles(mol_f)
                frags = frags.split(".")
                for frag in frags:
                    if Descriptors.MolWt(Chem.MolFromSmiles(frag)) > 60:
                        fragments.add(frag)
                    else:
                        continue
        
        fragments = list(fragments)
        negative_fragments_indices = search_fragments(smiles_list[0], decompose_index)
        cnames = []
        rnames = []
        with open(prepare_path, 'w', encoding='utf-8') as f:
            
            for nf in negative_fragments_indices:
                fragment_seed = None
                for idx in nf:
                    atom = mol.GetAtomWithIdx(idx)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() not in nf:
                            fragment_seed = neighbor.GetIdx()
                            f.write("branch: %s %s %s %s\r" % ("cname", atom_names_list[fragment_seed], "rname", atom_names_list[idx]))
                            cnames.append(atom_names_list[fragment_seed])
                            rnames.append(atom_names_list[idx])
                            
                            if "C" in atom_names_list[idx] or "N" in atom_names_list[idx] or "O" in atom_names_list[idx]:
                                f.write("branch: %s %s %s %s\r" % ("cname", atom_names_list[idx], "rname", atom_names_list[fragment_seed]))
                                cnames.append(atom_names_list[idx])
                                rnames.append(atom_names_list[fragment_seed])
                            else:
                                continue
        
        # Running DeepFrag to generate branches.
        branch_path = []
        if len(cnames) == len(rnames):
            i = 1
            print("\n[*] Running DeepFrag to generate fragments!")
            print(f'[*] Loading receptor: {pr_data["rec_path"]} ... ', end='')
            print('done.')
            fgs = []
            for j in range(int(mode_num)):
                for cname, rname in zip(cnames, rnames):
    
                    try:
                        branch_output = f"{pr_data['Output_path']}/branch_{i}.csv"
                        df_pdb = f"{output_pdb}_{j+1}.pdb"
                        fg = run_deepfrag('cpu', 100, pr_data["rec_path"], df_pdb, cname, rname, 10, branch_output)
                        if fg not in fgs:
                            fgs.append(fg)
                            
                        i = i + 1
                        branch_path.append(branch_output)
                    except:
                        pass
                
        else:
            print("SOMETHING WRONG!")
            exit()
        
        fgs_fps = []
        for ff in fgs:
            if len(ff) > 3:
                fgs_fps.append(Chem.RDKFingerprint(Chem.MolFromSmiles(ff)))
        
        fragments_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in fragments]
        sims_idx = []
        for ff in fgs_fps:
            sims = [DataStructs.FingerprintSimilarity(ff, frag_f) for frag_f in fragments_fps]
            s_idx = sims.index(max(sims))
            
            if s_idx not in sims_idx:
                sims_idx.append(s_idx)
        
        tm_fragments = fragments.copy()
        for sim_idx in sims_idx:
            fragments.remove(tm_fragments[sim_idx])
        
        print("\n[*] Remain fragments from original molecule!")            
        print(fragments)    
        
        if branch_path:
            branch_list = []
            complete_smiles = []
            print("\n[*] Generation Start ...")
            for fragment in fragments:
                com_frag = remove_star(fragment)
                complete_smiles.append(com_frag)
            
            for path in branch_path:
                df_branch = pd.read_csv(path, sep=',')
                ba = df_branch['SMILES'].tolist()
                branch_list = branch_list + ba
            
            branch_list = set(branch_list)
            branch_list = list(branch_list)
            
            gen_smiles = connection(fragments, branch_list, int(cpu_num))
            gen_list = parallel_validation(gen_smiles)
            print(f"\n{len(gen_list)} molecules generated in the first stage")
            g_df = pd.DataFrame()
            g_df['smiles'] = gen_list
            g_df.to_csv('step1.csv', index=False)
            gen_fps, _ = fingerprints_morgan(gen_list, radius, hash_size)
            prediction, _ = voting_predict(models_list, models_type, gen_fps)
            remain_gen_list= []
            for gen, pre in zip(gen_list, prediction):
                if pre == 1:
                    remain_gen_list.append(gen)
            
            remain_gen_list, _, _, _ = scoring(remain_gen_list, qed_cutoff=0.5, plogp_cutoff=0.0, num_cpu=int(cpu_num))

            fragments = []
            for gen in remain_gen_list:
                if '*' not in gen:
                    complete_smiles.append(gen)
                else:
                    gen_frag = remove_star(gen)
                    complete_smiles.append(gen_frag)
                    fragments.append(gen)
            
            fragments_number = len(fragments)

            if fragments:
                gen_smiles = connection(fragments, branch_list, int(cpu_num))
                gen_list = parallel_validation(gen_smiles)
                print(f"\n{len(gen_list)} molecules generated in the second stage")
                
                gen_fps, _ = fingerprints_morgan(gen_list, radius, hash_size)
                prediction, _ = voting_predict(models_list, models_type, gen_fps)
                remain_gen_list= []
                for gen, pre in zip(gen_list, prediction):
                    if pre == 1:
                        remain_gen_list.append(gen)
                
                fragments = []
                for gen in remain_gen_list:
                    if '*' not in gen:
                        complete_smiles.append(gen)
                    else:
                        gen_frag = remove_star(gen)
                        complete_smiles.append(gen_frag)
                        fragments.append(gen)
                
                remain_gen_list, scores, _, _ = scoring(fragments, qed_cutoff=0.5, plogp_cutoff=0.0, num_cpu=int(cpu_num))
                df_frags = pd.DataFrame()
                df_frags['SMILES'] = remain_gen_list
                df_frags['SCORE'] = scores
                df_fragments = df_frags.sort_values(by='SCORE', ascending=False)
                
                fragments = df_fragments['SMILES'].tolist()
                fragments = fragments[0:fragments_number]

            if fragments:
                gen_smiles = connection(fragments, branch_list, int(cpu_num))
                gen_list = parallel_validation(gen_smiles)
                print(f"\n{len(gen_list)} molecules generated in the third stage")
                
                gen_fps, _ = fingerprints_morgan(gen_list, radius, hash_size)
                prediction, _ = voting_predict(models_list, models_type, gen_fps)
                remain_gen_list= []
                for gen, pre in zip(gen_list, prediction):
                    if pre == 1:
                        remain_gen_list.append(gen)    

                num_with_star = 0
                for gen in remain_gen_list:
                    if '*' not in gen:
                        complete_smiles.append(gen)
                    else:
                        num_with_star = num_with_star + 1
                        gen_frag = remove_star(gen)
                        complete_smiles.append(gen_frag)
            
                print(f"There are {num_with_star} fragments remained.")
                
            final_smiles = parallel_validation(complete_smiles)
            smiles_list, scores, qeds, plogps = scoring(final_smiles, qed_cutoff=0.5, plogp_cutoff=0.0, num_cpu=int(cpu_num))
            
            if len(smiles_list) == len(scores):
                data = pd.DataFrame()
                data['SMILES'] = smiles_list
                data['scores'] = scores
                data['qed'] = qeds
                data['plogp'] = plogps
                
                output_data = data.sort_values(by='scores', ascending=False)
                output_data.index = range(len(smiles_list))
                
                if top_k != -1 or len(smiles_list) > top_k:
                    output_data = output_data[0:top_k]
                    output_data.to_csv(pr_data["generation_output"], index=False)
                    print("\n[*] Generation Done!")
                elif top_k == -1 or len(complete_smiles) <= top_k:
                    output_data.to_csv(pr_data["generation_output"], index=False)
                    print("\n[*] Generation Done!")
                
            
            else:
                print("SOMETHING WRONG!")
                exit()
            
        else:
            print("SOMETHING WRONG!")
            exit()
            
    else:
        print("[*] Only accept one molecule from input file, the input file format see example.txt.")
        exit()
        
        
if __name__ == '__main__':
    main()
