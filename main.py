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

VERSION = "1.1.0"

activity_labels = {0: 'Inactive', 1: 'Active'}

def main():
    global VERSION

    parser = argparse.ArgumentParser(description='Identification of small molecules')
    parser.add_argument('-j', '--json_file', type=str, required=True, help='Input all parameters of the identification process')
    args = parser.parse_args()
    
    print("\n[*] FragOPT " + VERSION)
    
    # Load the json file to dict
    j_file = args.json_file
    with open(j_file, 'r') as pr_file:
        pr_data = json.load(pr_file)

    try:
        os.makedirs(pr_data["Output_path"])
    except FileExistsError:
        pass
    
    # Setting the model path
    rf_path = pr_data["RF_model_path"]
    svr_path = pr_data["SVM_model_path"]
    lstm_path = pr_data["LSTM_model_path"]
    mlp_path = pr_data["MLP_model_path"]
    xgb_path = pr_data["XGB_model_path"]
    
    # Setting the explainer type for each model
    models_path = [rf_path, svr_path, lstm_path, mlp_path, xgb_path]
    models_type = pr_data["models_type"]
    explainers_type = pr_data["explainers_type"]
    
    # Setting the training set path for the explainer
    training_path = pr_data["Training_set_path"]
    
    # Setting the color for the output plot
    color = pr_data["Drawing_color"]
    
    # Setting the ECFP fingerprint parameters
    radius = pr_data["ECFP_radius"]
    hash_size = pr_data["ECFP_hash_size"]
    
    # Setting the input molecule
    df = pd.read_csv(pr_data["filename_of_target_molecule"], header=None)
    smiles_list = df.iloc[:, 0].tolist()
    
    # Setting the 3D structure save path
    pdb_file = pr_data["PDB_file_path"]
    sdf_file = pr_data["SDF_file_path"]
    pdbqt_file = pr_data["PDBQT_file_path"]
    
    # Setting the process information path
    atom_csv = pr_data["WEIGHT_file_path"]
    img_file = pr_data["IMG_file_path"]
    
    # DeepFrag Config path
    prepare_path = pr_data["DEEPFRAG_config"]
    
    # Setting Sampling program
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
    
    # Setting the number of saved molecules
    top_k = pr_data["top_k"]
    size = pr_data["gen_size"]
    
    num = len(smiles_list)
    
    if num == 1:
        print(f"\n[*] Input Molecule Smiles:{smiles_list[0]}")
        
        # Atomic importance identification
        fps, _ = fingerprints_morgan(smiles_list, radius, hash_size)
        fps = np.array(fps)
        
        # assign model type for each model
        models_list = []
        for model_path, model_type in zip(models_path, models_type):
            if model_type == "sklearn":
                model = joblib.load(model_path)
                models_list.append(model)
            elif model_type == "keras":
                model = keras.models.load_model(model_path)
                models_list.append(model)
            else:
                raise ValueError(f"The model type is not exist: {model_type}, please check your json file!")
        
        # Predict the bioactivity for the target molecule 
        pred, pred_probability = voting_predict(models_list, models_type, fps)
        
        print(f"\n[*] The molecule {smiles_list[0]} was predicted as {activity_labels[pred[0]]}")
        
        # Explainer 
        # Step1: preparation of reference molecules (using training set)
        print("\n[*] Starting interpretation module ...")
        
        od = pd.read_csv(training_path)
        od_list = od.iloc[:, 0].values.tolist()
        labels = np.array(od.iloc[:, 1].tolist())
        X_train, _, _, _, _, _ = dataset_sklearn(od_list, labels, normalization=False, val_set=True, random_state=pr_data["Random_seed"])
        
        # Step2: Running shap method with three times to get shapley values
        print("[*] Running weights calculation ...")
        weights = []
        X = np.array([fps[0]])
        
        for _ in range(3):
            w = interpretation(models_list, models_type, explainers_type, X, X_train, num_sample=10)
            weights.append(w)
        
        w_matrix = (weights[0] + weights[1] + weights[2]) / 3
        w_matrix = list(w_matrix)
        
        # Mapping the shapley values into all fragments
        print("\n[*] Mapping the weights on the atoms...")
        bit_name = range(0, hash_size)
        atom_nums, contributions_list = contribution_visualization(smiles=smiles_list[0], bit_numbers=bit_name, importance=w_matrix, 
                                                                   morgan_radius=radius, morgan_hash_size=hash_size, color_map=color, 
                                                                   weights_path=atom_csv, image_path=img_file, method=pr_data["Distribution"], 
                                                                   fig_save=True, weights_save=False)        
        
        # Generating the 3D structures
        print("\n[*] Processing the molecule to 3D structure...")
        num_atoms = get_smiles_atom_count(smiles_list[0])
        generate_optimized_3d_structure(smiles_list[0], pdb_file)
        atom_names_list = extract_atoms_from_pdb(pdb_file, num_atoms)
        prepare_pdbqt(pdb_file, pdbqt_file)
        run_qvina(vina_path, receptor_pdbqt, pdbqt_file, cpu_num, center_x, center_y, 
                  center_z, size_x, size_y, size_z, output_pdbqt, mode_num)
        prepare_multi_pdb(output_pdbqt, output_pdb)
        
        # Get the indexes of positive and negative fragments
        mol = Chem.MolFromSmiles(smiles_list[0])
        compose_index = [num for num, w in zip(atom_nums, contributions_list) if w > 0]
        decompose_index = [num for num, w in zip(atom_nums, contributions_list) if w <= 0]
        fragments_index = search_fragments(smiles_list[0], compose_index)
        
        # Get the fragments by rdkit
        fragments = set()
        for fi in fragments_index:
            # Searching the boundary_atoms between positive and negative fragments
            bound_atoms = boundary_atoms(smiles_list[0], fi)
            bond_idxs = []
            for ap in bound_atoms:
                bond = mol.GetBondBetweenAtoms(ap[0], ap[1])
                if bond:
                    bond_idxs.append(bond.GetIdx())
                else:
                    continue
                    
                fs = Chem.FragmentOnBonds(mol, bond_idxs)
                frags = Chem.MolToSmiles(fs)
                frags = frags.split(".")
                fmws = [Descriptors.MolWt(Chem.MolFromSmiles(frag)) for frag in frags]
                df_f = pd.DataFrame({"frag":frags,"mw": fmws})
                df_fs = df_f.sort_values(by=['mw'], ascending=[False])
                fsl = len(frags) // 2
                fs = df_fs['frag'].values.tolist()
                fs = fs[:fsl]
                for f in fs:
                    fragments.add(f)
        
        fragments = list(fragments)
        print("The fragments split from original molecule:")
        print(fragments)
        
        # DeepFrag Configuration
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
        
        
        # Running DeepFrag to generate branches
        branch_path = []
        if len(cnames) == len(rnames):
            i = 1
            print("\n[*] Running DeepFrag to generate fragments!")
            print(f'[*] Loading receptor: {pr_data["rec_path"]} ... ', end='')
            print('done.')
            fgs = set()
            for j in range(int(mode_num)):
                for cname, rname in zip(cnames, rnames):
                    try:
                        branch_output = f"{pr_data['Output_path']}/branch_{i}.csv"
                        df_pdb = f"{output_pdb}_{j+1}.pdb"
                        fg = run_deepfrag('cpu', 100, pr_data["rec_path"], df_pdb, cname, rname, 10, branch_output)
                        fgs.update(fg)
                        i = i + 1
                        branch_path.append(branch_output)
                    except:
                        pass
        else:
            raise ValueError("Could not configure the DeepFrag for Generating fragments")
        
        # Similarity Check(Remove the same fragments)
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

        # Connecting the fragments with generated branches
        epochs = max([f.count('*') for f in fragments])
        
        if branch_path:
            branch_lib = set()
            complete_smiles = set()
            print("\n[*] Generation Start ...")
            for fragment in fragments:
                com_frag = remove_star(fragment)
                complete_smiles.add(com_frag)
                
            # Loading the all branches
            for path in branch_path:
                df_branch = pd.read_csv(path, sep=',')
                ba = df_branch['SMILES'].values.tolist()
                branch_lib.update(ba)
            
            branch_lib = list(branch_lib)
            frag_lib = fragments.copy()
            
            # Start to generate
            for epoch in range(epochs):
                print(f"\n[*] Starting the {epoch + 1} epoch for molecular generation")
                gen_smiles = connection(frag_lib, branch_lib, int(cpu_num))
                gen_list = parallel_validation(gen_smiles)
                print(f"\n{len(gen_list)} molecules generated in the {epoch + 1} epoch, see step{epoch + 1}.csv")
                df_g = pd.DataFrame({'smiles': gen_list})
                df_g.to_csv(f"{pr_data['Output_path']}/step{epoch + 1}.csv", index=False)
                gen_fps, _ = fingerprints_morgan(gen_list, radius, hash_size)
                gen_fps = np.array(gen_fps)
                prediction, _ = voting_predict(models_list, models_type, gen_fps)
                remains = []
                for gen, pre in zip(gen_list, prediction):
                    if pre == 1:
                        remains.append(gen)
                
                if remains:
                    remain_wi = []
                    remain_wo = []
                    for r in remains:
                        if '*' not in r:
                            remain_wo.append(r)
                        else:
                            remain_wi.append(r)
                else:
                    break
                
                if remain_wo:
                    complete_smiles.update(remain_wo)
                else:
                    pass
                
                if remain_wi:
                    # Selecting fragments for next generation epoch
                    remain_wi, scores, _, _ = scoring(remain_wi, qed_cutoff=0.2, plogp_cutoff=0.0, num_cpu=int(cpu_num))
                    if remain_wi:
                        df_f = pd.DataFrame({'SMILES': remain_wi, 'SCORES': scores})
                        df_fs = df_f.sort_values(by='SCORES', ascending=False)
                        frag_lib = df_fs['SMILES'].values.tolist()
                        frag_lib = frag_lib[:size]
                        complete_smiles.update([remove_star(ri) for ri in remain_wi])
                    else:
                        break
                else:
                    break
                
                
        else:
            raise ValueError("\n[*] No branch.csv file exist!")
        
        # Finally Output 
        final_smiles = parallel_validation(complete_smiles)
        smiles_list, scores, qeds, plogps = scoring(final_smiles, qed_cutoff=0.5, plogp_cutoff=0.0, num_cpu=int(cpu_num))
        if len(smiles_list) == len(scores):
            data = pd.DataFrame({"SMILES": smiles_list, "SCORES": scores, "QED": qeds, "PlogP": plogps})
            data_o = data.sort_values(by='SCORES', ascending=False)
            data_o.index = range(len(smiles_list))
            
            if top_k != -1 or len(smiles_list) > top_k:
                data_o = data_o[0:top_k]
                data_o.to_csv(pr_data["generation_output"], index=False)
                print("\n[*] Generation Done!")
            elif top_k == -1 or len(complete_smiles) <= top_k:
                data_o.to_csv(pr_data["generation_output"], index=False)
                print("\n[*] Generation Done!")
            else:
                raise ValueError("\n[*] The top_k value should be set to 100")
        else:
            raise ValueError("\n[*] Something wrong in scoring module!")

    else:
        raise ValueError("\n[*] Only accept one molecule from input file, the input file format see example.txt.")

if __name__ == '__main__':
    main()
    