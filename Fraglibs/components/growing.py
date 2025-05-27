import os
import pathlib
import time
from typing import Tuple
from tqdm.auto import tqdm
import h5py
import numpy as np
import rdkit.Chem.AllChem as Chem
import torch
import prody

from .grow_branches.model_conf import LeadoptModel, REC_TYPER, LIG_TYPER, DIST_FN
from .grow_branches import util, grid_util


USER_DIR = './models'


VERSION = "1.0.0"


def get_deepfrag_user_dir() -> pathlib.Path:
    user_dir = pathlib.Path(os.path.realpath(__file__)).parent / USER_DIR
    os.makedirs(str(user_dir), exist_ok=True)
    return user_dir


def get_model_path():
    return get_deepfrag_user_dir() / 'model'


def get_fingerprints_path():
    return get_deepfrag_user_dir() / 'fingerprints.h5'


def get_structure_paths(receptor_path, ligand_path):
    """Get structure paths specified by the command line args.
    Returns (rec_path, lig_path)
    """
    if receptor_path is not None and ligand_path is not None:
        return (receptor_path, ligand_path)
    else:
        raise NotImplementedError()


def preprocess_ligand_without_removal_point(lig, conn):
    """
    Mark the atom at conn as a connecting atom. Useful when adding a fragment.
    """

    lig_pos = lig.GetConformer().GetPositions()
    lig_atm_conn_dist = np.sum((lig_pos - conn) ** 2, axis=1)
    
    # Get index of min
    min_idx = int(np.argmin(lig_atm_conn_dist))

    # Get atom at that position
    lig_atm_conn = lig.GetAtomWithIdx(min_idx)

    # Add a dummy atom to the ligand, connected to lig_atm_conn
    dummy_atom = Chem.MolFromSmiles("*")
    merged = Chem.RWMol(Chem.CombineMols(lig, dummy_atom))

    idx_of_dummy_in_merged = int([a.GetIdx() for a in merged.GetAtoms() if a.GetAtomicNum() == 0][0])
    bond = merged.AddBond(min_idx, idx_of_dummy_in_merged, Chem.rdchem.BondType.SINGLE)

    return merged


def preprocess_ligand_with_removal_point(lig, conn, rvec):
    """
    Remove the fragment from lig connected via the atom at conn and containing
    the atom at rvec. Useful when replacing a fragment.
    """
    # Generate all fragments.
    frags = util.generate_fragments(lig)
    
    for parent, frag in frags:
        # Get the index of the dummy (connection) atom on the fragment.
        cidx = [a for a in frag.GetAtoms() if a.GetAtomicNum() == 0][0].GetIdx()

        # Get the coordinates of the associated atom (the dummy atom's
        # neighbor).
        vec = frag.GetConformer().GetAtomPosition(cidx)
        c_vec = np.array([vec.x, vec.y, vec.z])

        # Check connection point.
        if np.linalg.norm(c_vec - conn) < 1e-3:
            # Check removal point.
            frag_pos = frag.GetConformer().GetPositions()
            min_dist = np.min(np.sum((frag_pos - rvec) ** 2, axis=1))

            if min_dist < 1e-3:
                # You have found the parent/fragment split that correctly
                # exposes the user-specified connection-point atom.

                # Found fragment.
                print('[*] Removing fragment with %d atoms (%s)' % (
                    frag_pos.shape[0] - 1, Chem.MolToSmiles(frag, False)))

                return parent, Chem.MolToSmiles(frag, False)
        
    # print('[!] Could not find a suitable fragment to remove.')
    # exit(-1)
    raise ValueError('[!] Could not find a suitable fragment to remove.')


def lookup_atom_name(lig_path, name):
    """Try to look up an atom by name. Returns the coordinate of the atom if
    found."""
    with open(lig_path, 'r') as f:
        p = prody.parsePDBStream(f)
    p = p.select(f'name {name}')
    if p is None:
        print(f'[!] Error: no atom with name "{name}" in ligand')
        exit(-1)
    elif len(p) > 1:
        print(f'[!] Error: multiple atoms with name "{name}" in ligand')
        exit(-1)
    return p.getCoords()[0]


def get_structures(receptor_path, ligand_path, cname, rname):
    rec_path, lig_path = get_structure_paths(receptor_path, ligand_path)

    #print(f'[*] Loading receptor: {rec_path} ... ', end='')
    rec_coords, rec_types = util.load_receptor_ob(rec_path)
    #print('done.')

    #print(f'[*] Loading ligand: {lig_path} ... ', end='')
    lig = Chem.MolFromPDBFile(lig_path)
    #print('done.')

    conn = None
    if cname is not None:
        conn = lookup_atom_name(lig_path, cname)
    else:
        raise NotImplementedError()

    rvec = None
    if rname is not None:
        rvec = lookup_atom_name(lig_path, rname)
    else:
        pass

    if rvec is not None:
        # Fragment repalcement (rvec specified)
        lig, fg = preprocess_ligand_with_removal_point(lig, conn, rvec)
        
    else:
        # Only fragment addition
        lig = preprocess_ligand_without_removal_point(lig, conn)
        fg = None
        
    parent_coords = util.get_coords(lig)
    parent_types = np.array(util.get_types(lig)).reshape((-1,1))
    
    return (rec_coords, rec_types, parent_coords, parent_types, conn, lig, fg)


def get_model(device):
    """Load a pre-trained DeepFrag model."""
    # print('[*] Loading model ... ', end='')
    model = LeadoptModel.load(str(get_model_path() / 'final_model'), device=device)
    # print('done.')
    return model


def get_fingerprints():
    """Load the fingerprint library.
    Returns (smiles, fingerprints).
    """
    f_smiles = None
    f_fingerprints = None
    # print('[*] Loading fingerprint library ... ', end='')
    with h5py.File(str(get_fingerprints_path()), 'r') as f:
        f_smiles = f['smiles'][()]
        f_fingerprints = f['fingerprints'][()].astype(float)
    # print('done.')

    return (f_smiles, f_fingerprints)

def generate_grids(num_grids, model_args, rec_coords, rec_types, parent_coords, parent_types, conn, device):
    start = time.time()

    print('[*] Generating grids ... ', end='', flush=True)
    batch = grid_util.get_raw_batch(
        rec_coords, rec_types, parent_coords, parent_types,
        rec_typer=REC_TYPER[model_args['rec_typer']],
        lig_typer=LIG_TYPER[model_args['lig_typer']],
        conn=conn,
        num_samples=num_grids,
        width=model_args['grid_width'],
        res=model_args['grid_res'],
        point_radius=model_args['point_radius'],
        point_type=model_args['point_type'],
        acc_type=model_args['acc_type'],
        cpu=(device == 'cpu')
    )
    # print('done.')
    end = time.time()
    print(f'[*] Generated grids in {end-start:.3f} seconds.')

    return batch


def get_predictions(model, batch, f_smiles, f_fingerprints):
    start = time.time()
    pred = model.predict(torch.tensor(batch).float()).cpu().numpy()
    end = time.time()
    print(f'[*] Generated prediction in {end-start} seconds.')

    avg_fp = np.mean(pred, axis=0)
    dist_fn = DIST_FN[model._args['dist_fn']]

    # The distance functions are implemented in pytorch so we need to convert our
    # numpy arrays to a torch Tensor.
    dist = 1 - dist_fn(
        torch.tensor(avg_fp).unsqueeze(0),
        torch.tensor(f_fingerprints))

    # Pair smiles strings and distances.
    dist = list(dist.numpy())
    scores = list(zip(f_smiles, dist))
    scores = sorted(scores, key=lambda x:x[1], reverse=True)
    scores = [(a.decode('ascii'), b) for a,b in scores]

    return scores


def gen_output(output, scores):
    if output is None:
        # Write results to stdout.
        print('%4s %8s %s' % ('#', 'Score', 'SMILES'))
        for i in range(len(scores)):
            smi, score = scores[i]
            print('%4d %8f %s' % (i+1, score, smi))
    else:
        # Write csv output.
        csv = 'Rank,SMILES,Score\n'
        for i in range(len(scores)):
            smi, score = scores[i]
            csv += '%d,%s,%f\n' % (
                i+1, smi, score
            )

        open(output, 'w').write(csv)
        # print('[*] Wrote output to %s' % output)


def run_deepfrag(device, num_grid, receptor, ligand, cname, rname, top_k, output):
    device = device

    model = get_model(device)
    f_smiles, f_fingerprints = get_fingerprints()

    rec_coords, rec_types, parent_coords, parent_types, conn, lig, fg = get_structures(
        receptor_path=receptor,
        ligand_path=ligand,
        cname=cname,
        rname=rname
    )

    batch = generate_grids(
        num_grid,
        model._args,
        rec_coords,
        rec_types,
        parent_coords,
        parent_types,
        conn,
        device
    )

    scores = get_predictions(model, batch, f_smiles, f_fingerprints)

    if top_k != -1:
        scores = scores[:top_k]

    gen_output(output, scores)
    
    return fg
# run_deepfrag('cpu', 10, './lib/pdb/pt.pdb','./lib/ligands/bms_200_1.pdb','N1','C24',-1,'branch.csv')
