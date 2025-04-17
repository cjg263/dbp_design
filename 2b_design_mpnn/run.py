import argparse

def main(args,paths_dict) -> None:
    """
    Inference function
    """
    import copy
    import json
    import os.path
    import random
    import sys
    import io
    import numpy as np
    import torch
    
    sys.path.append( paths_dict['ligand_mpnn_path'])
    from data_utils import (
        alphabet,
        element_dict_rev,
        featurize,
        get_score,
        get_seq_rec,
        #parse_PDB,
        restype_1to3,
        restype_int_to_str,
        restype_str_to_int,
        #write_full_PDB,
    )
    from model_utils import ProteinMPNN
    from prody import writePDB, writePDBStream, parsePDB, parsePDBStream, AtomGroup
    from sc_utils import Packer, pack_side_chains
    
    def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
        """
        protein_atoms: prody atom group
        CA_dict: mapping between chain_residue_idx_icodes and integers
        atom_name: atom to be parsed; e.g. CA
        """
        atom_atoms = protein_atoms.select(f"name {atom_name}")

        if atom_atoms != None:
            atom_coords = atom_atoms.getCoords()
            atom_resnums = atom_atoms.getResnums()
            atom_chain_ids = atom_atoms.getChids()
            atom_icodes = atom_atoms.getIcodes()

        atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
        atom_coords_m = np.zeros([len(CA_dict)], np.int32)
        if atom_atoms != None:
            for i in range(len(atom_resnums)):
                code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
                if code in list(CA_dict):
                    atom_coords_[CA_dict[code], :] = atom_coords[i]
                    atom_coords_m[CA_dict[code]] = 1
        return atom_coords_, atom_coords_m
    
    def parse_PDB(
        input_path: str,
        device: str = "cpu",
        chains: list = [],
        parse_all_atoms: bool = False,
        parse_atoms_with_zero_occupancy: bool = False
    ):
        """
        input_path : path for the input PDB
        device: device for the torch.Tensor
        chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
        parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
        parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
        """
        element_list = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mb",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Cn",
            "Uut",
            "Fl",
            "Uup",
            "Lv",
            "Uus",
            "Uuo",
        ]
        element_list = [item.upper() for item in element_list]
        element_dict = dict(zip(element_list, range(1, len(element_list))))
        restype_3to1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        restype_STRtoINT = {
            "A": 0,
            "C": 1,
            "D": 2,
            "E": 3,
            "F": 4,
            "G": 5,
            "H": 6,
            "I": 7,
            "K": 8,
            "L": 9,
            "M": 10,
            "N": 11,
            "P": 12,
            "Q": 13,
            "R": 14,
            "S": 15,
            "T": 16,
            "V": 17,
            "W": 18,
            "Y": 19,
            "X": 20,
        }

        atom_order = {
            "N": 0,
            "CA": 1,
            "C": 2,
            "CB": 3,
            "O": 4,
            "CG": 5,
            "CG1": 6,
            "CG2": 7,
            "OG": 8,
            "OG1": 9,
            "SG": 10,
            "CD": 11,
            "CD1": 12,
            "CD2": 13,
            "ND1": 14,
            "ND2": 15,
            "OD1": 16,
            "OD2": 17,
            "SD": 18,
            "CE": 19,
            "CE1": 20,
            "CE2": 21,
            "CE3": 22,
            "NE": 23,
            "NE1": 24,
            "NE2": 25,
            "OE1": 26,
            "OE2": 27,
            "CH2": 28,
            "NH1": 29,
            "NH2": 30,
            "OH": 31,
            "CZ": 32,
            "CZ2": 33,
            "CZ3": 34,
            "NZ": 35,
            "OXT": 36,
        }

        if not parse_all_atoms:
            atom_types = ["N", "CA", "C", "O"]
        else:
            atom_types = [
                "N",
                "CA",
                "C",
                "CB",
                "O",
                "CG",
                "CG1",
                "CG2",
                "OG",
                "OG1",
                "SG",
                "CD",
                "CD1",
                "CD2",
                "ND1",
                "ND2",
                "OD1",
                "OD2",
                "SD",
                "CE",
                "CE1",
                "CE2",
                "CE3",
                "NE",
                "NE1",
                "NE2",
                "OE1",
                "OE2",
                "CH2",
                "NH1",
                "NH2",
                "OH",
                "CZ",
                "CZ2",
                "CZ3",
                "NZ",
            ]

        if type(input_path) == str:
            atoms = parsePDB(input_path)   
        else:
            atoms = parsePDBStream(input_path)   
            
        if not parse_atoms_with_zero_occupancy:
            atoms = atoms.select("occupancy > 0")
        if chains:
            str_out = ""
            for item in chains:
                str_out += " chain " + item + " or"
            atoms = atoms.select(str_out[1:-3])

        protein_atoms = atoms.select("protein")
        backbone = protein_atoms.select("backbone")
        other_atoms = atoms.select("not protein and not water")
        water_atoms = atoms.select("water")

        CA_atoms = protein_atoms.select("name CA")
        CA_resnums = CA_atoms.getResnums()
        CA_chain_ids = CA_atoms.getChids()
        CA_icodes = CA_atoms.getIcodes()

        CA_dict = {}
        for i in range(len(CA_resnums)):
            code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]
            CA_dict[code] = i

        xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
        xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
        for atom_name in atom_types:
            xyz, xyz_m = get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
            xyz_37[:, atom_order[atom_name], :] = xyz
            xyz_37_m[:, atom_order[atom_name]] = xyz_m

        N = xyz_37[:, atom_order["N"], :]
        CA = xyz_37[:, atom_order["CA"], :]
        C = xyz_37[:, atom_order["C"], :]
        O = xyz_37[:, atom_order["O"], :]

        N_m = xyz_37_m[:, atom_order["N"]]
        CA_m = xyz_37_m[:, atom_order["CA"]]
        C_m = xyz_37_m[:, atom_order["C"]]
        O_m = xyz_37_m[:, atom_order["O"]]

        mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

        b = CA - N
        c = C - CA
        a = np.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
        R_idx = np.array(CA_resnums, dtype=np.int32)
        S = CA_atoms.getResnames()
        S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
        S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)
        X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

        try:
            Y = np.array(other_atoms.getCoords(), dtype=np.float32)
            Y_t = list(other_atoms.getElements())
            Y_t = np.array(
                [
                    element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                    for y_t in Y_t
                ],
                dtype=np.int32,
            )
            Y_m = (Y_t != 1) * (Y_t != 0)

            Y = Y[Y_m, :]
            Y_t = Y_t[Y_m]
            Y_m = Y_m[Y_m]
        except:
            Y = np.zeros([1, 3], np.float32)
            Y_t = np.zeros([1], np.int32)
            Y_m = np.zeros([1], np.int32)

        output_dict = {}
        output_dict["X"] = torch.tensor(X, device=device, dtype=torch.float32)
        output_dict["mask"] = torch.tensor(mask, device=device, dtype=torch.int32)
        output_dict["Y"] = torch.tensor(Y, device=device, dtype=torch.float32)
        output_dict["Y_t"] = torch.tensor(Y_t, device=device, dtype=torch.int32)
        output_dict["Y_m"] = torch.tensor(Y_m, device=device, dtype=torch.int32)

        output_dict["R_idx"] = torch.tensor(R_idx, device=device, dtype=torch.int32)
        output_dict["chain_labels"] = torch.tensor(
            chain_labels, device=device, dtype=torch.int32
        )

        output_dict["chain_letters"] = CA_chain_ids

        mask_c = []
        chain_list = list(set(output_dict["chain_letters"]))
        chain_list.sort()
        for chain in chain_list:
            mask_c.append(
                torch.tensor(
                    [chain == item for item in output_dict["chain_letters"]],
                    device=device,
                    dtype=bool,
                )
            )

        output_dict["mask_c"] = mask_c
        output_dict["chain_list"] = chain_list

        output_dict["S"] = torch.tensor(S, device=device, dtype=torch.int32)

        output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
        output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

        return output_dict, backbone, other_atoms, CA_icodes, water_atoms

    def write_PDB(save_file, 
                  atoms):
        """
        save_file: path or file object where PDB will be written to
        atoms: prody atoms to save
        """
        if type(save_file) == str:
            writePDB(save_file, atoms)
        else:
            writePDBStream(save_file, atoms)
            
    def write_full_PDB(save_file, 
                       X: np.ndarray, 
                       X_m: np.ndarray, 
                       b_factors: np.ndarray,
                       R_idx: np.ndarray, 
                       chain_letters: np.ndarray, 
                       S: np.ndarray, 
                       other_atoms=None, 
                       icodes=None,
                       force_hetatm=False):
        """
        save_file : path or file object where PDB will be written to
        X : protein atom xyz coordinates shape=[length, 14, 3]
        X_m : protein atom mask shape=[length, 14]
        b_factors: shape=[length, 14]
        R_idx: protein residue indices shape=[length]
        chain_letters: protein chain letters shape=[length]
        S : protein amino acid sequence shape=[length]
        other_atoms: other atoms parsed by prody
        icodes: a list of insertion codes for the PDB; e.g. antibody loops
        """

        restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL', 'X': 'UNK'}
        restype_INTtoSTR = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X'}
        restype_name_to_atom14_names = {'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''], 
                                        'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''], 
                                        'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''], 
                                        'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''], 
                                        'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''], 
                                        'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''], 
                                        'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''], 
                                        'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''], 
                                        'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''], 
                                        'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''], 
                                        'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''], 
                                        'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''], 
                                        'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''], 
                                        'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''], 
                                        'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''], 
                                        'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''], 
                                        'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''], 
                                        'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'], 
                                        'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''], 
                                        'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''], 
                                        'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}

        S_str = [restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]]
        
        X_list = []
        b_factor_list = []
        atom_name_list = []
        element_name_list = []
        residue_name_list = []
        residue_number_list = []
        chain_id_list = []
        icodes_list = []
        for i, AA in enumerate(S_str):
            sel = (X_m[i].astype(np.int32) == 1)
            total = np.sum(sel)
            tmp = np.array(restype_name_to_atom14_names[AA])[sel]
            X_list.append(X[i][sel])
            b_factor_list.append(b_factors[i][sel])
            atom_name_list.append(tmp)
            element_name_list += [AA[:1] for AA in list(tmp)]
            residue_name_list += total*[AA]
            residue_number_list += total*[R_idx[i]]
            chain_id_list += total*[chain_letters[i]]
            icodes_list += total*[icodes[i]]

        X_stack = np.concatenate(X_list, 0)
        b_factor_stack = np.concatenate(b_factor_list, 0)
        atom_name_stack = np.concatenate(atom_name_list, 0)
        
        protein = AtomGroup()
        protein.setCoords(X_stack)
        protein.setBetas(b_factor_stack)
        protein.setNames(atom_name_stack)
        protein.setResnames(residue_name_list)
        protein.setElements(element_name_list)
        protein.setOccupancies(np.ones([X_stack.shape[0]]))
        protein.setResnums(residue_number_list)
        protein.setChids(chain_id_list)
        protein.setIcodes(icodes_list)
        
        if other_atoms:
            other_atoms_g = AtomGroup()
            other_atoms_g.setCoords(other_atoms.getCoords())
            other_atoms_g.setNames(other_atoms.getNames())
            other_atoms_g.setResnames(other_atoms.getResnames())
            other_atoms_g.setElements(other_atoms.getElements())
            other_atoms_g.setOccupancies(other_atoms.getOccupancies())
            other_atoms_g.setResnums(other_atoms.getResnums())
            other_atoms_g.setChids(other_atoms.getChids())
            if force_hetatm:
                other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
            if type(save_file) == str:
                writePDB(save_file, protein + other_atoms_g)
            else:
                writePDBStream(save_file, protein + other_atoms_g)
        else:
            if type(save_file) == str:
                writePDB(save_file, protein)
            else:
                writePDBStream(save_file, protein)

    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    if not args.return_output_no_save:
        folder_for_outputs = args.out_folder
        base_folder = folder_for_outputs
        if base_folder[-1] != "/":
            base_folder = base_folder + "/"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder, exist_ok=True)
        if not os.path.exists(base_folder + "seqs"):
            os.makedirs(base_folder + "seqs", exist_ok=True)
        if not os.path.exists(base_folder + "backbones"):
            os.makedirs(base_folder + "backbones", exist_ok=True)
        if not os.path.exists(base_folder + "packed"):
            os.makedirs(base_folder + "packed", exist_ok=True)
        if args.save_stats:
            if not os.path.exists(base_folder + "stats"):
                os.makedirs(base_folder + "stats", exist_ok=True)
                
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.pack_side_chains:
        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=device,
            num_mix=3,
        )

        checkpoint_sc = torch.load(args.checkpoint_path_sc, map_location=device)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(device)
        model_sc.eval()

    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
        pdbs = copy.copy(pdb_files)
    elif args.pdb_path:
        pdb_files = [args.pdb_path]
        pdbs = copy.copy(pdb_files)
    else:
        pdb_file_input = io.StringIO(args.pdb_input_as_string)
        pdb_files = [pdb_file_input]

        pdbs = [args.pdb_string_name]
        
    pdb_to_pdb_files = dict(zip(pdbs, pdb_files))

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdbs:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
            redesigned_residues_multi = json.load(fh)
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdbs:
            redesigned_residues_multi[pdb] = redesigned_residues

    bias_AA = torch.zeros([21], device=device, dtype=torch.float32)
    if args.bias_AA:
        tmp = [item.split(":") for item in args.bias_AA.split(",")]
        a1 = [b[0] for b in tmp]
        a2 = [float(b[1]) for b in tmp]
        for i, AA in enumerate(a1):
            bias_AA[restype_str_to_int[AA]] = a2[i]

    if args.bias_AA_per_residue_multi:
        with open(args.bias_AA_per_residue_multi, "r") as fh:
            bias_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": {"G": 1.1}}}
    else:
        if args.bias_AA_per_residue:
            with open(args.bias_AA_per_residue, "r") as fh:
                bias_AA_per_residue = json.load(fh)  # {"A12": {"G": 1.1}}
            bias_AA_per_residue_multi = {}
            for pdb in pdbs:
                bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    if args.omit_AA_per_residue_multi:
        with open(args.omit_AA_per_residue_multi, "r") as fh:
            omit_AA_per_residue_multi = json.load(
                fh
            )  # {"pdb_path" : {"A12": "PQR", "A13": "QS"}}
    else:
        if args.omit_AA_per_residue:
            with open(args.omit_AA_per_residue, "r") as fh:
                omit_AA_per_residue = json.load(fh)  # {"A12": "PG"}
            omit_AA_per_residue_multi = {}
            for pdb in pdbs:
                omit_AA_per_residue_multi[pdb] = omit_AA_per_residue
    omit_AA_list = args.omit_AA
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=device,
    )
    
    if args.return_output_no_save:
        return_dict = dict()

    # loop over PDB paths
    for pdb in pdbs:
        
        if args.return_output_no_save:
            return_dict[pdb] = dict()
            
            
        if args.verbose:
            print("Designing protein from this path:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
            args.pack_side_chains and not args.repack_everything
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb_to_pdb_files[pdb],
            device=device,
            chains=args.parse_these_chains_only,
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
        )
        
        if not args.pdb_path_multi and not args.pdb_path:
            pdb_to_pdb_files[pdb].close()
            
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.bias_AA_per_residue_multi or args.bias_AA_per_residue:
            bias_dict = bias_AA_per_residue_multi[pdb]
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=device, dtype=torch.float32
        )
        if args.omit_AA_per_residue_multi or args.omit_AA_per_residue:
            omit_dict = omit_AA_per_residue_multi[pdb]
            for residue_name, v1 in omit_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid in v1:
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            omit_AA_per_residue[i1, j1] = 1.0

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )
        if type(args.chains_to_design) == str:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        # specify linking weights
        if args.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in args.symmetry_weights.split("|")
            ]
        else:
            symmetry_weights = [[]]

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            symmetry_weights = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    # for i, atom_type in enumerate(atom_types):
                    #     print(
                    #         f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                    #     )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = args.temperature
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights

            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                output_dict = model.sample(feature_dict)

                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if args.model_type == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [args.fasta_seq_separation]
            seq_out_str = "".join(seq_out_str)[:-1]

            if not args.return_output_no_save:
                output_fasta = base_folder + "/seqs/" + name + args.file_ending + ".fa"
                output_backbones = base_folder + "/backbones/"
                output_packed = base_folder + "/packed/"
                output_stats_path = base_folder + "stats/" + name + args.file_ending + ".pt"

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = seed
            out_dict["temperature"] = args.temperature
            if args.save_stats:
                if not args.return_output_no_save:
                    torch.save(out_dict, output_stats_path)

            if args.pack_side_chains:
                if args.verbose:
                    print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=args.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = args.batch_size
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(args.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(args.number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            model_sc,
                            args.sc_num_denoising_steps,
                            args.sc_num_samples,
                            args.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])

                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)

                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)
            
            if not args.return_output_no_save:
                with open(output_fasta, "w") as f:
                    f.write(
                        ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
                            name,
                            args.temperature,
                            seed,
                            torch.sum(rec_mask).cpu().numpy(),
                            torch.sum(combined_mask[:1]).cpu().numpy(),
                            bool(args.ligand_mpnn_use_atom_context),
                            float(args.ligand_mpnn_cutoff_for_score),
                            args.batch_size,
                            args.number_of_batches,
                            checkpoint_path,
                            seq_out_str,
                        )
                    )
            else:
                fasta_header_dict = dict()

                fasta_header_dict["name"] = name
                fasta_header_dict["T"] = args.temperature
                fasta_header_dict["seed"] = seed
                fasta_header_dict["num_res"] = torch.sum(rec_mask).cpu().numpy()
                fasta_header_dict["num_ligand_res"] = torch.sum(combined_mask[:1]).cpu().numpy()
                fasta_header_dict["use_ligand_context"] = bool(args.ligand_mpnn_use_atom_context)
                fasta_header_dict["batch_size"] = args.batch_size
                fasta_header_dict["number_of_batches"] = args.number_of_batches
                fasta_header_dict["model_path"] = checkpoint_path
                fasta_header_dict["input_sequence"] = seq_out_str

                return_dict[pdb]["fasta_header"] = fasta_header_dict

                designs_dict = dict()
                
                
                
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix
                if not args.zero_indexed:
                    ix_suffix += 1
                    
                
                if args.return_output_no_save:
                    designs_dict[ix_suffix] = dict()
                    
                    
                seq_rec_print = np.format_float_positional(
                    rec_stack[ix].cpu().numpy(), unique=False, precision=4
                )
                loss_np = np.format_float_positional(
                    np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
                )
                loss_XY_np = np.format_float_positional(
                    np.exp(-loss_XY_stack[ix].cpu().numpy()),
                    unique=False,
                    precision=4,
                )
                seq = "".join(
                    [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                )

                # write new sequences into PDB with backbone coordinates
                seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                    None,
                ].repeat(4, 1)
                bfactor_prody = (
                    loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                )
                backbone.setResnames(seq_prody)
                backbone.setBetas(
                    np.exp(-bfactor_prody)
                    * (bfactor_prody > 0.01).astype(np.float32)
                )
                
                if not args.return_output_no_save:
                    backbone_output_file = output_backbones+name+'_'+str(ix_suffix)+".pdb"+ args.file_ending
                else:
                    backbone_output_file = io.StringIO()

                if other_atoms:
                    write_PDB(backbone_output_file, backbone+other_atoms)
                else:
                    write_PDB(backbone_output_file, backbone)
                    
                    
                if args.return_output_no_save:
                    designs_dict[ix_suffix]["backbone_PDB_string"] = backbone_output_file.getvalue()
                    backbone_output_file.close()
                    
                # write full PDB files
                if args.pack_side_chains:
                    for c_pack in range(args.number_of_packs_per_design):
                        X_stack = X_stack_list[c_pack]
                        X_m_stack = X_m_stack_list[c_pack]
                        b_factor_stack = b_factor_stack_list[c_pack]
                        
                        if not args.return_output_no_save:
                            packed_output_file = output_packed + name + args.packed_suffix + "_" + str(ix_suffix) + "_" + str(c_pack + 1) + args.file_ending + ".pdb"
                        else:
                            packed_output_file = io.StringIO()
                            
                        write_full_PDB(
                            packed_output_file,
                            X_stack[ix].cpu().numpy(),
                            X_m_stack[ix].cpu().numpy(),
                            b_factor_stack[ix].cpu().numpy(),
                            feature_dict["R_idx_original"][0].cpu().numpy(),
                            protein_dict["chain_letters"],
                            S_stack[ix].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                            force_hetatm=args.force_hetatm,
                        )
                        
                        if args.return_output_no_save:
                            designs_dict[ix_suffix]["packed_PDB_string"] = packed_output_file.getvalue()
                            packed_output_file.close()
                            
                #-----
                # -----

                # write fasta lines
                seq_np = np.array(list(seq))
                seq_out_str = []
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [args.fasta_seq_separation]
                seq_out_str = "".join(seq_out_str)[:-1]
                if not args.return_output_no_save:
                    f.write(
                        ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
                            name,
                            ix_suffix,
                            args.temperature,
                            seed,
                            loss_np,
                            loss_XY_np,
                            seq_rec_print,
                            seq_out_str,
                        )
                    )
                else:
                    designs_dict[ix_suffix]["name"] = name
                    designs_dict[ix_suffix]["id"] = ix_suffix
                    designs_dict[ix_suffix]["T"] = args.temperature
                    designs_dict[ix_suffix]["seed"] = seed
                    designs_dict[ix_suffix]["overall_confidence"] = loss_np
                    designs_dict[ix_suffix]["ligand_confidence"] = loss_XY_np
                    designs_dict[ix_suffix]["seq_rec"] = seq_rec_print
                    designs_dict[ix_suffix]["sequence"] = seq_out_str
                    
            if not args.return_output_no_save:
                f.close()
            else:
                return_dict[pdb]["designs"] = designs_dict
                
    if args.return_output_no_save:
        return return_dict
            



def get_argument_parser():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="./model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="./model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="./model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="./model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="./model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--fasta_seq_separation",
        type=str,
        default=":",
        help="Symbol to use between sequences from different chains",
    )
    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--bias_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'A:-1.024,P:2.34,C:-12.34'",
    )
    argparser.add_argument(
        "--bias_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}",
    )
    argparser.add_argument(
        "--bias_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': {'G': -0.3, 'C': -2.0, 'H': 0.8}, 'A13': {'G': -1.3}}}",
    )

    argparser.add_argument(
        "--omit_AA",
        type=str,
        default="",
        help="Bias generation of amino acids, e.g. 'ACG'",
    )
    argparser.add_argument(
        "--omit_AA_per_residue",
        type=str,
        default="",
        help="Path to json mapping of bias {'A12': 'APQ', 'A13': 'QST'}",
    )
    argparser.add_argument(
        "--omit_AA_per_residue_multi",
        type=str,
        default="",
        help="Path to json mapping of bias {'pdb_path': {'A12': 'QSPC', 'A13': 'AGE'}}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    argparser.add_argument(
        "--symmetry_weights",
        type=str,
        default="",
        help="Add weights that match symmetry_residues, e.g. '1.01,1.0,1.0|-1.0,2.0|2.0,2.3'",
    )
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output sequences, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature to sample sequences.",
    )
    argparser.add_argument(
        "--save_stats", type=int, default=0, help="Save output statistics"
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )
    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )
    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )
    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default=None,
        help="Specify which chains to redesign, all others will be kept fixed.",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'ABCF'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--pack_side_chains",
        type=int,
        default=0,
        help="1 - to run side chain packer, 0 - do not run it",
    )

    argparser.add_argument(
        "--checkpoint_path_sc",
        type=str,
        default="./model_params/ligandmpnn_sc_v_32_002_16.pt",
        help="Path to model weights.",
    )

    argparser.add_argument(
        "--number_of_packs_per_design",
        type=int,
        default=4,
        help="Number of independent side chain packing samples to return per design",
    )

    argparser.add_argument(
        "--sc_num_denoising_steps",
        type=int,
        default=3,
        help="Number of denoising/recycling steps to make for side chain packing",
    )

    argparser.add_argument(
        "--sc_num_samples",
        type=int,
        default=16,
        help="Number of samples to draw from a mixture distribution and then take a sample with the highest likelihood.",
    )

    argparser.add_argument(
        "--repack_everything",
        type=int,
        default=0,
        help="1 - repacks side chains of all residues including the fixed ones; 0 - keeps the side chains fixed for fixed residues",
    )

    argparser.add_argument(
        "--force_hetatm",
        type=int,
        default=0,
        help="To force ligand atoms to be written as HETATM to PDB file after packing.",
    )

    argparser.add_argument(
        "--packed_suffix",
        type=str,
        default="_packed",
        help="Suffix for packed PDB paths",
    )

    argparser.add_argument(
        "--pack_with_ligand_context",
        type=int,
        default=1,
        help="1-pack side chains using ligand context, 0 - do not use it.",
    )
    
    argparser.add_argument(
        "--pdb_input_as_string", 
        type=str, default="", 
        help="If a string is passed and pdb_path is the empty string, inference will use this string for the pdb data instead"
    )
    
    argparser.add_argument(
        "--pdb_string_name", 
        type=str, default="", 
        help="If pdb_input_as_string is utilizied, this should be a unique, short identifier of the pdb string (PDB ID would work)"
    )
    
    argparser.add_argument(
        "--return_output_no_save", 
        type=int, default=1, 
        help="1 - Instead of saving fasta, backbone pdbs, side-chain packed pdbs, return an dictionary containing this information instead, 0 - normal saving of outputs"
    )

    return argparser

if __name__ == "__main__":
    argparser = get_argument_parser()
    args = argparser.parse_args()    
    main(args)   