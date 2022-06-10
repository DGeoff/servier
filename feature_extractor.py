from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
import numpy


def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )


def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=4096)
    ar = numpy.zeros((1,), dtype=numpy.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar