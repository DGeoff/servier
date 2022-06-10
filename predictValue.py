import torch
import feature_extractor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import torch.nn as nn
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size) 
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)        
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
     
    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.ln3(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_out(out)
        return out


model = Net(4096,5,0.6,1)
model.load_state_dict(torch.load('mymodel.pth'))

scaler = StandardScaler()
feature_select = VarianceThreshold(threshold=0.05)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict_smiles(smiles):
    fp =feature_extractor.mol2fp(Chem.MolFromSmiles(smiles)).reshape(1,-1)
    fp_filtered = feature_select.transform(fp)
    fp_tensor = torch.tensor(fp, device=device).float()
    prediction = model(fp_tensor)
    p1 = scaler.inverse_transform(prediction.cpu().detach().numpy())
    return p1[0][0]

