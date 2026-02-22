import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===============================
# Element definitions
# ===============================
Z_TO_SYM = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F"}
SYM_LIST = ["H","C","N","O","F"]
SYM_TO_IDX = {s:i for i,s in enumerate(SYM_LIST)}


# ===============================
# Utilities
# ===============================
def read_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def load_split_file(path):
    if not path.exists():
        raise RuntimeError(f"Missing split file: {path}")
    names=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                names.append(s)
    return set(names)


def merge_by_name(dftb_rows, dft_rows):
    dftb = {r["name"]: r for r in dftb_rows if "name" in r and "mulliken_dftb3" in r}
    dft  = {r["name"]: r for r in dft_rows  if "name" in r and "mbis_dft" in r}
    names = sorted(set(dftb.keys()) & set(dft.keys()))

    merged=[]
    for name in names:
        a=dftb[name]; b=dft[name]
        if a.get("z") != b.get("z"):
            continue
        z=a["z"]; qd=a["mulliken_dftb3"]; qm=b["mbis_dft"]
        if len(z)!=len(qd) or len(z)!=len(qm):
            continue
        merged.append({"name":name,"z":z,"q_dftb":qd,"q_mbis":qm})
    return merged


def mae(x):
    x=np.asarray(x,float)
    return float(np.mean(np.abs(x))) if len(x) else float("nan")


def rmse(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x))) if len(x) else float("nan")


# ===============================
# Feature construction
# ===============================
def one_hot_elements(z):
    z=np.asarray(z,int)
    N=len(z)
    oh=np.zeros((N, len(SYM_LIST)), float)
    for i,Zi in enumerate(z):
        sym=Z_TO_SYM.get(int(Zi))
        if sym is None:
            raise ValueError(f"Unexpected element Z={Zi}")
        oh[i, SYM_TO_IDX[sym]] = 1.0
    return oh


def build_geom_features(pos, z, K=8, cutoff=4.0):
    pos=np.asarray(pos,float)
    z=np.asarray(z,int)
    N=len(z)

    diff = pos[:,None,:] - pos[None,:,:]
    dmat = np.sqrt(np.sum(diff*diff, axis=-1))
    np.fill_diagonal(dmat, np.inf)

    sorted_d = np.sort(dmat, axis=1)
    knn = sorted_d[:, :min(K, max(0,N-1))]
    if knn.shape[1] < K:
        pad = np.full((N, K-knn.shape[1]), cutoff, float)
        knn = np.concatenate([knn, pad], axis=1)

    counts=np.zeros((N, len(SYM_LIST)), float)
    within = dmat <= cutoff
    for i in range(N):
        js = np.where(within[i])[0]
        for j in js:
            sym = Z_TO_SYM.get(int(z[j]), None)
            if sym is None:
                continue
            counts[i, SYM_TO_IDX[sym]] += 1.0

    feats = np.concatenate([knn, counts], axis=1)
    if feats.shape[1] != K + 5:
        raise RuntimeError(f"geom feature dim wrong: got {feats.shape[1]}")
    return feats


# ===============================
# Dataset
# ===============================
class MolDataset(Dataset):
    def __init__(self, records, pos_by_name, K=8, cutoff=4.0):
        self.records=records
        self.pos_by_name=pos_by_name
        self.K=K
        self.cutoff=cutoff

    def __len__(self): 
        return len(self.records)

    def __getitem__(self, idx):
        r=self.records[idx]
        name=r["name"]
        z=np.asarray(r["z"],int)
        pos=self.pos_by_name[name]
        q_dftb=np.asarray(r["q_dftb"],float)
        q_mbis=np.asarray(r["q_mbis"],float)

        oh = one_hot_elements(z)
        geom = build_geom_features(pos, z, self.K, self.cutoff)
        x = np.concatenate([oh, q_dftb[:,None], geom], axis=1)

        return {
            "name": name,
            "z": torch.tensor(z, dtype=torch.long),
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(q_mbis, dtype=torch.float32),
        }


def collate_mols(batch): 
    return batch


# ===============================
# Model
# ===============================
class AtomMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers=[]
        d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d=hidden
        layers += [nn.Linear(d,1)]
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x).squeeze(-1)


# ===============================
# Charge conservation
# ===============================
def apply_charge_conservation(q_pred, q_true):
    N=q_pred.shape[0]
    delta = (q_true.sum() - q_pred.sum()) / float(N)
    return q_pred + delta


# ===============================
# Training / Evaluation
# ===============================
def run_epoch(model, loader, device, optimizer=None, lambda_cons=0.5, use_cons_loss=True):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss=0.0
    n_atoms=0

    for batch in loader:
        if train_mode: 
            optimizer.zero_grad()

        losses=[]
        for mol in batch:
            x=mol["x"].to(device)
            y=mol["y"].to(device)
            pred=model(x)

            loss_atom=torch.mean((pred-y)**2)

            if use_cons_loss:
                sum_err=torch.sum(pred)-torch.sum(y)
                loss=loss_atom + lambda_cons*(sum_err**2)/(y.numel()**2)
            else:
                loss=loss_atom

            losses.append(loss)

        loss_batch=torch.stack(losses).mean()

        if train_mode:
            loss_batch.backward()
            optimizer.step()

        with torch.no_grad():
            atoms=sum(m["y"].numel() for m in batch)
            total_loss += float(loss_batch)*atoms
            n_atoms += atoms

    return total_loss/max(1,n_atoms)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    err_all=[]

    for batch in loader:
        for mol in batch:
            x=mol["x"].to(device)
            y=mol["y"].to(device)
            pred=model(x)
            e=(pred-y).cpu().numpy()
            err_all.extend(e.tolist())

    return {
        "atomwise_MAE": mae(err_all),
        "atomwise_RMSE": rmse(err_all),
    }


# ===============================
# Main
# ===============================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dftb", required=True)
    ap.add_argument("--dft", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--outdir", default="mlp_results")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_mols", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cpu", action="store_true")
    args=ap.parse_args()

    outdir=Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load splits
    split_dir=Path(args.split_dir)
    train_set=load_split_file(split_dir/"train.txt")
    val_set=load_split_file(split_dir/"val.txt")
    test_set=load_split_file(split_dir/"test.txt")

    print(f"Loaded split: train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # Merge data
    merged=merge_by_name(read_jsonl(args.dftb), read_jsonl(args.dft))
    name_to_record={r["name"]:r for r in merged}

    train=[name_to_record[n] for n in train_set]
    val=[name_to_record[n] for n in val_set]
    test=[name_to_record[n] for n in test_set]

    # Load coords
    need=train_set|val_set|test_set
    ds=load_dataset("lisn519010/QM9", split="full")
    pos_by_name={}
    for row in tqdm(ds, desc="Index QM9 coords"):
        nm=row.get("name")
        if nm in need:
            pos_by_name[nm]=np.array(row["pos"], float)

    # Dataset
    train_ds=MolDataset(train, pos_by_name, args.K, args.cutoff)
    val_ds=MolDataset(val, pos_by_name, args.K, args.cutoff)
    test_ds=MolDataset(test, pos_by_name, args.K, args.cutoff)

    train_loader=DataLoader(train_ds, batch_size=args.batch_mols, shuffle=True, collate_fn=collate_mols)
    val_loader=DataLoader(val_ds, batch_size=args.batch_mols, shuffle=False, collate_fn=collate_mols)
    test_loader=DataLoader(test_ds, batch_size=args.batch_mols, shuffle=False, collate_fn=collate_mols)

    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    in_dim=train_ds[0]["x"].shape[1]

    model=AtomMLP(in_dim, args.hidden, args.depth, args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val=float("inf")
    best_state=None

    for epoch in range(1,args.epochs+1):
        tr=run_epoch(model, train_loader, device, opt)
        va=run_epoch(model, val_loader, device)

        if va<best_val:
            best_val=va
            best_state=model.state_dict()

        if epoch==1 or epoch%10==0:
            print(f"Epoch {epoch}: train={tr:.6e} val={va:.6e}")

    model.load_state_dict(best_state)

    metrics=evaluate(model, test_loader, device)

    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    torch.save(model.state_dict(), outdir/"model.pt")

    print("\nFinal test metrics:")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":
    main()