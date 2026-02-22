import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


Z_TO_SYM = {1:"H", 6:"C", 7:"N", 8:"O", 9:"F"}
SYM_LIST = ["H","C","N","O","F"]
SYM_TO_IDX = {s:i for i,s in enumerate(SYM_LIST)}


def read_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def merge_by_name(dftb_rows, dft_rows):
    dftb = {r["name"]: r for r in dftb_rows if "name" in r and "mulliken_dftb3" in r and "z" in r}
    dft  = {r["name"]: r for r in dft_rows  if "name" in r and "mbis_dft" in r and "z" in r}
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


def mae(x):
    x=np.asarray(x,float)
    return float(np.mean(np.abs(x))) if len(x) else float("nan")

def rmse(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x))) if len(x) else float("nan")


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


class MolDatasetNoGeom(Dataset):
    def __init__(self, records):
        self.records=records

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r=self.records[idx]
        z=np.asarray(r["z"],int)
        q_dftb=np.asarray(r["q_dftb"],float)
        q_mbis=np.asarray(r["q_mbis"],float)

        oh = one_hot_elements(z)
        x  = np.concatenate([oh, q_dftb[:,None]], axis=1)

        return {
            "name": r["name"],
            "z": torch.tensor(z, dtype=torch.long),
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(q_mbis, dtype=torch.float32),
        }


def collate_mols(batch): return batch


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


def apply_charge_conservation(q_pred, q_true):
    N=q_pred.shape[0]
    delta = (q_true.sum() - q_pred.sum()) / float(N)
    return q_pred + delta


def run_epoch(model, loader, device, optimizer=None, lambda_cons=0.5, use_cons_loss=True):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss=0.0
    n_atoms=0

    for batch in loader:
        if train_mode: optimizer.zero_grad()
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
            atoms_in_batch=sum(m["y"].numel() for m in batch)
            total_loss += float(loss_batch)*atoms_in_batch
            n_atoms += atoms_in_batch

    return total_loss/max(1,n_atoms)


@torch.no_grad()
def evaluate(model, loader, device, do_postprocess=True):
    model.eval()

    err_all=[]; err_all_post=[]
    err_by_el=defaultdict(list); err_by_el_post=defaultdict(list)
    mol_sum_err=[]; mol_sum_err_post=[]
    preds_out=[]

    for batch in loader:
        for mol in batch:
            name=mol["name"]
            z=mol["z"].cpu().numpy().astype(int)
            x=mol["x"].to(device)
            y=mol["y"].to(device)

            pred=model(x)
            pred_post=apply_charge_conservation(pred,y) if do_postprocess else pred

            e=(pred-y).cpu().numpy()
            e_post=(pred_post-y).cpu().numpy()

            err_all.extend(e.tolist())
            err_all_post.extend(e_post.tolist())

            for Zi,ei,eip in zip(z,e,e_post):
                sym=Z_TO_SYM[int(Zi)]
                err_by_el[sym].append(float(ei))
                err_by_el_post[sym].append(float(eip))

            mol_sum_err.append(float(pred.sum().cpu()-y.sum().cpu()))
            mol_sum_err_post.append(float(pred_post.sum().cpu()-y.sum().cpu()))

            preds_out.append({
                "name": name,
                "z": z.tolist(),
                "q_true_mbis": y.cpu().numpy().tolist(),
                "q_pred_raw": pred.cpu().numpy().tolist(),
                "q_pred_cons": pred_post.cpu().numpy().tolist(),
            })

    metrics={
        "atomwise_MAE_raw": mae(err_all),
        "atomwise_MAE_cons": mae(err_all_post),
        "mol_charge_MAE_raw": mae(mol_sum_err),
        "mol_charge_MAE_cons": mae(mol_sum_err_post),
    }

    return metrics, preds_out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dftb", required=True)
    ap.add_argument("--dft", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--outdir", default="ablation_nogeom_results")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_mols", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_cons", type=float, default=0.5)
    ap.add_argument("--no_cons_loss", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args=ap.parse_args()

    outdir=Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    split_dir=Path(args.split_dir)

    train_set = load_split_file(split_dir/"train.txt")
    val_set   = load_split_file(split_dir/"val.txt")
    test_set  = load_split_file(split_dir/"test.txt")

    print(f"[INFO] Loaded split from {split_dir}")
    print(f"Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}")

    merged=merge_by_name(read_jsonl(args.dftb), read_jsonl(args.dft))
    if len(merged)==0:
        raise RuntimeError("No merged records. Check inputs.")

    name_to_record={r["name"]:r for r in merged}

    train=[name_to_record[nm] for nm in train_set]
    val=[name_to_record[nm] for nm in val_set]
    test=[name_to_record[nm] for nm in test_set]

    train_ds=MolDatasetNoGeom(train)
    in_dim=int(train_ds[0]["x"].shape[1])
    print(f"[INFO] in_dim={in_dim} (expected 6)")

    train_loader=DataLoader(train_ds, batch_size=args.batch_mols, shuffle=True, collate_fn=collate_mols)
    val_loader=DataLoader(MolDatasetNoGeom(val), batch_size=args.batch_mols, shuffle=False, collate_fn=collate_mols)
    test_loader=DataLoader(MolDatasetNoGeom(test), batch_size=args.batch_mols, shuffle=False, collate_fn=collate_mols)

    device=torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model=AtomMLP(in_dim=in_dim, hidden=args.hidden, depth=args.depth, dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val=float("inf"); best_state=None

    for epoch in range(1, args.epochs+1):
        tr=run_epoch(model, train_loader, device, optimizer=opt,
                     lambda_cons=args.lambda_cons,
                     use_cons_loss=(not args.no_cons_loss))
        va=run_epoch(model, val_loader, device, optimizer=None,
                     lambda_cons=args.lambda_cons,
                     use_cons_loss=(not args.no_cons_loss))
        if va < best_val:
            best_val=va
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%10==0:
            print(f"Epoch {epoch:3d}/{args.epochs}  train~{tr:.6e}  val~{va:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, preds = evaluate(model, test_loader, device)

    (outdir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "in_dim": in_dim},
               outdir/"model.pt")

    print("\nSaved to:", outdir)
    print(json.dumps(metrics, indent=2))


if __name__=="__main__":
    main()
