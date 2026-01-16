import argparse
import torch
from dqn import MLP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True, help='Path to dqn checkpoint (pt)')
    ap.add_argument('--out', type=str, default='policy.pt', help='Output torchscript file')
    ap.add_argument('--hidden', type=int, default=128)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    obs_dim = int(ckpt.get('obs_dim', 6))
    act_dim = int(ckpt.get('act_dim', 3))

    model = MLP(obs_dim, act_dim, hidden=args.hidden)
    model.load_state_dict(ckpt['model'])
    model.eval()

    example = torch.zeros((1, obs_dim), dtype=torch.float32)
    ts = torch.jit.trace(model, example)
    ts.save(args.out)
    print(f"Saved torchscript policy to {args.out}")

if __name__ == '__main__':
    main()
