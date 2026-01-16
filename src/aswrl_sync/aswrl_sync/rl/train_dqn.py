import argparse
import os
import time
import numpy as np
import torch

from aswrl_env import ASWRLEnv
from dqn import MLP, ReplayBuffer, Transition, dqn_update

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', type=str, default='multimodal', choices=['stable','jitter','multimodal','route_switch'])
    ap.add_argument('--steps', type=int, default=300000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--outdir', type=str, default='outputs')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')

    env = ASWRLEnv(scenario=args.scenario, seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q = MLP(obs_dim, act_dim).to(device)
    q_tgt = MLP(obs_dim, act_dim).to(device)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    rb = ReplayBuffer(cap=200000)

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.999995

    s, _ = env.reset(seed=args.seed)
    s = torch.tensor(s, dtype=torch.float32, device=device)

    last_save = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        # epsilon-greedy
        if np.random.rand() < eps:
            a = np.random.randint(act_dim)
        else:
            with torch.no_grad():
                a = int(torch.argmax(q(s.unsqueeze(0)), dim=1).item())

        s2, r, term, trunc, info = env.step(a)
        done = term or trunc
        s2t = torch.tensor(s2, dtype=torch.float32, device=device)

        rb.push(Transition(
            s=s.detach().cpu(),
            a=torch.tensor([a], dtype=torch.int64),
            r=torch.tensor([r], dtype=torch.float32),
            s2=s2t.detach().cpu(),
            d=torch.tensor([1.0 if done else 0.0], dtype=torch.float32),
        ))

        s = s2t if not done else torch.tensor(env.reset()[0], dtype=torch.float32, device=device)

        eps = max(eps_min, eps * eps_decay)

        # train
        if len(rb) >= args.batch:
            batch = rb.sample(args.batch)
            # move to device
            for t in batch:
                t.s = t.s.to(device)
                t.a = t.a.to(device)
                t.r = t.r.to(device)
                t.s2 = t.s2.to(device)
                t.d = t.d.to(device)
            loss = dqn_update(q, q_tgt, opt, batch, args.gamma)
            losses.append(loss)

        # target network update
        if step % 2000 == 0:
            q_tgt.load_state_dict(q.state_dict())

        # checkpoint
        if step % 10000 == 0 or (time.time() - last_save) > 600:
            ckpt = {
                "step": step,
                "scenario": args.scenario,
                "model": q.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }
            path = os.path.join(args.outdir, "dqn_latest.pt")
            torch.save(ckpt, path)
            last_save = time.time()

        if step % 5000 == 0:
            avg_loss = float(np.mean(losses[-1000:])) if losses else 0.0
            print(f"[{args.scenario}] step={step} eps={eps:.3f} avg_loss={avg_loss:.6f}")

    # final save
    ckpt = {"step": args.steps, "scenario": args.scenario, "model": q.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim}
    torch.save(ckpt, os.path.join(args.outdir, "dqn_final.pt"))
    print("Done. Saved outputs/dqn_final.pt")

if __name__ == '__main__':
    main()
