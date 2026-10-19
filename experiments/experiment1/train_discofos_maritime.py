#!/usr/bin/env python3
"""
Fine-tune DiscoFOS (ICCV 2023) on the maritime library, initialized from the
released S-FOSD student (sd.pth) and using the released teacher (teacher_sfosd.pth)
for the composite-feature distillation loss.

Reuses the official TrainDataset (compositing pipeline) and the official loss terms
(triplet + BCE classify + distillation), driven by a minimal single-GPU loop instead
of the repo's DDP launcher. Losses exactly mirror train/train.py backward().
"""
import os
import sys
import time
import argparse
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange, repeat

REPO = "/path/to/sealect_root"
DISCOFOS = os.path.join(REPO, "experiments/baselines/DiscoFOS")
sys.path.insert(0, DISCOFOS)
os.chdir(DISCOFOS)  # dataset.datasets anchors data/ at PROJ_dir

from dataset.datasets import TrainDataset          # noqa: E402
from network.networks import StudentModel, SingleScaleD  # noqa: E402
from utils import loss_function                     # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_STUDENT = os.path.join(DISCOFOS, "checkpoints/sd.pth")
CKPT_TEACHER = os.path.join(DISCOFOS, "checkpoints/teacher/teacher_sfosd.pth")


def make_cfg(bs, workers, num_neg=6, encoder_classify=False):
    # merges config/config_sfosd.yaml fields needed by TrainDataset + StudentModel
    return SimpleNamespace(
        config_name="maritime", backbone="vgg19", feature_dim=512,
        encoder_feature="gap", encoder_classify=encoder_classify, distill_type="roiconcat",
        triplet_margin=0.1, image_size=224, fill_pixel=128, pad_pixel=255,
        padding_max=0, img_aug=False, num_positive=1, num_negative=num_neg,
        num_classes=1, batch_size=bs, num_workers=workers,
        distill_weight=1.0, classify_weight=1.0,
    )


class SafeTrainDataset(TrainDataset):
    """Skip the rare composite that fails (tiny/degenerate box) by resampling."""
    def __getitem__(self, i):
        for _ in range(8):
            try:
                return super().__getitem__(i)
            except Exception:
                i = (i * 1103515245 + 12345) % len(self)
        return super().__getitem__(0)


def set_input(sample):
    bg_im = sample["bg"].to(DEVICE)
    bs = bg_im.shape[0]
    pos_fg = sample["pos_fg"].to(DEVICE)
    neg_fg = sample["neg_fg"].to(DEVICE)
    num_pos, num_neg = pos_fg.shape[1], neg_fg.shape[1]
    num_fg = num_pos + num_neg
    all_fg = rearrange(torch.cat([pos_fg, neg_fg], 1), "b n c h w -> (b n) c h w")
    fg_label = torch.cat([torch.ones(bs, num_pos), torch.zeros(bs, num_neg)], 1).reshape(-1).to(DEVICE)
    all_scale = rearrange(torch.cat([sample["pos_scale_comp"], sample["neg_scale_comp"]], 1).to(DEVICE),
                          "b n c h w -> (b n) c h w")
    qbox = repeat(sample["query_box"].to(DEVICE), "(b 1) d -> (b n) d", n=num_fg)
    cbox = repeat(sample["crop_box"].to(DEVICE), "(b 1) d -> (b n) d", n=num_fg)
    repeat_bg = repeat(bg_im, "(b 1) c h w -> (b n) c h w", n=num_fg)
    return repeat_bg, all_fg, qbox, cbox, fg_label, all_scale, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--num_neg", type=int, default=6)
    ap.add_argument("--max_iters", type=int, default=0)
    ap.add_argument("--encoder_classify", action="store_true", help="auxiliary embedding classifier (anti-collapse)")
    ap.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--save_every", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(DISCOFOS, "checkpoints/maritime"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    cfg = make_cfg(args.bs, args.workers, args.num_neg, args.encoder_classify)
    student = StudentModel(cfg, loadweights=False).to(DEVICE)
    sd = torch.load(CKPT_STUDENT, map_location="cpu", weights_only=False)
    miss, unexp = student.load_state_dict(sd, strict=False)
    print(f"student init from sd.pth (missing {len(miss)}, unexpected {len(unexp)})", flush=True)
    teacher = SingleScaleD(False).to(DEVICE)
    teacher.load_state_dict(torch.load(CKPT_TEACHER, map_location="cpu", weights_only=False))
    teacher.eval()

    dataset = SafeTrainDataset(cfg)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)
    opt = optim.Adam(student.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp_dtype]
    use_amp = args.amp_dtype != "fp32"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_dtype == "fp16"))
    print(f"dataset {len(dataset)} pairs | {len(loader)} batches/epoch | lr {args.lr} bs {args.bs} "
          f"| amp={args.amp_dtype} encoder_classify={args.encoder_classify}", flush=True)

    for ep in range(args.epochs):
        student.train()
        t0 = time.time(); agg = {"trip": 0.0, "cls": 0.0, "kd": 0.0}; n = 0
        for bi, sample in enumerate(loader):
            repeat_bg, all_fg, qbox, cbox, fg_label, all_scale, bs = set_input(sample)
            opt.zero_grad()
            # memory-heavy encoders in low precision; losses in fp32 (BCE is not autocast-safe)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                _bg, _fg, _score = student(repeat_bg, all_fg, qbox, cbox, bs)
                with torch.no_grad():
                    scl_feat, _ = teacher(all_scale)
            student.fg_emb = student.fg_emb.float(); student.bg_emb = student.bg_emb.float()
            student.distill_feat = student.distill_feat.float(); student.score = student.score.float()
            if getattr(student, "aux_score", None) is not None:
                student.aux_score = student.aux_score.float()
            scl_feat = [scl_feat[0].float(), scl_feat[1].float()]
            l_trip = student.calcu_triplet_loss(fg_label, bs)
            l_kd = student.calcu_distillation_loss(scl_feat)
            l_cls = student.calcu_classify_loss(fg_label)
            loss = l_trip + cfg.distill_weight * l_kd + cfg.classify_weight * l_cls
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            agg["trip"] += float(l_trip); agg["cls"] += float(l_cls)
            agg["kd"] += float(l_kd) if torch.is_tensor(l_kd) else float(l_kd); n += 1
            if bi % 100 == 0 or (args.max_iters and bi == args.max_iters - 1):
                rate = (bi + 1) / (time.time() - t0)
                peak = torch.cuda.max_memory_allocated() / 1e9
                eta = (len(loader) - bi - 1) / max(rate, 1e-6) / 60
                print(f"  ep{ep} b{bi}/{len(loader)} trip {l_trip:.4f} cls {l_cls:.4f} kd {float(l_kd):.4f} | {rate:.2f} it/s peak {peak:.1f}GB eta {eta:.0f}min", flush=True)
            if args.smoke and bi >= 20:
                print("smoke ok"); return
            if args.max_iters and bi + 1 >= args.max_iters:
                path = os.path.join(args.out, "probe.pth")
                torch.save(student.state_dict(), path)
                print(f"probe done: {(bi+1)/(time.time()-t0):.2f} it/s, peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB, saved {path}"); return
        dt = time.time() - t0
        print(f"epoch {ep}: trip {agg['trip']/n:.4f} cls {agg['cls']/n:.4f} kd {agg['kd']/n:.4f} | {dt/60:.1f} min", flush=True)
        if (ep + 1) % args.save_every == 0 or ep == args.epochs - 1:
            path = os.path.join(args.out, f"epoch{ep+1}.pth")
            torch.save(student.state_dict(), path)
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
