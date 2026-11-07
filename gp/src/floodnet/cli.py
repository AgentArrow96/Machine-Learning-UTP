"""Command entrypoint. Heavy ML dependencies load only for relevant commands."""
import argparse
import json
import sys
from .common import ProjectError, config, environment, log


def main():
    parser = argparse.ArgumentParser(prog="floodnet", description="FloodNet: CPU SLIC/RF and CUDA U-Net, inside gp/")
    parser.add_argument("--config", help="Configuration JSON inside gp/; defaults to configs/base.json")
    subs = parser.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("doctor"); doctor.add_argument("--gpu-smoke", action="store_true")
    subs.add_parser("download")
    audit = subs.add_parser("audit"); audit.add_argument("--raw", default="data/raw")
    subs.add_parser("prepare"); subs.add_parser("features"); subs.add_parser("overlap")
    check = subs.add_parser("overlap-check", help="Verify one evaluation/train pair (threshold calibration)")
    check.add_argument("--eval", required=True); check.add_argument("--train", required=True)
    train = subs.add_parser("train")
    train.add_argument("--model", choices=("rf", "unet"), required=True)
    train.add_argument("--budget", type=int, choices=(25, 100), required=True)
    train.add_argument("--seed", type=int, help="Training seed; defaults to the first configured train seed")
    train.add_argument("--resume", action="store_true"); train.add_argument("--smoke", action="store_true")
    evaluate = subs.add_parser("evaluate")
    evaluate.add_argument("--run", required=True, help="Run directory, relative to gp/")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--final", action="store_true", help="Freeze the complete experiment before test evaluation")
    subs.add_parser("export-web")
    subs.add_parser("export-hires", help="Sharper 1024-px test photos for the app's expanded views (git-ignored)")
    subs.add_parser("export-walkthrough", help="Step-by-step model walkthrough for the app, checked against frozen predictions")
    subs.add_parser("export-preprocessing", help="Data-preprocessing page for the app (checks, fixes and transforms with real numbers)")
    args = parser.parse_args()
    environment()
    try:
        cfg = config(args.config)
        match args.command:
            case "doctor":
                from .doctor import doctor
                print(json.dumps(doctor(cfg, args.gpu_smoke), indent=2))
            case "download":
                from .acquire import download
                download()
            case "audit":
                from .data import audit
                audit(cfg, args.raw)
            case "prepare":
                from .data import prepare
                prepare(cfg)
            case "features":
                from .features import build_features
                build_features(cfg)
            case "overlap":
                from .overlap import analyse
                analyse(cfg)
            case "overlap-check":
                from .overlap import check_pair
                print(json.dumps(check_pair(cfg, args.eval, args.train), indent=2))
            case "train":
                from .training import train_rf, train_unet
                seed = cfg["train_seeds"][0] if args.seed is None else args.seed
                if seed not in cfg["train_seeds"] and not args.smoke:
                    raise ProjectError(f"Seed {seed} is not in the configured train_seeds {cfg['train_seeds']}")
                path = (train_rf if args.model == "rf" else train_unet)(cfg, args.budget, seed, resume=args.resume, smoke=args.smoke)
                log(f"Saved run: {path}")
            case "evaluate":
                from .evaluation import evaluate
                evaluate(cfg, args.run, args.split, args.final)
            case "export-web":
                from .export import export_web
                export_web(cfg)
            case "export-hires":
                from .visuals import export_hires
                export_hires(cfg)
            case "export-walkthrough":
                from .visuals import export_walkthrough
                export_walkthrough(cfg)
            case "export-preprocessing":
                from .visuals import export_preprocessing
                export_preprocessing(cfg)
    except (ProjectError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
