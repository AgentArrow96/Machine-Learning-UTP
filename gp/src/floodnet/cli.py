"""Command entrypoint. Heavy ML dependencies load only for relevant commands."""
import argparse
import json
import sys
from .common import ProjectError, config, environment, log


def main():
    parser = argparse.ArgumentParser(prog="floodnet", description="FloodNet: CPU SLIC/RF and CUDA U-Net, inside gp/")
    parser.add_argument("--config", help="Configuration JSON inside gp/; defaults to configs/base.json")
    parser.add_argument("--hybrid-config", help="Hybrid (CRF) settings inside gp/; defaults to configs/hybrid.json")
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
    subs.add_parser("tune-crf", help="Select DenseCRF settings on validation for every U-Net run")
    refine = subs.add_parser("refine", help="Record a hybrid run that refines one U-Net run")
    refine.add_argument("--budget", type=int, choices=(25, 100), required=True)
    refine.add_argument("--seed", type=int, required=True)
    evaluate = subs.add_parser("evaluate")
    evaluate.add_argument("--run", required=True, help="Run directory, relative to gp/")
    evaluate.add_argument("--split", choices=("val", "test"), default="val")
    evaluate.add_argument("--final", action="store_true", help="Freeze the complete four-run experiment before test evaluation")
    subs.add_parser("export-web")
    subs.add_parser("export-hires", help="Sharper 1024-px test photos for the app's expanded views (git-ignored)")
    walk = subs.add_parser("export-walkthrough", help="Step-by-step model walkthrough for the app, checked against frozen predictions")
    walk.add_argument("--rf-variant", help='Replay another study\'s forests, e.g. "undersample/uniform"')
    subs.add_parser("export-preprocessing", help="Data-preprocessing page for the app (checks, fixes and transforms with real numbers)")
    imp = subs.add_parser("improve", help="Improvement study stages (see docs/improve.md)")
    imp.add_argument("stage", choices=("preregister", "fetch-encoder", "prepare512", "features", "train", "combine", "freeze", "test", "report", "export", "export-after"))
    imp.add_argument("--variant"); imp.add_argument("--method", choices=("rf", "unet", "hybrid"))
    imp.add_argument("--budget", type=int, choices=(25, 100)); imp.add_argument("--seed", type=int)
    und = subs.add_parser("undersample", help="Uniform Before and undersampling pipeline stages (see docs/undersample.md)")
    und.add_argument("stage", choices=("preregister", "features-blacken", "train", "choose", "freeze", "test", "report", "export", "walkthrough"))
    und.add_argument("--kind", help="uniform, r1, r3, r10 or blacken"); und.add_argument("--method", choices=("rf", "unet", "hybrid"))
    und.add_argument("--budget", type=int, choices=(25, 100)); und.add_argument("--seed", type=int)
    imb = subs.add_parser("imbalance", help="Class-imbalance study stages (see docs/imbalance.md)")
    imb.add_argument("stage", choices=("preregister", "masks10", "gan", "set", "train", "freeze", "test", "report", "export"))
    imb.add_argument("--variant"); imb.add_argument("--method", choices=("rf", "unet", "hybrid"))
    imb.add_argument("--budget", type=int, choices=(25, 100)); imb.add_argument("--seed", type=int)
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
            case "tune-crf":
                from .hybrid import hybrid_config, tune
                tune(cfg, hybrid_config(args.hybrid_config))
            case "refine":
                from .hybrid import create_run
                if args.seed not in cfg["train_seeds"]:
                    raise ProjectError(f"Seed {args.seed} is not in the configured train_seeds {cfg['train_seeds']}")
                create_run(cfg, args.budget, args.seed)
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
                export_walkthrough(cfg, args.rf_variant)
            case "improve":
                from . import improve
                if args.stage == "preregister":
                    print(json.dumps(improve.preregister(), indent=2))
                elif args.stage == "fetch-encoder":
                    improve.fetch_encoder()
                elif args.stage == "prepare512":
                    improve.prepare_size(cfg, 512)
                elif args.stage == "features":
                    improve.stage_features(cfg, args.variant)
                elif args.stage == "train":
                    improve.train_improve(cfg, args.variant, args.method, args.budget, args.seed)
                elif args.stage == "combine":
                    improve.decide_combination()
                elif args.stage == "freeze":
                    from .data import Store
                    improve.freeze_improve(cfg, Store(cfg))
                elif args.stage == "test":
                    improve.stage_test(cfg)
                elif args.stage == "report":
                    improve.report(cfg)
                elif args.stage == "export":
                    improve.export_improve(cfg)
                elif args.stage == "export-after":
                    improve.export_after(cfg)
            case "undersample":
                from . import undersample
                if args.stage == "preregister":
                    print(json.dumps(undersample.preregister(), indent=2))
                elif args.stage == "features-blacken":
                    undersample.stage_features_blacken(cfg)
                elif args.stage == "train":
                    undersample.train(cfg, args.kind, args.method, args.budget, args.seed)
                elif args.stage == "choose":
                    undersample.decide_ratio()
                elif args.stage == "freeze":
                    from .data import Store
                    undersample.freeze_undersample(cfg, Store(cfg))
                elif args.stage == "test":
                    undersample.stage_test(cfg)
                elif args.stage == "report":
                    undersample.report(cfg)
                elif args.stage == "export":
                    undersample.export_bundles(cfg)
                elif args.stage == "walkthrough":
                    from .visuals import export_walkthrough
                    export_walkthrough(cfg, undersample.study_path("uniform"))
            case "imbalance":
                from . import imbalance
                if args.stage == "preregister":
                    print(json.dumps(imbalance.preregister(), indent=2))
                elif args.stage == "masks10":
                    from .data import Store
                    from .gan_common import masks10_cache
                    store = Store(cfg)
                    for split in ("train", "val"):
                        masks10_cache(cfg, store, split)
                elif args.stage == "gan":
                    imbalance.stage_gan(cfg, args.variant, args.budget)
                elif args.stage == "set":
                    imbalance.stage_set(cfg, args.variant, args.budget)
                elif args.stage == "train":
                    imbalance.train_variant(cfg, args.variant, args.method, args.budget, args.seed)
                elif args.stage == "freeze":
                    from .data import Store
                    imbalance.freeze_imbalance(cfg, Store(cfg))
                elif args.stage == "test":
                    imbalance.stage_test(cfg)
                elif args.stage == "report":
                    imbalance.report(cfg)
                elif args.stage == "export":
                    imbalance.export_imbalance(cfg)
            case "export-preprocessing":
                from .visuals import export_preprocessing
                export_preprocessing(cfg)
    except (ProjectError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
