
import os
import pickle
import numpy as np
from src.utils.experiments import ExperimentBase
from src.loss_landscape.similarity.repfun import RepFunSimilarity
from src.loss_landscape.connectivity.mode_connectivity import CurveCoordinates
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--exp_a", "-ea")
parser.add_argument("--exp_b", "-eb")
parser.add_argument("--run_a", "-ra")
parser.add_argument("--run_b", "-rb")
parser.add_argument("--type", "-t")
parser.add_argument("--cwd", "-c")


args = parser.parse_args()

exp_a = ExperimentBase.from_file(f"experiments/new/{args.exp_a}.json")
exp_b = ExperimentBase.from_file(f"experiments/new/{args.exp_b}.json")

root_dir = f"{args.cwd}/models/{args.type}"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

filename = f"{args.exp_a}_{args.run_a}_{args.exp_b}_{args.run_b}.pkl"

if args.type == "cka":
    sim = RepFunSimilarity.from_custom(exp_a, exp_b, int(args.run_a), int(args.run_b), "cuda", on_test_set=False)
    result = sim.get_cka_similarities()
    m, m_minmax = sim.get_disagreement()
    result["m"] = m
    result["m_minmax"] = m_minmax
    with open(os.path.join(root_dir, filename), "wb") as f:
        pickle.dump(result, f)

elif args.type == "mode":
    
    result = dict()
    result["linear"] = dict()
    result["bezier"] = {
        "hamming": dict(),
        "bce": dict()
    }

    # Initial: Curve with linear interpolation between two modes
    curve = CurveCoordinates.from_custom(exp_a, exp_b, int(args.run_a), int(args.run_b), "cuda", use_bce_loss=True)
    
    ts_train, loss_train, loss_train_bce, tpr_train, fpr_train = curve.eval(51, False)
    ts_test, loss_test, loss_test_bce, tpr_test, fpr_test = curve.eval(51, on_test_set=True)

    result["ts"] = ts_train

    result["linear"]["hamming_train"] = loss_train
    result["linear"]["bce_train"] = loss_train_bce
    result["linear"]["tpr_train"] = tpr_train
    result["linear"]["fpr_train"] = fpr_train

    result["linear"]["hamming_test"] = loss_test
    result["linear"]["bce_test"] = loss_test_bce
    result["linear"]["tpr_test"] = tpr_test
    result["linear"]["fpr_test"] = fpr_test
    
    # Curve with learned bezier interpolation between two modes (hamming)
    curve = CurveCoordinates.from_custom(exp_a, exp_b, int(args.run_a), int(args.run_b), "cuda", use_bce_loss=False)
    curve.fit(1000, lr=1e-3, batch_size=8)
    curve_filename  = f"{args.exp_a}_{args.run_a}_{args.exp_b}_{args.run_b}_hamming_curve.pt"
    curve.to_file(os.path.join(root_dir, curve_filename))

    ts_train, loss_train, loss_train_bce, tpr_train, fpr_train = curve.eval(51, False)
    ts_test, loss_test, loss_test_bce, tpr_test, fpr_test = curve.eval(51, on_test_set=True)

    result["bezier"]["hamming"]["hamming_train"] = loss_train
    result["bezier"]["hamming"]["bce_train"] = loss_train_bce
    result["bezier"]["hamming"]["tpr_train"] = tpr_train
    result["bezier"]["hamming"]["fpr_train"] = fpr_train

    result["bezier"]["hamming"]["hamming_test"] = loss_test
    result["bezier"]["hamming"]["bce_test"] = loss_test_bce
    result["bezier"]["hamming"]["tpr_test"] = tpr_test
    result["bezier"]["hamming"]["fpr_test"] = fpr_test

    # Curve with learned bezier interpolation between two modes (bce)
    curve = CurveCoordinates.from_custom(exp_a, exp_b, int(args.run_a), int(args.run_b), "cuda", use_bce_loss=True)
    curve.fit(1000, lr=1e-3, batch_size=8)
    curve_filename = f"{args.exp_a}_{args.run_a}_{args.exp_b}_{args.run_b}_bce_curve.pt"
    curve.to_file(os.path.join(root_dir, curve_filename))

    ts_train, loss_train, loss_train_bce, tpr_train, fpr_train = curve.eval(51, False)
    ts_test, loss_test, loss_test_bce, tpr_test, fpr_test = curve.eval(5 1, on_test_set=True)

    result["bezier"]["bce"]["hamming_train"] = loss_train
    result["bezier"]["bce"]["bce_train"] = loss_train_bce
    result["bezier"]["bce"]["tpr_train"] = tpr_train
    result["bezier"]["bce"]["fpr_train"] = fpr_train

    result["bezier"]["bce"]["hamming_test"] = loss_test
    result["bezier"]["bce"]["bce_test"] = loss_test_bce
    result["bezier"]["bce"]["tpr_test"] = tpr_test
    result["bezier"]["bce"]["fpr_test"] = fpr_test
    

    with open(os.path.join(root_dir, filename), "wb") as f:
        pickle.dump(result, f)

else:
    raise Exception("Invalid type")
