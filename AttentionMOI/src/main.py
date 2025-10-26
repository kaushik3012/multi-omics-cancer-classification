import os
import argparse
from .utils import check_files, init_log
from .fsd import feature_selection_distribution
from .train2 import train, ml_models
from .train2 import train_net
from .train2 import train_moanna, train_mogonet
from .explain import explain
'''

print("--- DEBUG: main.py script has started ---")
import os
import traceback

try:
    print("--- DEBUG: Attempting to import .utils...")
    from .utils import check_files, init_log
    print("--- DEBUG: Successfully imported .utils")

    print("--- DEBUG: Attempting to import .fsd...")
    from .fsd import feature_selection_distribution
    print("--- DEBUG: Successfully imported .fsd")

    print("--- DEBUG: Attempting to import .train...")
    from .train import train, ml_models, train_net, train_moanna, train_mogonet
    print("--- DEBUG: Successfully imported .train")

    print("--- DEBUG: Attempting to import .explain...")
    from .explain import explain
    print("--- DEBUG: Successfully imported .explain")

    print("\n--- DEBUG: ALL IMPORTS SUCCEEDED ---\n")

except Exception as e:
    print("\n--- A HIDDEN ERROR WAS FOUND! ---")
    print(f"The error likely comes from the last module we tried to import above.")
    print(f"Error Message: {e}")
    print("--- Full Traceback: ---")
    traceback.print_exc()
    print("---------------------------\n")
'''
#print("--- DEBUG: Script is about to define the run() function...")

def run(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # init log.txt
    init_log(args=args)

    # features selection - with FSD or not
    data, feature_name, feature_group, labels = feature_selection_distribution(args)

    # training model - using different models
    if args.model == "DNN":
        model, dataset_test = train(args, data, labels)
        if args.explain:
            explain(args, model, dataset_test, feature_name, feature_group, labels)
    elif args.model in ["RF", "XGboost", "svm"]:
        name = args.model
        ml_models(args, data, feature_name, feature_group, labels, model_name=name)
    elif args.model == "Net":
        model, dataset_test = train_net(args, data, feature_group, labels)
        if args.explain:
            explain(args, model, dataset_test, feature_name, feature_group, labels)
    
    elif args.model == "moanna":
        train_moanna(args, data, labels)
        if args.explain:
            print("[Warn] Integrated gradients explanations are only supported for DNN and Net models. Skipping.")

    elif args.model == "mogonet":
        train_mogonet(args, data, labels)
        if args.explain:
            print("[Warn] Integrated gradients explanations are only supported for DNN and Net models. Skipping.")
    
    # multiple models
    elif args.model == "all":
        # DNN model
        model, dataset_test = train(args, data, labels)
        if args.explain:
            explain(args, model, dataset_test, feature_name, feature_group, labels, name="DNN")
        # Net model
        model, dataset_test = train_net(args, data, feature_group, labels)
        if args.explain:
            explain(args, model, dataset_test, feature_name, feature_group, labels, name="Net")
        # ml model
        for name in ["RF", "XGboost", "svm"]:
            ml_models(args, data, feature_name, feature_group, labels, model_name=name)
        if args.explain:
            print("[Info] Integrated gradients were generated for DNN/Net and SHAP values were generated for RF/XGboost/svm.")

'''            
if __name__ == '__main__':
    # This part defines all the command-line arguments the script can accept.
    parser = argparse.ArgumentParser(description="AttentionMOI: Multi-omics Integration Model Training")

    parser.add_argument('-f', '--omic_file', action='append', required=True, 
                        help='Path to an omic data file. Specify this argument multiple times for multiple files.')

    parser.add_argument('-n', '--omic_name', action='append', required=True, 
                        help='Name for an omic data type (e.g., rna, met). Specify in the same order as the files.')

    parser.add_argument('-l', '--label_file', type=str, required=True, 
                        help='Path to the label file.')

    parser.add_argument('-m', '--model', type=str, required=True, 
                        help='Model to use (e.g., MLP, Net, RF, XGboost, svm, moanna, mogonet, all).')

    parser.add_argument('-o', '--outdir', type=str, required=True, 
                        help='Directory to save the output results.')

    parser.add_argument('--FSD', action='store_true', 
                        help='Use this flag to enable Feature Selection Distribution.')
    
    parser.add_argument('--method', type=str, default='t-test', 
                    help='Feature selection method to use when not using FSD (e.g., t-test, chi2).')

    parser.add_argument('--clin_file', type=str, required=False, 
                    help='Path to the clinical data file (optional).')

    # This part reads the arguments you provide in the terminal
    args = parser.parse_args()

    # This part finally calls the run() function to start the analysis
    run(args)

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AttentionMOI: Multi-omics Integration Model Training")

    # ---- File Paths ----
    parser.add_argument('-f', '--omic_file', action='append', required=True, help='Path to an omic data file. Specify multiple times for multiple files.')
    parser.add_argument('-n', '--omic_name', action='append', required=True, help='Name for an omic data type (e.g., rna, met). Use the same order as files.')
    parser.add_argument('-l', '--label_file', type=str, required=True, help='Path to the label file.')
    parser.add_argument('-o', '--outdir', type=str, required=True, help='Directory to save the output results.')
    parser.add_argument('--clin_file', type=str, required=False, help='Path to the clinical data file (optional).')

    # ---- Model and Training Parameters ----
    parser.add_argument('-m', '--model', type=str, required=True, help='Model to use (e.g., DNN, Net, RF, XGboost, svm, all).')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs for training.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')

    # ---- Feature Selection Parameters ----
    parser.add_argument('--FSD', action='store_true', help='Use this flag to enable Feature Selection Distribution.')
    parser.add_argument('--method', type=str, default='t-test', help='Feature selection method if not using FSD (e.g., t-test, chi2).')
    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for FSD.')
    parser.add_argument('--iteration', type=int, default=100, help='Number of iterations for FSD.')
    parser.add_argument('--percentile', type=int, default=10, help='Percentile of features to select if using percentile-based method.')
    parser.add_argument('--num_pc', type=int, default=50, help='Number of principal components for PCA.')

    # ---- Reproducibility ----
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')

    # ---- Explainability ----
    parser.add_argument('--explain', action='store_true', help='Generate integrated gradients explanations for supported models.')

    # ---- Parse arguments and run the main function ----
    args = parser.parse_args()
    run(args)







