import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import joblib
from .utils import evaluate


def _select_device(args, torch):
    requested = getattr(args, "device", None)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested:
        req = str(requested).lower()
        if req == "cuda":
            if not has_cuda:
                raise ValueError("Requested CUDA device via --device, but CUDA is not available.")
            return torch.device("cuda")
        if req == "mps":
            if not has_mps:
                raise ValueError("Requested MPS device via --device, but MPS is not available.")
            return torch.device("mps")
        if req == "cpu":
            return torch.device("cpu")
        print(f"[Warn] Unknown device string '{requested}' provided; falling back to automatic selection.")

    if has_cuda:
        return torch.device("cuda")
    if has_mps:
        return torch.device("mps")
    return torch.device("cpu")


# builing machine learning models
def ml_models(args, data, chosen_feat_name, chosen_omic_group, labels, model_name):
    """
    args: using args - test_size, seed
    data: combined omics data (format:np.array)
    chosen_feat_name: feature names
    chosen_omic_group: feature of omic group
    labels: labels of classes
    return: print train and test ACC, AUC, Fi_score, Recall, Precision, output feature importance as a .csv file
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args.test_size, random_state=args.seed)
    if args.model == "RF" or model_name == "RF":
        model = RandomForestClassifier(random_state=args.seed)
        model.fit(X_train, y_train)
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

    elif args.model == "XGboost" or model_name == "XGboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Model 'XGboost' requires the 'xgboost' package. Please install xgboost>=1.7.4 to use this model."
            ) from exc
        model = XGBClassifier(random_state=args.seed)
        model.fit(X_train, y_train)
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

    elif args.model == "svm" or model_name == "svm":
        model = svm.SVC(random_state=args.seed, probability=True)
        model.fit(X_train, y_train)
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_train),
                                          real_label=np.array(y_train))
    train_evaluate = 'Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
                acc, auc, f1, recall, prec)
    print(train_evaluate)

    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_test),
                                          real_label=np.array(y_test))
    test_evaluate = 'Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Recall  {:.3f} | Test_Precision  {:.3f}\n'.format(
                acc, auc, f1, recall, prec)
    print(test_evaluate)
    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        file.writelines("------Running {} model------\n".format(model_name))
        file.writelines(train_evaluate + '\n')
        file.writelines(test_evaluate + '\n')
    file.close()

    # write output for model evaluation
    # model, feature selection, accuracy, precision, f1_score, AUC, recall
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"{model_name}\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(f"{model_name}\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
        else:
            if args.clin_file:
                txt.writelines(f"{model_name}\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(
                    f"{model_name}\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
    txt.close()
    if args.explain:
        from .explain import explain_ml_shap
        explain_ml_shap(args, model, X_train, X_test, chosen_feat_name, chosen_omic_group, model_name)


# using DNN model
def train(args, data, labels):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Model 'DNN' requires the 'torch' package. Please install PyTorch to use this model."
        ) from exc
    from .module import DeepMOI

    device = _select_device(args, torch)
    print(f"[Info] Using device: {device}")

    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    # in_dim, out_dim
    dim_out = len(set(labels))
    in_dim = data.shape[1]
    model = DeepMOI(in_dim, dim_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    if len(dataset_train) == 0 or len(dataset_test) == 0:
        raise ValueError(
            "[train] The train/test split produced an empty dataset "
            f"(train={len(dataset_train)}, test={len(dataset_test)}). "
            "Reduce --test_size or verify that enough samples remain after preprocessing."
        )
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=False)

    for epoch in range(args.epoch):
        # 1) training model
        model.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data_batch, label_batch in train_loader:
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            out = model(data_batch)
            out = out.squeeze(-1)
            # loss
            loss = nn.CrossEntropyLoss()(out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(label_batch.detach().cpu().numpy())
        if len(loss_epoch) == 0 or len(y_pred_probs) == 0:
            raise ValueError(
                "[train] No training batches were generated. "
                "Check batch size and dataset size to ensure at least one batch is available."
            )
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data_batch, label_batch in test_loader:
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)
                out = model(data_batch)
                out = out.squeeze(-1)
                # loss
                loss = nn.CrossEntropyLoss()(out, label_batch)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                # predict labels
                y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(label_batch.detach().cpu().numpy())

            if len(loss_epoch) == 0 or len(y_pred_probs) == 0:
                raise ValueError(
                    "[train] No evaluation batches were generated for the test split. "
                    "Adjust --batch or --test_size so the test loader yields data."
                )
            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()

    return model, test_loader


def train_net(args, data, chosen_omic_group, labels):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Model 'Net' requires the 'torch' package. Please install PyTorch to use this model."
        ) from exc
    from .module import Net

    device = _select_device(args, torch)
    print(f"[Info] Using device: {device}")

    idx_rna = []
    for i, v in enumerate(chosen_omic_group):
        if v == 'rna':
            idx_rna.append(i)

    idx_dna = []
    for i, v in enumerate(chosen_omic_group):
        if v == 'cnv' or v == 'met':
            idx_dna.append(i)

    data_dna = data[:, idx_dna]
    data_rna = data[:, idx_rna]
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data_dna[i], dtype=torch.float),
                        torch.tensor(data_rna[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    # dim_dna, dim_rna, dim_out
    dim_dna = data_dna.shape[1]
    dim_rna = data_rna.shape[1]
    dim_out = len(set(labels))
    model = Net(dim_dna, dim_rna, dim_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    if len(dataset_train) == 0 or len(dataset_test) == 0:
        raise ValueError(
            "[train] The train/test split produced an empty dataset "
            f"(train={len(dataset_train)}, test={len(dataset_test)}). "
            "Reduce --test_size or verify that enough samples remain after preprocessing."
        )
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=False)

    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        file.writelines("------Running Net model------" + '\n')

    for epoch in range(args.epoch):
        # 1) training model
        model.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data_dna_batch, data_rna_batch, label_batch in train_loader:
            data_dna_batch = data_dna_batch.to(device)
            data_rna_batch = data_rna_batch.to(device)
            label_batch = label_batch.to(device)
            out = model(data_dna_batch, data_rna_batch)
            out = out.squeeze(-1)
            # loss
            loss = nn.CrossEntropyLoss()(out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(label_batch.detach().cpu().numpy())
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        print(log_train)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data_dna_batch, data_rna_batch, label_batch in test_loader:
                data_dna_batch = data_dna_batch.to(device)
                data_rna_batch = data_rna_batch.to(device)
                label_batch = label_batch.to(device)
                out = model(data_dna_batch, data_rna_batch)
                out = out.squeeze(-1)
                # loss
                loss = nn.CrossEntropyLoss()(out, label_batch)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                # predict labels
                y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(label_batch.detach().cpu().numpy())

            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)

            # to write log info
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')

    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"Net\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"Net\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"Net\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"Net\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()

    return model, test_loader


def evaluation(model, dataset):
    """
    To evaluate model performance for training dataset and testing dataset.

    model: taining model
    dataset: training dataset or testing dataset.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Evaluating torch-based models requires the 'torch' package. Please install PyTorch to use this helper."
        ) from exc
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    y_pred_probs, real_labels = [], []
    with torch.no_grad():
        for data_batch, label_batch in dataset:
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            out = model(data_batch)
            out = out.squeeze(-1)
            y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
            y_pred_probs.append(np.atleast_2d(y_pred_prob))
            real_labels.append(np.atleast_1d(label_batch.detach().cpu().numpy()))
    if not y_pred_probs or not real_labels:
        raise ValueError("[evaluation] Dataset contained no samples to evaluate.")
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    real_labels = np.concatenate(real_labels, axis=0)
    if was_training:
        model.train()
    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_probs),
                                          real_label=np.array(real_labels))
    return acc, prec, f1, auc, recall


# using Moanna model
def train_moanna(args, data, labels):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Model 'moanna' requires the 'torch' package. Please install PyTorch to use this model."
        ) from exc
    from .moanna.model.Moanna import Moanna, Moanna_cls

    device = _select_device(args, torch)
    print(f"[Info] Using device: {device}")

    # Pre-defined hyperparameters
    params = {}
    params["input_size"] = data.shape[1] # Number of features
    params["n_layers"] = 1
    params["encoded_size"] = 64 # Number of Autoencoders neurons
    params["hidden_size"] = params["encoded_size"] * (2 ** (params["n_layers"] + 1))
    params["drop_prob"] = 0.5
    params["fnn_hidden_size"] = 40
    params["fnn_number_layers"] = 2
    params["num_classes"] = len(set(labels))
    params["fnn_num_epoch"] = 100
    params["fnn_learning_rate"] = 0.05

    # Setup model
    model = Moanna(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    ).to(device)
    
    model_cls = Moanna_cls(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    ).to(device)
    
    
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    if len(dataset_train) == 0 or len(dataset_test) == 0:
        raise ValueError(
            "[train] The train/test split produced an empty dataset "
            f"(train={len(dataset_train)}, test={len(dataset_test)}). "
            "Reduce --test_size or verify that enough samples remain after preprocessing."
        )
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=False)

    criterion1 = torch.nn.MSELoss(reduction="mean")
    for _ in range(50):
        model.train()
        for data_batch, _ in train_loader:
            data_batch = data_batch.to(device)
            encoded, decoded = model(data_batch)
            loss = criterion1(decoded, data_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    for epoch in range(args.epoch):
        # 1) training model
        model_cls.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data_batch, label_batch in train_loader:
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device)
            encoded, decoded = model(data_batch)
            out = model_cls(encoded)
            loss = nn.CrossEntropyLoss()(out, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(label_batch.detach().cpu().numpy())
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        print(log_train)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data_batch, label_batch in test_loader:
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device)
                encoded, decoded = model(data_batch)
                out = model_cls(encoded)
                loss = nn.CrossEntropyLoss()(out, label_batch)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                y_pred_prob = F.softmax(out, dim=-1).detach().cpu().numpy()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(label_batch.detach().cpu().numpy())

            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()


# using MOGONET model
def train_mogonet(args, data, labels):
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Model 'mogonet' requires the 'torch' package. Please install PyTorch to use this model."
        ) from exc
    from .mogonet.main_mogonet import run_mogonet

    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    
    os.makedirs('tmp', exist_ok=True)
    dats, labs = [], []
    for dat, label in dataset_train:
        dats.append(dat.numpy())
        labs.append(label.numpy())
    a = np.vstack(dats)
    b = np.vstack(labs)
    np.savetxt('tmp/1_tr.csv', a, delimiter=',')
    np.savetxt('tmp/labels_tr.csv', b, delimiter=',')
    
    dats, labs = [], []
    for dat, label in dataset_test:
        dats.append(dat.numpy())
        labs.append(label.numpy())
    a = np.vstack(dats)
    b = np.vstack(labs)
    np.savetxt('tmp/1_te.csv', a, delimiter=',')
    np.savetxt('tmp/labels_te.csv', b, delimiter=',')
    
    
    logs = run_mogonet(num_epoch=args.epoch)
    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        for log in logs:
            file.writelines(log + '\n')
