import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ig, ig_net

def explain(args, model, dataset, feature_names, omic_group, labels, name=None):
    print('Perform model interpretation')
    if args.model == "DNN" or name == "DNN":
        for target in set(labels):
            ig(args, model, dataset, feature_names, omic_group, target=int(target))
    if args.model == "Net" or name == "Net":
        for target in set(labels):
            ig_net(args, model, dataset, feature_names, omic_group, target=int(target))


def _lazy_import_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "Generating SHAP values requires the 'shap' package. Install shap>=0.41.0 to enable this feature."
        ) from exc
    return shap


def _format_omic_name(omic_name):
    if isinstance(omic_name, (list, tuple)):
        return "_".join(str(v) for v in omic_name)
    return str(omic_name)


def _sample_rows(array, sample_size, rng):
    if array.shape[0] == 0:
        return array
    if array.shape[0] <= sample_size:
        return array
    idx = rng.choice(array.shape[0], size=sample_size, replace=False)
    return array[idx]


def explain_ml_shap(args, model, X_train, X_test, feature_names, omic_group, model_name):
    shap = _lazy_import_shap()
    rng = np.random.default_rng(getattr(args, "seed", 42))
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    background = _sample_rows(X_train, sample_size=min(100, max(1, X_train.shape[0])), rng=rng)
    eval_source = X_test if X_test.shape[0] > 0 else X_train
    eval_sample = _sample_rows(eval_source, sample_size=min(200, max(1, eval_source.shape[0])), rng=rng)
    if background.shape[0] == 0:
        background = eval_sample
    if eval_sample.shape[0] == 0:
        raise ValueError("No samples available to compute SHAP explanations.")

    lower_name = model_name.lower()
    if lower_name == "svm":
        print(f"Computing SHAP values with KernelExplainer for {model_name}.")
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(eval_sample)
    else:
        print(f"Computing SHAP values with TreeExplainer for {model_name}.")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(eval_sample)

    if isinstance(shap_values, list):
        shap_arrays = shap_values
    else:
        shap_arrays = [shap_values]

    try:
        class_labels = getattr(model, "classes_", None)
    except AttributeError:
        class_labels = None
    if class_labels is None or len(class_labels) != len(shap_arrays):
        class_labels = list(range(len(shap_arrays)))

    feature_names = list(feature_names)
    omic_group = list(omic_group)
    omic_tag = _format_omic_name(getattr(args, "omic_name", "NA"))
    if getattr(args, "clin_file", None):
        omic_tag = f"{omic_tag}_clin"
    fsd_tag = f"FSD_{args.method}" if args.FSD else str(args.method)
    features_eval = eval_sample
    os.makedirs(args.outdir, exist_ok=True)
    for idx, shap_matrix in enumerate(shap_arrays):
        if hasattr(shap_matrix, "values"):
            shap_matrix = shap_matrix.values
        shap_matrix = np.asarray(shap_matrix)
        # Reduce extra dimensions while preserving feature alignment
        if shap_matrix.ndim > 3:
            raise ValueError(
                "Unsupported SHAP output dimensionality "
                f"(shape={shap_matrix.shape}). Expected up to 3 dimensions."
            )
        if shap_matrix.ndim == 3:
            # Figure out which axis represents features
            axes = shap_matrix.shape
            if axes[1] == len(feature_names):
                # Shape: (samples, features, something_else) -> average over trailing axis
                shap_matrix = shap_matrix.mean(axis=2)
            elif axes[2] == len(feature_names):
                # Shape: (samples, classes, features) -> take positive class if available else mean
                class_axis = 1
                if axes[class_axis] > 1:
                    pos_index = 1 if axes[class_axis] > 1 else 0
                    shap_matrix = shap_matrix[:, pos_index, :]
                else:
                    shap_matrix = shap_matrix.mean(axis=class_axis)
            else:
                raise ValueError(
                    "Cannot align SHAP tensor with feature metadata. "
                    f"tensor shape={axes}, features={len(feature_names)}"
                )
        if shap_matrix.shape[1] != len(feature_names):
            if shap_matrix.shape[0] == len(feature_names):
                shap_matrix = shap_matrix.T
            else:
                raise ValueError(
                    "Mismatch between SHAP matrix and feature metadata. "
                    f"matrix shape={shap_matrix.shape}, features={len(feature_names)}"
                )

        # save csv
        target_label = class_labels[idx]
        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Omic": omic_group,
            "Target": [target_label] * len(feature_names),
            "Importance_value": np.mean(shap_matrix, axis=0),
            "Importance_value_abs": np.mean(np.abs(shap_matrix), axis=0)
        }).sort_values("Importance_value_abs", ascending=False)
        filename = f"feature_importance_SHAP_{fsd_tag}_{model_name}_{omic_tag}_target{target_label}.csv"
        df_imp.to_csv(os.path.join(args.outdir, filename), index=False)

        # generate plots
        plot_prefix = f"{fsd_tag}_{model_name}_{omic_tag}_target{target_label}"
        max_display = min(20, shap_matrix.shape[1])
        try:
            plt.figure()
            shap.summary_plot(shap_matrix, features=features_eval, feature_names=feature_names,
                              show=False, plot_type="bar", max_display=max_display)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"summary_bar_SHAP_{plot_prefix}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()

            plt.figure()
            shap.summary_plot(shap_matrix, features=features_eval, feature_names=feature_names,
                              show=False, max_display=max_display)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f"summary_beeswarm_SHAP_{plot_prefix}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            plt.close()
            print(f"[Warn] Unable to generate SHAP summary plots for {model_name} target {target_label}: {exc}")
