import matplotlib.pyplot as plt
from src.ai_model import build_models, save_models, train_models
from src.config import ExperimentConfig
from src.cyber_risk import assign_risk_level, calculate_decision_score
from src.data_loader import dataset_summary, load_dataset, make_binary_target
from src.decision_intelligence import generate_decision_outputs
from src.evaluation import evaluate_models, plot_confusion, plot_model_comparison, plot_roc_pr
from src.explainability import explain_model
from src.preprocessing import prepare_data
from src.utils import ensure_directories, save_json, set_seed, setup_logging

def main() -> None:
    config = ExperimentConfig(); logger = setup_logging(); set_seed(config.random_state)
    ensure_directories(config.raw_dir, config.processed_dir, config.models_dir, config.figures_dir, config.tables_dir, config.results_dir)
    logger.info("[1/9] Loading dataset..."); df, label_col = load_dataset(config); dataset_summary(df, label_col).to_csv(config.tables_dir / "table1_dataset_characteristics.csv", index=False)
    target = make_binary_target(df[label_col]); fig, ax = plt.subplots(figsize=(6, 4)); target.value_counts().sort_index().rename(index={0: "Benign", 1: "Attack"}).plot(kind="bar", ax=ax); ax.set_title("Cybersecurity Class Distribution"); ax.set_ylabel("Rows"); plt.tight_layout(); plt.savefig(config.figures_dir / "class_distribution.png", dpi=300); plt.close()
    logger.info("[2/9] Preprocessing..."); prepared = prepare_data(df, label_col, config.test_size, config.random_state)
    logger.info("[3/9] Training models..."); trained = train_models(build_models(prepared.preprocessor, config.random_state), prepared.X_train, prepared.y_train); save_models(trained, config.models_dir)
    logger.info("[4/9] Evaluating models..."); metrics, predictions = evaluate_models(trained, prepared.X_test, prepared.y_test); metrics.to_csv(config.tables_dir / "table2_model_performance.csv", index=False); plot_model_comparison(metrics, config.figures_dir / "model_performance.png")
    best_name = metrics.iloc[0]["model"]; best_model = trained[best_name]; best_pred = predictions[best_name]["pred"]; best_prob = predictions[best_name]["prob"]
    plot_confusion(prepared.y_test, best_pred, config.figures_dir / "confusion_matrix.png"); plot_roc_pr(prepared.y_test, best_prob, config.figures_dir / "roc_curve.png", config.figures_dir / "precision_recall_curve.png")
    logger.info("[5/9] Running explainability..."); explain_sample = prepared.X_test.iloc[:min(config.shap_sample_size, len(prepared.X_test))]; top_features, local_explanations = explain_model(best_model, explain_sample, config.figures_dir); top_features.to_csv(config.tables_dir / "table3_top_features.csv", index=False)
    logger.info("[6/9] Calculating cybersecurity risk..."); risk_df = calculate_decision_score(best_prob, config.decision_weights); risk_df["risk_level"] = assign_risk_level(risk_df["decision_score"], config.risk_thresholds); risk_counts = risk_df["risk_level"].value_counts().rename_axis("risk_level").reset_index(name="count"); risk_counts.to_csv(config.tables_dir / "table4_cyber_risk_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4)); risk_counts.set_index("risk_level")["count"].plot(kind="bar", ax=ax); ax.set_title("Cyber Risk Distribution"); ax.set_ylabel("Events"); plt.tight_layout(); plt.savefig(config.figures_dir / "cyber_risk_distribution.png", dpi=300); plt.close()
    logger.info("[7/9] Generating Decision Intelligence..."); decision_df = generate_decision_outputs(best_pred, risk_df, local_explanations["important_features"] if not local_explanations.empty else None); decision_df.head(50).to_csv(config.tables_dir / "table5_decision_intelligence_examples.csv", index=False); decision_df.to_csv(config.results_dir / "decision_outputs.csv", index=False)
    summary_data = {"dataset_rows_loaded": int(len(df)), "dataset_columns_loaded": int(len(df.columns)), "label_column": label_col, "train_rows": int(len(prepared.X_train)), "test_rows": int(len(prepared.X_test)), "models_evaluated": metrics["model"].tolist(), "best_model": best_name, "best_model_metrics": metrics.iloc[0].to_dict(), "risk_category_counts": risk_df["risk_level"].value_counts().to_dict(), "decision_priority_counts": decision_df["decision_priority"].value_counts().to_dict(), "research_note": "Risk thresholds and Decision Intelligence weights are experimental prototype settings, not validated industry standards."}; save_json(summary_data, config.results_dir / "experiment_summary.json")
    logger.info("[8/9] Saved tables, figures and results."); logger.info("[9/9] Experiment completed. Best model: %s", best_name)

if __name__ == "__main__": main()
