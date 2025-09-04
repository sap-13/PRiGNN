import argparse
import json
import os
import yaml
from dataclasses import dataclass
import torch
import torch_geometric.nn as gnn
from prignn.prompts import PROMPT_TEMPLATE
from prignn.task_adapter import get_task
from prignn.profiler import profile_data, load_search_space
from prignn.llm_driver import get_llm_driver
from prignn.trainer import train_and_eval
from prignn.result_store import ResultStore, hash_code, TrialResult

@dataclass
class BuildResult:
    model: torch.nn.Module
    meta: dict

def extract_architecture_and_meta(reply: str):
    try:
        # A more robust way to find the JSON block
        json_start = reply.find("```json")
        if json_start == -1:
            json_start = reply.find("{")
        else:
            json_start += len("```json")

        json_end = reply.rfind("```")
        if json_end == -1:
            json_end = reply.rfind("}") + 1
        
        json_str = reply[json_start:json_end].strip()
        data = json.loads(json_str)
        architecture = data.get("architecture", {})
        hyperparameters = data.get("hyperparameters", {})
        return architecture, hyperparameters
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing LLM reply: {e}")
        return {}, {}

def build_model_from_json(architecture_config, hyperparameters, data, num_classes):
    layers = []
    in_channels = data.num_node_features

    for layer_config in architecture_config.get("layers", []):
        layer_type = layer_config.get("layer_type")
        out_channels = layer_config.get("out_channels")
        
        if out_channels == "auto":
            out_channels = num_classes

        conv_class = getattr(gnn, layer_type, None)
        if conv_class:
            # Handle GATConv heads
            if layer_type == "GATConv" and "heads" in layer_config:
                # For GAT, the output features are out_channels * heads
                layers.append(conv_class(in_channels, out_channels, heads=layer_config["heads"]))
                in_channels = out_channels * layer_config["heads"]
            else:
                layers.append(conv_class(in_channels, out_channels))
                in_channels = out_channels
        
        activation = layer_config.get("activation")
        if activation and hasattr(torch.nn, activation):
            layers.append(getattr(torch.nn, activation)())

    model = gnn.Sequential("x, edge_index", [(layer, "x, edge_index -> x") for layer in layers])
    
    return BuildResult(model=model, meta=hyperparameters)

def run_search(
    dataset_name: str,
    llm_driver_name: str,
    model_name: str,
    max_rounds: int,
    stop_delta: float,
    mlflow_experiment: str,
    project_id: str,
    location: str,
) -> None:
    # 1. Initialization
    task = get_task(dataset_name)
    data = task.dataset[0]
    metric_name = task.metric_fn.__name__
    if metric_name == "<lambda>":
        metric_name = f"{task.name}_metric"
    profile = profile_data(task.name, data, task.task_type)
    search_space = load_search_space()
    llm = get_llm_driver(llm_driver_name, model_name, project_id=project_id, location=location)
    store = ResultStore(experiment_name=mlflow_experiment)

    best_metric = -1.0e9
    past_performance_history = []

    # 2. Main Search Loop
    for rnd in range(max_rounds):
        # New: Generate history string for the prompt
        history_string = ""
        if past_performance_history:
            history_string += "\n**PAST ARCHITECTURE PERFORMANCE**\n"
            # Sort by metric value (descending)
            sorted_history = sorted(past_performance_history, key=lambda x: x["metric_value"], reverse=True)
            for entry in sorted_history:
                history_string += f"- Architecture: {json.dumps(entry['architecture'])}\\n  Validation Metric: {entry['metric_value']:.4f}\\n"
        
        # New: Dynamic search strategy guidance
        strategy_guidance = ""
        if rnd < max_rounds * 0.3: # First 30% of rounds for exploration
            strategy_guidance = "You are in the **Exploration** phase. Focus on exploring diverse architectures and hyperparameter combinations across the entire search space. Do not be afraid to try novel and unconventional designs."
        else: # Later rounds for exploitation
            strategy_guidance = "You are in the **Exploitation** phase. Focus on refining and improving architectures that have shown promising results in the past. Make small, incremental changes around successful patterns. Avoid designs that have consistently performed poorly."

        prompt = PROMPT_TEMPLATE.format(
            dataset_name=profile.dataset_name,
            task_type=profile.task_type,
            num_nodes=profile.num_nodes,
            num_edges=profile.num_edges,
            num_features=profile.num_features,
            num_classes=profile.num_classes,
            metric_name=metric_name,
            search_space=yaml.dump(search_space),
            past_performance_history=history_string,
            round_no=rnd,
            max_rounds=max_rounds,
            strategy_guidance=strategy_guidance,
        )

        # 3. Generate, Build, and Train
        try:
            reply = llm.generate(prompt)
            cost = llm.get_cost(0, 0) # Placeholder for token counting

            architecture_config, hyperparameters = extract_architecture_and_meta(reply)

            if not architecture_config or not architecture_config.get("layers"):
                print("Skipping round due to empty or invalid architecture.")
                continue
            
            build_result = build_model_from_json(
                architecture_config, hyperparameters, data, task.dataset.num_classes
            )
            model = build_result.model

            val_accuracy, test_accuracy, loss = train_and_eval(
                model,
                data,
                task.task_type,
                data.train_mask,
                data.val_mask,
                data.test_mask,
                task.metric_fn,
                epochs=hyperparameters.get("epochs", 200),
                lr=hyperparameters.get("lr", 0.005),
            )
            
            if val_accuracy > best_metric:
                best_metric = val_accuracy
                print(f"New best metric: {best_metric:.4f}")

            tr = TrialResult(
                round_no=rnd,
                metric_name=metric_name,
                metric_value=val_accuracy,
                meta=hyperparameters,
                architecture_config=architecture_config,
                code=reply,
                code_hash=hash_code(reply),
                cost=cost,
            )
            store.log_trial(tr)

            past_performance_history.append({
                "architecture": architecture_config,
                "metric_value": val_accuracy
            })

        except Exception as e:
            print(f"An unexpected error occurred in round {rnd}: {e}")
            continue

    print("--- Search Finished ---")
    print(f"Best metric ({task.metric_fn.__name__}): {best_metric:.4f}")


if __name__ == "__main__":
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        # Clean value of null bytes and extra quotes
                        cleaned_value = value.strip().strip('"').replace('\x00', '')
                        os.environ[key.strip()] = cleaned_value
        except Exception as e:
            print(f"Error reading .env file: {e}")

    parser = argparse.ArgumentParser(description="PRiGNN v2 Orchestrator")
    parser.add_argument("--dataset", type=str, default="Cora", help="Name of the dataset (e.g., Cora, ogbn-arxiv)")
    parser.add_argument("--llm-driver", type=str, default="gemini", choices=["gemini"], help="The LLM to use")
    parser.add_argument("--model-name", type=str, default="gemini-2.5-flash-lite", help="The specific LLM model to use")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of times to run the search")
    parser.add_argument("--rounds", type=int, default=20, help="Number of architecture search rounds")
    parser.add_argument("--stop-delta", type=float, default=0.001, help="Minimum improvement to update the best model")
    parser.add_argument("--mlflow-experiment", type=str, default="prignn-search", help="Name of the MLflow experiment")
    parser.add_argument("--project-id", type=str, default=os.getenv("PROJECT_ID"), help="Google Cloud Project ID for Vertex AI")
    parser.add_argument("--location", type=str, default=os.getenv("LOCATION"), help="Google Cloud Location for Vertex AI")
    args = parser.parse_args()

    if not args.project_id or not args.location:
        raise ValueError("Please provide --project-id and --location arguments, or set PROJECT_ID and LOCATION in your .env file.")

    for i in range(args.num_trials):
        print(f"--- Starting Trial {i+1}/{args.num_trials} ---")
        run_search(
            dataset_name=args.dataset,
            llm_driver_name=args.llm_driver,
            model_name=args.model_name,
            max_rounds=args.rounds,
            stop_delta=args.stop_delta,
            mlflow_experiment=args.mlflow_experiment,
            project_id=args.project_id,
            location=args.location,
        )
