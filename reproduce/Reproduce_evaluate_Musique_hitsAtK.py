import json
import os
from typing import Any, Dict, List, Optional

import faiss
import torch
from gritlm import GritLM
from tqdm import tqdm

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


def gritlm_instruction(instruction: str, embed_bos: str) -> str:
    return "<|user|>\n" + instruction + embed_bos


# Define constants
BASE_BOS: str = "<s>"
TURN_SEP: str = "\n"
USER_BOS: str = "<|user|>\n"
USER_EOS: str = ""
EMBED_EOS: str = ""
ASSISTANT_BOS: str = "\n<|assistant|>\n"
ASSISTANT_EOS: str = "</s>"
EMBED_BOS: str = "\n<|embed|>\n"

retrieve_action = "Action: Retrieve the next document.\n"

# Define ks for hits@k
ks = [1, 5, 10, 25, 100]

# Defaults (can be overridden with env vars)
DEFAULT_MODEL_BASE = os.environ.get("GRITHOPPER_MODEL_BASE", "GritLM/GritLM-7B")
DEFAULT_WEIGHTS = os.environ.get("GRITHOPPER_WEIGHTS", "UKPLab/GritHopper-7B")
DEFAULT_MUSIQUE_PATH = os.environ.get(
    "MUSIQUE_TRANSFORMED_PATH",
    "/mnt/beegfs/work/erker/reproduce/musique_dev_transformed.json",
)
RESULTS_DIR = os.environ.get("GRITHOPPER_RESULTS_DIR", "./grithopper_results")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define datasets to evaluate (Musique only for this script)
datasets = {
    "Musique": {
        "path": DEFAULT_MUSIQUE_PATH,
        "name": "Musique",
    }
}

# List of GritHopper checkpoints / repo-ids to evaluate (HF weights by default)
models = [DEFAULT_WEIGHTS]

# Initialize results dictionary with dataset-specific results
results: Dict[str, Any] = {
    "model": [],
    "datasets": {},
}

# Set the instruction
instruction = (
    "You answer multi-hop questions by iteratively retrieving documents. "
    "If a document is relevant, think by yourself which information is important to continue the search. "
    "If a document is irrelevant, stop the search. Once all information is extracted to answer the question, provide the final answer."
)


def resolve_weight_file(weights_root: str) -> Optional[str]:
    """
    Resolve a pytorch_model.bin for the given weights_root.
    Mirrors the HippoRAG loading: try local dir/file, then Hugging Face snapshot.
    """
    # 1) Local directory containing pytorch_model.bin
    candidate_dir = weights_root
    if os.path.isdir(candidate_dir):
        maybe_bin = os.path.join(candidate_dir, "pytorch_model.bin")
        if os.path.exists(maybe_bin):
            return maybe_bin

    # 2) Direct .bin path
    if os.path.isfile(weights_root) and weights_root.endswith(".bin"):
        return weights_root

    # 3) Hugging Face snapshot download
    if snapshot_download is None:
        return None

    try_repo_ids = (
        [weights_root, "UKPLab/GritHopper-7B"]
        if weights_root != "UKPLab/GritHopper-7B"
        else [weights_root]
    )

    for repo_id in try_repo_ids:
        try:
            local_dir = snapshot_download(
                repo_id=repo_id, ignore_patterns=["*.md", "static/*"]
            )
            maybe_bin = os.path.join(local_dir, "pytorch_model.bin")
            if os.path.exists(maybe_bin):
                print(f"Downloaded GritHopper snapshot to {local_dir} using {repo_id}")
                return maybe_bin
        except Exception as ie:
            last_error = ie  # noqa: F841
            continue

    return None


def load_grithopper_model(weights_root: str, device: str = DEFAULT_DEVICE) -> GritLM:
    """
    Load the GritLM base model and apply GritHopper weights from HF/local,
    matching the HippoRAG loading path.
    """
    model = GritLM(
        model_name_or_path=DEFAULT_MODEL_BASE,
        mode="unified",
        pooling_method="mean",
        normalized=True,
        projection=None,
        is_inference=True,
        embed_eos="",
        attn="bbcc",
        torch_dtype=torch.bfloat16,
    )

    weight_file = resolve_weight_file(weights_root)
    if weight_file and os.path.exists(weight_file):
        try:
            state = torch.load(weight_file, map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"Loaded GritHopper weights from {weight_file}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load weights from {weight_file}: {exc}")
    else:
        print(f"No weight file found for {weights_root}; using base model only.")

    try:
        model.to(device)
    except Exception:
        # Some gritlm versions wrap the model attribute
        try:
            model.model.to(device)
        except Exception as exc:
            print(f"Could not move model to {device}: {exc}")

    return model


def generate_rows_for_model_data(model_data: Dict[str, Any]) -> List[str]:
    """
    Given the model_data dictionary, generate LaTeX table rows for each dataset.
    Returns a list of strings, each representing a LaTeX table row for a dataset.
    """
    model_name = model_data.get("model_name", "model")
    rows = []

    for dataset_name, dataset_data in model_data.get("datasets", {}).items():
        # Extract hits_at_k_data and sample counts
        hits_at_k_data = dataset_data["hits_at_k"]
        n_samples = dataset_data["n_samples"]
        n_one_step = dataset_data.get("n_one_step", 0)
        n_two_step = dataset_data.get("n_two_step", 0)
        n_three_step = dataset_data.get("n_three_step", 0)
        n_four_step = dataset_data.get("n_four_step", 0)

        # Cumulative samples at each hop
        total_samples = n_one_step + n_two_step + n_three_step + n_four_step
        n_two_four_steps = n_two_step + n_three_step + n_four_step
        n_three_four_steps = n_three_step + n_four_step
        n_four_steps = n_four_step

        # Initialize row values with the model name and dataset name
        row_values = [model_name, dataset_name]

        # List of HITs@K values, ensuring correct order
        hits_k = ["1", "5", "10"]

        # Calculate percentages for each Hits@K and hop, then add to row
        for k in hits_k:
            # Step 1 for Hits@K=k
            step1_value = (
                hits_at_k_data["step1"].get(k, 0) / total_samples * 100
                if total_samples > 0
                else "-"
            )
            row_values.append(f"{step1_value:.2f}" if step1_value != "-" else "-")

            # Step 2 for Hits@K=k
            step2_value = (
                hits_at_k_data["step2"].get(k, 0) / n_two_four_steps * 100
                if n_two_four_steps > 0
                else "-"
            )
            row_values.append(f"{step2_value:.2f}" if step2_value != "-" else "-")

            # Step 3 for Hits@K=k
            step3_value = (
                hits_at_k_data["step3"].get(k, 0) / n_three_four_steps * 100
                if n_three_four_steps > 0
                else "-"
            )
            row_values.append(f"{step3_value:.2f}" if step3_value != "-" else "-")

            # Step 4 for Hits@K=k
            step4_value = (
                hits_at_k_data["step4"].get(k, 0) / n_four_steps * 100
                if n_four_steps > 0
                else "-"
            )
            row_values.append(f"{step4_value:.2f}" if step4_value != "-" else "-")

            # Calculate weighted average for Hits@K=k
            first_acc = (
                hits_at_k_data["step1"].get(k, 0) / total_samples
                if total_samples > 0
                else 0
            )
            second_acc = (
                hits_at_k_data["step2"].get(k, 0) / n_two_four_steps
                if n_two_four_steps > 0
                else 0
            )
            third_acc = (
                hits_at_k_data["step3"].get(k, 0) / n_three_four_steps
                if n_three_four_steps > 0
                else 0
            )
            fourth_acc = (
                hits_at_k_data["step4"].get(k, 0) / n_four_steps
                if n_four_steps > 0
                else 0
            )

            weighted_avg_acc = (
                (first_acc * total_samples)
                + (second_acc * n_two_four_steps)
                + (third_acc * n_three_four_steps)
                + (fourth_acc * n_four_steps)
            ) / (
                total_samples + n_two_four_steps + n_three_four_steps + n_four_steps
            ) * 100

            # Append formatted weighted average
            row_values.append(
                "\\cellcolor{lightgray}\\itshape "
                + (f"{weighted_avg_acc:.2f}" if weighted_avg_acc != 0 else "-")
            )

        # Join row values with '&' and add the ending '\\' without any extra escaping
        row_str = " & ".join(row_values) + " \\\\"
        rows.append(row_str)

    return rows


def describe_hits_calc(dataset_results: Dict[str, Any], dataset_name: str) -> None:
    total_samples = (
        dataset_results.get("n_one_step", 0)
        + dataset_results.get("n_two_step", 0)
        + dataset_results.get("n_three_step", 0)
        + dataset_results.get("n_four_step", 0)
    )
    print(
        f"\nHits@1 calculation for {dataset_name}: "
        "step1 counts / all samples; "
        "step2 / samples with >=2 hops; "
        "step3 / samples with >=3 hops; "
        "step4 / samples with 4 hops."
    )
    print(f"Sample counts -> total: {total_samples}, "
          f"1-hop: {dataset_results.get('n_one_step', 0)}, "
          f"2-hop: {dataset_results.get('n_two_step', 0)}, "
          f"3-hop: {dataset_results.get('n_three_step', 0)}, "
          f"4-hop: {dataset_results.get('n_four_step', 0)}")


# Iterate over each model
os.makedirs(RESULTS_DIR, exist_ok=True)
for m in tqdm(models, desc="Evaluating models"):
    # Extract both model name and checkpoint number from path/repo
    tokens = m.rstrip("/").split("/")
    if len(tokens) >= 2 and tokens[-1].startswith("tmp-checkpoint-"):
        model_name = tokens[-2]
        checkpoint_num = tokens[-1].replace("tmp-checkpoint-", "")
        model_identifier = f"{model_name}_checkpoint{checkpoint_num}"
    else:
        model_name = tokens[-1]
        checkpoint_num = ""
        model_identifier = model_name

    # Initialize model-specific results
    model_results = {
        "model_path": m,
        "model_name": model_name,
        "checkpoint": checkpoint_num,
        "datasets": {},
    }

    # Load model once for all datasets (HF-style, like HippoRAG)
    model = load_grithopper_model(m, device=DEFAULT_DEVICE)

    # Evaluate on each dataset
    for dataset_name, dataset_info in datasets.items():
        print(f"\nEvaluating {model_name} on {dataset_name}")

        # Load dataset
        try:
            with open(dataset_info["path"], "r") as f:
                dataset = json.load(f)
        except FileNotFoundError:
            print(f"Dataset {dataset_name} not found at {dataset_info['path']}")
            continue

        # Initialize dataset results
        dataset_results = {
            "hits_at_k": {
                "step1": {str(k): 0 for k in ks},
                "step2": {str(k): 0 for k in ks},
                "step3": {str(k): 0 for k in ks},
                "step4": {str(k): 0 for k in ks},
            },
            "n_samples": len(dataset),
            "n_one_step": 0,
            "n_two_step": 0,
            "n_three_step": 0,
            "n_four_step": 0,
        }

        # Collect all unique evidence facts and their metadata
        unique_evidences = {}
        for sample in dataset:
            for evidence in sample["evidence_list"]:
                fact = evidence["title"] + ". " + evidence["fact"]
                if fact not in unique_evidences:
                    unique_evidences[fact] = {
                        "meta_data": evidence,
                        "queries": [sample["query"]],
                    }
                else:
                    unique_evidences[fact]["queries"].append(sample["query"])

        # Create a list of all unique evidence facts
        all_evidence_facts = list(unique_evidences.keys())

        # Encode all evidence facts
        print(f"Encoding all evidence facts using model {m}")
        all_evidence_facts_encoded = []
        for fact in tqdm(all_evidence_facts, desc="Encoding evidences"):
            encoded_fact = model.encode(
                fact,
                instruction=gritlm_instruction("Represent the document: ", EMBED_BOS),
                convert_to_tensor=True,
                max_length=2048,
            )
            all_evidence_facts_encoded.append(encoded_fact.cpu())

        # Stack encoded facts into a single tensor
        all_evidence_facts_encoded = torch.stack(all_evidence_facts_encoded)

        # Build Faiss index (GPU if available, CPU otherwise)
        print(f"Building Faiss index for model {m}")
        dimension = all_evidence_facts_encoded.shape[1]
        use_gpu = faiss.get_num_gpus() > 0 and torch.cuda.is_available()
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            search_index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            search_index = faiss.IndexFlatL2(dimension)
        search_index.add(all_evidence_facts_encoded.numpy())

        # Evaluate each sample
        for sample in tqdm(dataset, desc=f"Evaluating samples with model {m}"):
            n_hop_questions = len(sample["evidence_list"])

            # Update counts of n-step questions
            if n_hop_questions == 1:
                dataset_results["n_one_step"] += 1
            elif n_hop_questions == 2:
                dataset_results["n_two_step"] += 1
            elif n_hop_questions == 3:
                dataset_results["n_three_step"] += 1
            elif n_hop_questions == 4:
                dataset_results["n_four_step"] += 1

            # Initialize prompt
            query = sample["query"]
            prompt = f"Question: {query}\n{retrieve_action}"

            # Copy the list of evidence facts to manipulate during evaluation
            remaining_evidence_indices = list(range(len(all_evidence_facts)))
            evidence_encodings = all_evidence_facts_encoded.clone()

            target_evidences = set(
                evidence["title"] + ". " + evidence["fact"]
                for evidence in sample["evidence_list"]
            )
            remaining_target_evidences = target_evidences.copy()

            # Track failed k values for this question
            failed_k_values = set()

            # Evaluate each hop
            for step in range(n_hop_questions):
                step_key = f"step{step + 1}"

                # Encode the current prompt
                encoded_query = model.encode(
                    prompt,
                    instruction=gritlm_instruction(instruction, EMBED_BOS),
                    convert_to_tensor=True,
                    max_length=2048,
                ).cpu()

                # Compute cosine similarities
                similarities = torch.nn.functional.cosine_similarity(
                    encoded_query, evidence_encodings, dim=1
                )

                # Find the first positive passage in any k
                retrieved_evidence = None
                retrieved_idx = None

                # For each k, find positive passages
                for k in ks:
                    # Skip this k if we failed in a previous step
                    if k in failed_k_values:
                        continue

                    top_k = min(k, similarities.shape[0])
                    topk_values, topk_indices = torch.topk(
                        similarities, top_k, largest=True
                    )
                    retrieved_indices = topk_indices.tolist()

                    # Check which evidences are found in top k
                    current_found = set()
                    for idx in retrieved_indices:
                        fact = all_evidence_facts[remaining_evidence_indices[idx]]
                        if fact in remaining_target_evidences:
                            current_found.add(fact)
                            # Store the first positive passage found for actual retrieval
                            if retrieved_evidence is None:
                                retrieved_evidence = fact
                                retrieved_idx = idx

                    # Update successful retrievals for this k
                    if current_found:
                        dataset_results["hits_at_k"][step_key][str(k)] += 1
                    else:
                        # If nothing found for this k, mark it as failed
                        failed_k_values.add(k)

                # If no positive passage was found in any k, break
                if retrieved_evidence is None:
                    break

                # Update prompt with the retrieved evidence
                prompt += f"Document: {retrieved_evidence}\n"
                prompt += "Action: Evaluating retrieved Document: Relevant.\n"
                prompt += "Extracted information: Think yourself\n"
                prompt += retrieve_action

                # Remove the retrieved evidence from consideration
                evidence_encodings = torch.cat(
                    [
                        evidence_encodings[:retrieved_idx],
                        evidence_encodings[retrieved_idx + 1 :],
                    ],
                    dim=0,
                )
                del remaining_evidence_indices[retrieved_idx]
                remaining_target_evidences.remove(retrieved_evidence)

                # Break if all evidences have been retrieved
                if len(remaining_target_evidences) == 0:
                    break

        # Store dataset results
        model_results["datasets"][dataset_name] = dataset_results

        # Save intermediate results after each dataset evaluation
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_path = os.path.join(
            RESULTS_DIR, f"{model_identifier}_{dataset_name}_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(model_results, f, indent=4)

        describe_hits_calc(dataset_results, dataset_name)

    # Save LaTeX rows / aggregated hits including hits@1
    rows = generate_rows_for_model_data(model_results)
    rows_path = os.path.join(RESULTS_DIR, f"{model_identifier}_rows.txt")
    with open(rows_path, "w") as f:
        for row in rows:
            f.write(row + "\n")
    print("\nHits@K aggregation (LaTeX rows):")
    for row in rows:
        print(row)

    results["model"].append(model_results)
    print(f"\nModel {model_name} evaluation completed on all datasets")

print("All models evaluated on all datasets")
