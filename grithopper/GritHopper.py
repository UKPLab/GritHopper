# grithopper.py

import os
import pickle
from typing import List, Dict, Optional, Union, Tuple

import torch
from gritlm import GritLM

# Some defaults from your original code
BASE_BOS: str = "<s>"
USER_BOS: str = "<|user|>\n"
USER_EOS: str = ""
ASSISTANT_BOS: str = "\n<|assistant|>\n"
ASSISTANT_EOS: str = "</s>"
EMBED_BOS: str = "\n<|embed|>\n"
EMBED_EOS: str = ""

DEFAULT_MULTI_HOP_INSTRUCTION = (
    "You answer multi-hop questions by iteratively retrieving documents. "
    "If a document is relevant, think by yourself which information is important to continue the search. "
    "If a document is irrelevant, stop the search. Once all information is extracted to answer the question, "
    "provide the final answer."
)

DEFAULT_FACT_CHECK_INSTRUCTION = (
     "You verify multi-hop fact checking claims by iteratively retrieving documents. "
     "If a document is relevant, think by yourself which information is important to continue the search. "
     "If a document is irrelevant, stop the search. Once all information is extracted to verify the claim, provide whether it is supported or not."
)

document_gap = "Action: Evaluating retrieved Document: Relevant. \n" + "Extracted information: " + "Think yourself" + "\n"
# We also tried shorter versions but this one worked best, probably because it adds more compute

retrieve_action = "Action: Retrieve the next document.\n"

def gritlm_instruction(instruction: str) -> str:
    """
    Official GritLM format for embedding
    """
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

class GritHopper:
    """
    A wrapper class around GritLM for multi-hop retrieval scenarios.
    """

    def __init__(
        self,
        model_name_or_path: str,
        base_model_name: str = "GritLM/GritLM-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn: str = "bbcc",
        pooling_method: str = "mean",
        normalized: bool = True,
        embed_eos: str = "",
        is_inference: bool = True,
        projection: Optional[str] = None,
    ):
        """
        Initialize the GritHopper wrapper.
        :param model_name_or_path: Path or name of the Hugging Face model.
        :param device: Device to run the model on.
        :param torch_dtype: Torch dtype (float16, bfloat16, etc.)
        :param attn: attention setting from your GritLM code
        :param pooling_method: embedding pooling method
        :param normalized: whether to L2-normalize embeddings
        :param embed_eos: optional end-of-sequence token for embeddings
        :param is_inference: set to True if you are loading a checkpoint for inference
        :param projection: optional model projection layer setting
        """
        self.device = device
        self.model = GritLM(
            model_name_or_path=base_model_name,
            mode="unified",
            pooling_method=pooling_method,
            normalized=normalized,
            projection=projection,
            is_inference=is_inference,
            embed_eos=embed_eos,
            attn=attn,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = self.model.tokenizer


        print(f"Loading fine-tuned weights from Hugging Face Hub: {model_name_or_path}...")
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{model_name_or_path}/resolve/main/pytorch_model.bin",
            map_location=self.device
        )

        # Apply fine-tuned weights
        self.model.load_state_dict(state_dict, strict=False)

        # model to device
        self.model.to(self.device)

        self.model.eval()

        self.doc_embeddings = None  # Will store pre-encoded documents
        self.doc_texts = None       # Will store corresponding (title, passage) pairs
        self.stopping_ids = {
            "id_continue": 8637,
            "id_finished": 2595,
            "action_suffix": ":"
        }

    def load_document_candidates(
        self,
        document_candidates: List[Tuple[str, str]],
        device: str = "cuda",
        output_directory_candidates_dump: Optional[str] = None,
    ):
        """
        Takes a list of (title, passage) pairs, encodes them, and stores in memory.
        If `output_directory_candidates_dump` is provided, will also save them to file.
        :param document_candidates: List of (title, passage) pairs.
        :param device: Device used for encoding.
        :param output_directory_candidates_dump: Filepath to dump the index to disk.
        """
        self.doc_texts = document_candidates
        self.doc_embeddings = []

        for title, passage in document_candidates:
            # Use official GritLM embed format
            text = title + ". " + passage
            embedding = self.model.encode(
                text,
                instruction=gritlm_instruction("Represent the document: "),
                convert_to_tensor=True,
                max_length=2048,
            ).to(device)
            self.doc_embeddings.append(embedding)

        self.doc_embeddings = torch.stack(self.doc_embeddings, dim=0).to(device)

        if output_directory_candidates_dump:
            dump_data = {
                "doc_texts": self.doc_texts,
                "doc_embeddings": self.doc_embeddings.cpu().numpy(),
            }
            with open(output_directory_candidates_dump, "wb") as f:
                pickle.dump(dump_data, f)

    def load_candidates_from_file(
        self,
        dump_filepath: str,
        device: str = "cuda"
    ):
        """
        Loads pre-encoded document candidates from a file (pickle).
        :param dump_filepath: Path to the pickled file containing doc_texts + embeddings.
        :param device: Device for the embeddings.
        """
        with open(dump_filepath, "rb") as f:
            dump_data = pickle.load(f)

        self.doc_texts = dump_data["doc_texts"]
        # Re-load embeddings to device
        self.doc_embeddings = torch.tensor(dump_data["doc_embeddings"], dtype=torch.float32).to(device)

    def encode_query(
        self,
        multi_hop_question: str,
        previous_evidences: Optional[List[Tuple[str, str]]] = None,
        instruction_type: Optional[str] = "multi-hop",
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encodes a multi-hop question (and optional evidence).
        If `instruction` is not provided, uses a default template based on instruction_type.
        :param multi_hop_question: The main question being asked.
        :param previous_evidences: List of (title, passage) pairs that have been retrieved so far.
        :param instruction_type: "multi-hop" or "fact-check". Extendable.
        :param instruction: Custom override instruction.
        :return: A single embedding (Tensor).
        """
        if instruction is None:
            if instruction_type == "fact-check":
                instruction = DEFAULT_FACT_CHECK_INSTRUCTION
            else:
                instruction = DEFAULT_MULTI_HOP_INSTRUCTION

        # Construct the base prompt
        base_prompt = f"Question: {multi_hop_question}\n"

        if previous_evidences:
            for title, passage in previous_evidences:
                base_prompt += f"Document: {title}. {passage}\n"
                base_prompt += document_gap + retrieve_action

        # Example style: "Action: Retrieve the next document."
        # For the embedding, we just do GritLM embed style:
        query_embedding = self.model.encode(
            base_prompt,
            instruction=gritlm_instruction(instruction),
            convert_to_tensor=True,
            max_length=2048
        )

        return query_embedding

    def retrieve_(
        self,
        query: Optional[Union[torch.Tensor, str]] = None,
        previous_evidences: Optional[List[Tuple[str, str]]] = None,
        instruction_type: Optional[str] = "multi-hop",
        instruction: Optional[str] = None,
        top_k: int = 1,
        get_stopping_probability: bool = False,
    ) -> Dict:
        """
        Retrieves top-k documents by cosine similarity to the query.
        Optionally computes a stopping probability with a forward pass if requested.
        :param query: Either a pre-computed Tensor embedding or a string (multi-hop question).
        :param previous_evidences: Provide if you want to incorporate previously retrieved evidence in the query embedding.
        :param instruction_type: Template type for building the instruction if query is a string.
        :param instruction: Manual override instruction if query is a string.
        :param top_k: number of passages to retrieve
        :param get_stopping_probability: if True, will do an additional causal LM step
               to check if we "finish" or "continue" retrieval,
               returning the relevant probabilities.
        :return: Dictionary with:
             {
               "retrieved": [
                  {"title": ..., "passage": ..., "score": ...}, ...
               ],
               "stop_probability": float (optional, only if get_stopping_probability=True),
               "continue_probability": float (optional, only if get_stopping_probability=True)
             }
        """
        # If query is a string, encode it
        if isinstance(query, str):
            query_embedding = self.encode_query(
                multi_hop_question=query,
                previous_evidences=previous_evidences,
                instruction_type=instruction_type,
                instruction=instruction
            )
        else:
            query_embedding = query  # assume it's already a Tensor

        # Cosine similarity with doc_embeddings
        sim_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.doc_embeddings,
            dim=1
        )

        # Get top_k indexes
        topk_vals, topk_idxs = torch.topk(sim_scores, k=top_k, largest=True)

        results = []
        for score_val, idx_val in zip(topk_vals, topk_idxs):
            idx_val = idx_val.item()
            score_val = score_val.item()
            title, passage = self.doc_texts[idx_val]
            results.append(
                {
                    "title": title,
                    "passage": passage,
                    "score": score_val
                }
            )

        output_dict = {"retrieved": results}

        if not previous_evidences:
            previous_evidences = [[results[0]["title"], results[0]["passage"]]]

        # add tuple to previous_evidences tuple list
        else:
            previous_evidences = previous_evidences + [[results[0]["title"], results[0]["passage"]]]

        if get_stopping_probability and results and isinstance(query, str):
            # We'll do a forward pass to compare the "continue" vs. "finish" token logit
            # Similar to your example with "Action: Evaluating retrieved Document: Relevant. \n"
            doc_info = results[0]  # if top_k>1, we might pick the first or do a different logic
            # Build a mini prompt
            content = f"Question: {query}\n"
            for title, passage in previous_evidences:
                content += f"Document: {title}. {passage}\n"
                content += document_gap
                if previous_evidences[-1] != (title, passage):
                    content += retrieve_action
            # remove last retrieve ac
            content += f"Action{self.stopping_ids['action_suffix']}"

            decision_prompt = (
                BASE_BOS
                + USER_BOS + (instruction or DEFAULT_MULTI_HOP_INSTRUCTION) + USER_EOS
                + ASSISTANT_BOS + content
            )

            encoded_input = torch.tensor([self.tokenizer.encode(decision_prompt)]).to(self.device)
            with torch.no_grad():
                outputs = self.model.model(input_ids=encoded_input)
                logits = outputs.logits[0, -1, :]

            id_continue = self.stopping_ids["id_continue"]
            id_finished = self.stopping_ids["id_finished"]

            continue_score = logits[id_continue].item()
            finish_score = logits[id_finished].item()

            # Convert raw logits to probabilities
            probs = torch.softmax(torch.tensor([continue_score, finish_score]), dim=0).tolist()
            output_dict["continue_probability"] = probs[0]
            output_dict["stop_probability"] = probs[1]

        return output_dict

    def iterative_retrieve(
        self,
        multi_hop_question: str,
        previous_evidences: Optional[List[Tuple[str, str]]] = None,
        instruction_type: Optional[str] = "multi-hop",
        instruction: Optional[str] = None,
        automatic_stopping: bool = False,
        max_hops: int = 4,
    ) -> List[Dict]:
        """
        Iteratively retrieve documents for a multi-hop question.
        Optionally use automatic stopping after each retrieval via get_stopping_probability.
        :param multi_hop_question: The query or question to retrieve docs for.
        :param previous_evidences: Already retrieved documents (title, passage) pairs.
        :param instruction_type: "multi-hop", "fact-check", or custom.
        :param instruction: Manual override if needed.
        :param automatic_stopping: If True, will do a forward call each iteration
               to see if we should continue or stop.
        :param max_hops: The maximum number of retrieval iterations allowed.
        :return: A list of dictionaries describing each retrieval step:
            [
              {
                "retrieved": {"title": ..., "passage": ..., "score": ...},
                "continue_probability": ...,
                "stop_probability": ...
              }, ...
            ]
        """
        if previous_evidences is None:
            previous_evidences = []

        retrieval_chain = []

        for step in range(max_hops):
            # Retrieve top1
            retrieval_result = self.retrieve_(
                query=multi_hop_question,
                previous_evidences=previous_evidences,
                instruction_type=instruction_type,
                instruction=instruction,
                top_k=1,
                get_stopping_probability=automatic_stopping
            )

            # If we have at least one retrieved doc, add it to the chain
            if retrieval_result["retrieved"]:
                top_doc = retrieval_result["retrieved"][0]
                retrieval_chain.append(retrieval_result)

                # You could decide relevancy here. In your code example,
                # you rely on is_supporting. But we assume we don't know that at inference time.

                # If automatic_stopping is enabled, we can check the stop_probability
                if automatic_stopping:
                    stop_prob = retrieval_result.get("stop_probability", 0.0)
                    cont_prob = retrieval_result.get("continue_probability", 1.0)
                    # If model decides to stop
                    if stop_prob >= cont_prob:
                        break

                # Otherwise, continue
                # Add top doc to the "previous_evidences" so the next query sees it
                previous_evidences.append((top_doc["title"], top_doc["passage"]))
            else:
                # No documents found? Just break
                break

        return retrieval_chain
