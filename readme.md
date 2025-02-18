
[![GitHub - License](https://img.shields.io/github/license/UKPLab/arxiv2024-triple-encoders?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/triple-encoders?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/triple-encoders?logo=pypi&style=flat&color=orange)][#pypi-package]
<p align="center">
  <img src="static/GritHopperLogo.jpeg" alt="GritHopper Logo" height="250px" align="left" style="position: relative; z-index: 1;">
  <div align="center">
    <h1>
      <h1>GritHopper: Decomposition-Free<br>
      Multi-Hop Dense Retrieval</h1>
    </h1>
  </div>
</p>

<br clear="left"/>
<p align="center">
    🤗 <a href="https://huggingface.co/UKPLab/triple-encoders-dailydialog" target="_blank">Models</a>  | 📃 <a href="https://aclanthology.org/2024.acl-long.290/" target="_blank">Paper</a>
</p>
<!--- BADGES: START, copied from sentence transformers, will be replaced with the actual once (removed for anonymity)--->

---



GritHopper is a **state-of-the art multi-hop dense retrieval framework** that builds upon **GRITLM**. GritHopper is the first decoder-based decomposition-free multi-hop dense retrieval model, trained on a multi-hop tasks spanning both question-answering and fact-checking. By leveraging an **encoder-only** paradigm (similar to MDR), GritHopper requires a single forward pass for one hop while maintaining strong generalizability across out-of-distribution tasks. 

## Key Strengths of GritHopper
- **Scalable Multi-Hop**: Handles multi-hop questions in **open-domain** scenarios without relying on discrete decomposition steps.  
- **Encoder-Only Efficiency**: Each retrieval iteration requires only a single forward pass (rather than multiple autoregressive steps).  
- **Out-of-Distribution Robustness**: Achieves **state-of-the-art** performance on multiple OOD benchmarks.  
- **Unified Training**: Combines dense retrieval with generative objectives, exploring how to adapt ReAct as a state-control for the dense retrieval using causal generation. 
- **Easy Integration**: Functions as a Python library that lets you **load document candidates**, **encode queries**, and **iteratively retrieve** relevant passages with optional stopping logic.

---

## Staring with GritHopper

### GritHopper Models 
| Model Name                          | Datasets     | Description                                                                                                                                                              | Model Size |
|-------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| GritHopper-7B                | All Datasets | GritHopper trained on Answers as Post-Retrieval information (SOTA)                                                                                                       | 7B |
### 1. Installation

```bash
pip install grithopper
```
### 2. Initialization
```python
from grithopper import GritHopper

# Initialize GritHopper with your GRITLM model checkpoint or huggingface path
hopper = GritHopper(
    model_name_or_path="GritHopper-7B",  
    device="cuda"  # or "cpu"
)
```

### 3. Load Document Candidates

You can either load from a list of (title, passage) pairs and optionally dump them to a file:
```python
documents = [
    ("Title A", "Passage text for document A."),
    ("Title B", "Passage text for document B."),
    # ...
]

hopper.load_document_candidates(
    document_candidates=documents,
    device="cuda",
    output_directory_candidates_dump="my_candidates.pkl"  # optional
)
```

Or load them from a pre-encoded dump:
```python
hopper.load_candidates_from_file(
    dump_filepath="my_candidates.pkl",
    device="cuda"
)
```
### 4. Encode a Query

```python
question = "Who wrote the novel that was adapted into the film Blade Runner?"
previous_evidences = [("Blade Runner (Movie)", " The Movie....")] # optional


query_vector = hopper.encode_query(
    multi_hop_question=question,
    previous_evidences=previous_evidences, # optional
    instruction_type="multi-hop"  # or "fact-check" alternatively you can provide a custom instruction with insruction="your_instruction"
)
```
### 5. Single-Step Retrieval
```python
result = hopper.retrieve_(
    query=query_vector,
    top_k=1,
    get_stopping_probability=True
)

# {
#   "retrieved": [
#       {
#         "title": "Title B",
#         "passage": "Passage text for document B.",
#         "score": 0.873
#       }
#   ],
#   "continue_probability": 0.65,  # present if get_stopping_probability=True
#   "stop_probability": 0.35
# }
```
If you prefer to pass the question string directly:

```python
result = hopper.retrieve_(
    query="Who is the mother of the writer who wrote the novel that was adapted into the film Blade Runner?",
    # optional previous_evidences=[("Blade Runner (Movie)", " The Movie....")],
    top_k=1,
    get_stopping_probability=True,
)

# {
#   "retrieved": [
#       { "title": "Blade Runner (Movie)", "passage": "...", "score": 0.92 }
#   ],
#   "continue_probability": 0.75,
#   "stop_probability": 0.25
# }
```
### 6. Iterative (Multi-Hop) Retrieval
```python
chain_of_retrieval = hopper.iterative_retrieve(
    multi_hop_question="Who wrote the novel that was adapted into the film Blade Runner?",
    instruction_type="multi-hop",
    automatic_stopping=True,
    max_hops=4
)

# [
#   {
#     "retrieved": [
#       { "title": "Blade Runner (Movie)", "passage": "...", "score": 0.92 }
#     ],
#     "continue_probability": 0.75,
#     "stop_probability": 0.25
#   },
#   {
#     "retrieved": [
#       { "title": "Philip K.", "passage": "...", "score": 0.88 }
#     ],
#     "continue_probability": 0.65,
#     "stop_probability": 0.35
# },
#   ...
# ]
```
This process continues until either:

	1.	The model determines it should stop (if automatic_stopping=True and stop_probability > continue_probability).
	2.	It hits max_hops.
	3.	Or no documents can be retrieved at a given step.

---
### Training 
for training, we used GRITLM.
