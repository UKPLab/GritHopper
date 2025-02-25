<!--- BADGES: START, copied from sentence transformers, will be replaced with the actual once (removed for anonymity)--->
[![GitHub - License](https://img.shields.io/github/license/UKPLab/GritHopper?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/grithopper?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/grithopper?logo=pypi&style=flat&color=orange)][#pypi-package]


[#github-license]: https://github.com/
[#pypi-package]: https://pypi.org/project/grithopper/
<p align="center">
  <img src="static/GritHopperLogo.jpeg" alt="GritHopper Logo" height="250px" align="left" style="position: relative; z-index: 1;">
  <div align="center">
    <h1>
      <h1>GritHopper: Decomposition-Free<br>
      Multi-Hop Dense Retrieval</h1>
    </h1>
    <p align="center">
    🤗 <a href="https://huggingface.co/UKPLab/GritHopper" target="_blank">Models</a>  | 📃 <a href="TBD" target="_blank">Paper</a>
</p>
  </div>
</p>

<br clear="left"/>

<!--- BADGES: START, copied from sentence transformers, will be replaced with the actual once (removed for anonymity)--->

---

GritHopper is the first **decoder-based** multi-hop dense retrieval model and achieves **state-of-the-art performance** on both **in-distribution and out-of-distribution** benchmarks for **decomposition-free multi-hop dense retrieval**. Built on [GRITLM](https://github.com/ContextualAI/gritlm), it is trained across diverse datasets spanning **question-answering** and **fact-checking**. Unlike traditional decomposition-based approaches, GritHopper iteratively retrieves passages without explicit sub-question decomposition, concatenating retrieved evidence with the query at each step.

Using the decoder model in an encoder-only approach (like [MDR](https://github.com/facebookresearch/multihop_dense_retrieval)), it performs each retrieval step in a single forward pass. In contrast to previous SOTA BERT-based approaches (like [BeamRetriever](https://github.com/canghongjian/beam_retriever) or [MDR](https://github.com/facebookresearch/multihop_dense_retrieval)), GritHopper generalizes significantly better to **out-of-distribution** data.

## Key Strengths of GritHopper
- **Encoder-Only Efficiency**: Each retrieval iteration requires only a single forward pass (rather than multiple autoregressive steps).  
- **Out-of-Distribution Robustness**: Achieves **state-of-the-art** performance compared to other decomposition-free methods on multiple OOD benchmarks.  
- **Unified Training**: Combines dense retrieval with generative objectives, exploring how post-retrieval information on the generation loss improves dense retrieval performance. 
- **Stopping**: GritHopper utilizes its generative capabilities via ReAct to control its own state. This way, it can stop itself through causal next-token prediction. 

---

## Staring with GritHopper
GritHopper is trained on [MuSiQue](https://aclanthology.org/2022.tacl-1.31/), [2WikiMultiHopQA](https://aclanthology.org/2020.coling-main.580.pdf), [HotPotQA](https://aclanthology.org/D18-1259.pdf), [EX-Fever](https://aclanthology.org/2024.findings-acl.556/) and [HoVer](https://aclanthology.org/2020.findings-emnlp.309/). 

### GritHopper Models 
| Model Name                          | Datasets     | Description                                                                                                                                                              | Model Size |
|-------------------------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| [GritHopper-7B](https://huggingface.co/UKPLab/GritHopper)                | All Datasets | GritHopper trained on Answers as Post-Retrieval information (SOTA)                                                                                                       | 7B |
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
## Citation
If you use GritHopper in your research, please cite the following paper:
```
TBD
```
## Contact
Contact person: Justus-Jonas Erker, justus-jonas.erker@tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

## License
GritHopper is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.


### Acknowledgement
this Model is based upon the [GRITLM](https://github.com/ContextualAI/gritlm). 
