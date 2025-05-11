# COHS Hazards VQA: Visual Question Answering for Construction Occupational Health and Safety Hazards
![035261c0-ad80-452a-8f14-ff80de294435](https://github.com/user-attachments/assets/04a74a44-9541-4488-9c85-1ba5d5973549)
# Project Overview📌
This repository contains the implementation of a visual knowledge enhancement framework of multimodal large language models (MLLMs) for construction occupational health and safety (COHS) hazards  visual question answering (VQA)  via retrieval augmented generation (RAG).
# Framework Components🚀
Our framework integrates the following key modules:
1) COHS Knowledge Bases : Domain-specific safety regulations and best practices.
2) Visual Semantic Anchor Extraction : Identifies critical visual elements relevant to safety hazards.
3) Agentic Decision-Making Dual-Stage Retrieval : Dynamically selects relevant knowledge based on query complexity.
4) Chunked Knowledge Delivery : Prevents cognitive overload by managing information flow to the model.
# Performance Highlights🔍
| Question type | Metric | Improevment vs baseline |
| :--- | :---: | ---: |
| Binary classification quesstion | Recall ↑ | +95.28% |
| Counting question | Accuracy ↑ | +194.13% |
| Open-ended question | Recall ↑ | +156.23% |

These results highlight the transformative potential of our framework in enhancing safety monitoring of MLLMs at construction sites.
# COHSD📦
In view of the absence of the relevant dataset, this study establishes the first construction industry-oriented visual benchmark dataset for COHS, termed COHSD, so as to systematically validate the effectiveness of the proposed method. COHSD is sourced from three categories: 1) site images containing COHS hazards collected from 12 Chinese construction projects; 2) construction images containing COHS hazards downloaded and filtered from the web; 3) a large-scale image dataset, Site Object Detection Dataset (SODA), specifically compiled and annotated for construction sites. COHSD comprises a total of 226 construction images, among which 53 images are classified as safe, and 173 images contain COHS hazards. The COHS hazard-containing images are categorized into two main types: construction objects and operations, comprehensively covering the predefined 15 types of construction objects and 20 types of operational scenarios. 
![c80f997a635c9784629041ca5c5b5a2](https://github.com/user-attachments/assets/d7a06a8b-82b0-4c47-9148-9e41a7389639)
<div align="center">Figure1：Images containing hazards number</div>

# Citation📚
If you use this code or dataset in your research, please cite our paper:
```
@article{yourpaper2025,
  title={Visual Knowledge Enhancement Framework for COHS Hazards VQA Based on MLLMs and RAG},
  author={L Yang et al.},
  year={2025}
}
```
# Contact🤝
If you have any questions or would like to contribute, feel free to open an issue or reach out via email: A1371599104@163.com.
