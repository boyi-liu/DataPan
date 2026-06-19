# Data Selector for LLM Finetuning 
This repository contains code and tools for curating datasets for fine-tuning large language models (LLMs). 
The goal is to provide a streamlined process for collecting, cleaning, and preparing data that can be used to enhance the performance of LLMs on specific tasks or domains.

### Supported Algorithms
+ **[NAACL 2024]** IFD (Cherry LLM): From Quantity to Quality — Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning.
+ **[ICML 2024]** LESS: Selecting Influential Data for Targeted Instruction Tuning.
+ **[ICLR 2026]** ADAPT: Adaptive Data reweighting for Pretraining and FineTuning — online per-sample reweighting instead of offline subset selection.
+ **[AAAI 2026]** MIWV: Importance-Aware Data Selection for Efficient LLM Instruction Tuning — rank samples by the ICL-based Model Instruction Weakness Value.

We also support some basic baselines:
+ Random selection.
+ BM25-based selection.
+ Embedding-based selection.
+ Perplexity-based selection.

### Supported Datasets
+ Alpaca
+ WizardLM
+ LESS (Mixture of Flan V2, CoT, Dolly and Open Assistant)
