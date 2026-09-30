# Conversational Text-to-SQL: Thesis Extensions to Semantic Grounding

This research fork extends [Semantic Grounding over Brute-Force Inference: A Symbiotic Evaluation of Text-to-SQL Systems](https://github.com/michelepao1993-dev/semantic-grounding-text2sql), the framework by Michele Paolicelli and collaborators.

The extensions were developed by **Mirco Catalano** for his thesis. They investigate clarification of ambiguous requests, execution feedback, and database-specific FAQ retrieval within a single conversational Text-to-SQL agent.

The original framework and evaluation metrics belong to the upstream project. This README describes the implementation and experimental artifacts in this version, rather than claiming to reproduce all experiments from the original paper.

## Original framework and thesis contributions

The upstream framework already provides a monolithic Text-to-SQL baseline, schema prompting, semantic grounding rules, conversation state, Jaccard-based few-shot retrieval, candidate generation, SQL execution, and cost-aware evaluation.

This version adds or extends the following components:

| Component | Thesis extension |
| --- | --- |
| `ExecutionFeedbackAgent` | Generates an initial SQL query, executes it, and uses a database error or a result sample for one revision with the same LLM. |
| `ClarificationAgent` | Chooses between `ANSWER` and `CLARIFY` and supports a subsequent user clarification. |
| `ConversationalExecutionFeedbackAgent` | Integrates clarification, FAQ retrieval, and execution feedback in one agent, using the existing conversation state to retain a pending question. |
| Revision policies | Supports `always`, `never`, and `on_failure_or_empty`. The adaptive policy also handles results containing only null values. |
| Database-specific FAQs | Loads question/SQL pairs from JSON and retrieves candidates using normalized-token Jaccard similarity, a threshold, and top-k selection. The LLM assesses whether a retrieved FAQ matches the request. |
| Configurable semantic grounding | Moves the existing database-specific rules into external text files and enables or disables them through `semantic_enrichment`. |
| Experiment runners | Adds execution-feedback and clarification evaluation, an integrated conversational runner, evaluation offsets, and the option to disable conversation memory. |
| SQL execution | Enables SQLite `PRAGMA query_only = ON` on execution connections. |
| Experimental artifacts | Adds three clarification holdout sets and twelve JSONL logs for the evaluated configurations. |
| Extended logging | Records clarification decisions, initial/final query correctness, useful and harmful revisions, LLM calls, token usage, and latency. |

The combined agent uses a code-controlled flow: decide whether clarification is needed, incorporate a user response if supplied, execute the initial SQL, and optionally revise it once. It does not implement a multi-agent system or an autonomous retry loop. Evaluation runners also execute queries to compute correctness metrics.

## Repository structure

```text
agents/
    baseline_agent.py
    clarification_agent.py
    execution_feedback_agent.py
    conversational_execution_feedback_agent.py
    telemetry.py
    types.py
analysis/                         # Inherited aggregation utilities
data/                            # Python loaders and local Spider dataset
    __init__.py
    schema_text.py
    spider_loader.py
    spider/                       # Download separately
enrichment/
    grounding/                    # Database-specific semantic rules
    faqs/                         # Questions and canonical SQL
evaluation/                      # Inherited evaluation metrics
evaluation_sets/                 # Three clarification holdout sets
execution/sql_executor.py
experiment_logging/
faq/                             # FAQ loading and retrieval
fewshot_retrieval/
models/openai_chat_model.py
outputs/conversational_execution_feedback/
runner/
    db_setup.py
    experiment_runner.py
    clarification_runner.py
    conversational_execution_feedback_runner.py
    grid_runner.py
    turn_runner.py
LICENSE.txt
requirements.txt
```

The upstream notebook notebooks/Experiments.ipynb and experimental logs under outputs/logs_clean/ are retained alongside the thesis extensions.

## Installation

Clone this repository using the URL shown by GitHub's **Code** button, then open its root directory in a terminal.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For live experiments, create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_api_key_here
```

The model wrapper loads this file. `.env` is ignored by Git and must not be committed. Running experiments makes live model API calls; reading the saved JSONL files does not.

## Dataset setup

Download the dataset from the [Spider project](https://yale-lily.github.io/spider) and place it under `data/spider/`:

```text
data/spider/
    database/
        <db_id>/
            <db_id>.sqlite
    train_spider.json
    dev.json
    tables.json
```

The included grounding files, FAQ files, and clarification sets cover:

- `concert_singer`
- `museum_visit`
- `student_transcripts_tracking`

## Running experiments

The implemented entry points are Python runner classes. There is no `scripts/run_spider_eval.py` entry point in this version.

Run the following examples from the repository root, either in an interactive Python session or in a Python file saved there, after completing the data setup. These examples use the implemented API; they are not a claim that a fresh installation has been verified.

### Integrated agent on Spider

```python
from runner.conversational_execution_feedback_runner import (
    ConversationalExecutionFeedbackRunner,
)

runner = ConversationalExecutionFeedbackRunner(
    spider_path="data/spider",
    outputs_dir="outputs/conversational_execution_feedback",
)

result = runner.run(
    run_id="example_full_concert_singer",
    db_id="concert_singer",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    semantic_enrichment=True,
    use_faq=True,
    revision_policy="on_failure_or_empty",
    include_spider=True,
)

print(result.metrics)
print(result.log_path)
```

### Clarification holdout evaluation

Using the same `runner`:

```python
result = runner.run(
    run_id="example_clarification_concert_singer",
    db_id="concert_singer",
    model_name="gpt-4.1-mini",
    temperature=0.0,
    semantic_enrichment=True,
    use_faq=True,
    revision_policy="on_failure_or_empty",
    include_spider=False,
    evaluation_set_path=(
        "evaluation_sets/clarification_concert_singer_holdout.json"
    ),
)

print(result.metrics)
```

For another database, change both `db_id` and the matching holdout filename. Each example starts a separate conversation; a supplied clarification is processed within that conversation. Holdout user replies are scripted evaluation inputs, not responses collected from a live user study.

Use a distinct `run_id` for a new experiment. The integrated runner rejects an existing output filename unless `overwrite=True` is explicitly supplied.

Other implemented entry points include `ExperimentRunner.run_baseline_experiment`, `ExperimentRunner.run_execution_feedback_experiment`, and `ClarificationExperimentRunner`. The standalone clarification runner is specific to `concert_singer`; supply its existing holdout path explicitly because its default filename is not present in this version.

## Metrics and saved experiments

The project reuses the upstream evaluation implementations for execution correctness, exact match, component matching, Relational Accuracy, Informational Relational Accuracy, and Symbiosis. These metric names refer to this framework's implementations.

The integrated runner additionally reports action accuracy, ambiguity recall, follow-up correctness, end-to-end success, query changes, helpful/harmful revisions, and resource usage. Interpret subset metrics only when the corresponding subset contains examples.

The twelve logs under `outputs/conversational_execution_feedback/` comprise three database files for each of these groups:

- `report_a0_no_additions`: internal baseline using the new pipeline.
- `report_full_gpt41mini_t00`: full configuration with GPT-4.1-mini.
- `report_full_gpt54mini_t00`: full configuration with GPT-5.4-mini.
- `report_ambiguity_only_gpt41mini_t00`: ambiguity evaluation with GPT-4.1-mini.

A0 is an internal comparison configuration, not an exact reproduction of the upstream baseline. The saved experiments do not include separate FAQ-only and feedback-only arms, so they do not isolate each component's causal contribution. Cross-model comparisons also include the effect of changing the model.

Saved logs document the completed runs. New API runs may produce different outputs, token counts, and latencies. The repository does not currently include an executable notebook reproducing all thesis tables.

## Attribution and citation

When citing this work, distinguish the original framework from the thesis extensions:

- **Original framework:** Michele Paolicelli and collaborators, *Semantic Grounding over Brute-Force Inference: A Symbiotic Evaluation of Text-to-SQL Systems*. See the [upstream repository](https://github.com/michelepao1993-dev/semantic-grounding-text2sql).
- **Thesis extensions:** Mirco Catalano, conversational clarification, execution feedback, FAQ integration, and the associated experiments in this repository.

The upstream README supplies the following bibliographic entry, with author names formatted for BibTeX:

```bibtex
@inproceedings{paolicelli2026semantic,
  title = {Semantic Grounding over Brute-Force Inference: A Symbiotic Evaluation of Text-to-SQL Systems},
  author = {Paolicelli, Michele and Musto, Cataldo and Semeraro, Giovanni and Crnjar, Alessandro and Landi, David and Sacc{\`a}, Claudio},
  booktitle = {Proceedings of AIxIA 2026},
  year = {2026}
}
```

The thesis extensions are available at https://github.com/catalanomircosav/semantic-grounding-text2sql. When citing this repository, include the commit or release tag used for the experiments.

## License

This project is released under the MIT License; see [LICENSE.txt](LICENSE.txt). The upstream copyright and license notice are preserved.