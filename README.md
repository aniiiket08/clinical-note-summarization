# NLP Clinical Note Summarization (BART-large-cnn)

Paper title: NLP Based Patient Record Summarization for Clinical Decision
Authors: Aniket Patil, Sujal Revankar, Viraj Patil, Srushti Bandi and Savita Bagewadi (KLE Technological University, Dr. M. S. Sheshgiri Campus, Belagavi)

Goal: turn unstructured clinical notes into structured JSON-style summaries. Three systems were compared: TF-IDF extractive, zero-shot BART-large-cnn and fine-tuned BART-large-cnn.

## Folder guide
| Path | What it is |
|---|---|
| notebook/clinical_summarization_bart_finetune.ipynb | Original Kaggle notebook with all saved outputs |
| src/finetune_and_evaluate_bart_clinical_summarizer.py | All code cells as one runnable script |
| requirements.txt | Packages needed to rerun |
| results/evaluation_results_50_test_notes.csv | Final metric table |
| results/training_log.csv | Per-epoch train and validation loss |
| results/notebook_printed_output.txt | Cleaned printed output of the run |
| results/sample_predictions.txt | Full printed sample outputs and demo |
| figures/ | 4 notebook figures and 3 paper figures |
| paper/ | IEEE paper PDF and its extracted text |
| docs/Project_Recovery_Report_and_Resource_Inventory.docx | Tools and links inventory |

## Setup
- Dataset: AGBonnet/augmented-clinical-notes (30,000 rows). Input note and target summary. Sample of 5,000 (seed 42) split 3,500 / 750 / 750
- Model: facebook/bart-large-cnn (406,290,432 parameters). 512 input and 384 target tokens
- Training: lr 3e-5 cosine with 131 warmup steps and per-device batch 2 and gradient accumulation 4 and up to 5 epochs with early stopping patience 2 and label smoothing 0.1 and weight decay 0.01 and fp16 on Tesla T4
- Runtime 5,777.6 s and final train loss 1.6406
- Inference: 4 beams and length penalty 1.2 and no_repeat_ngram_size 3 and max 384 and min 80 tokens

## Results (first 50 test notes)
| Model | R-1 | R-2 | R-L | BERT-F1 | METEOR | NER | MEDCON | Factual |
|---|---|---|---|---|---|---|---|---|
| TF-IDF | 0.2918 | 0.1751 | 0.1936 | 0.7291 | 0.0684 | 0.2459 | 0.5143 | 1.0000 |
| BART zero-shot | 0.1926 | 0.1169 | 0.1384 | 0.7128 | 0.0395 | 0.1814 | 0.4268 | 0.9995 |
| Fine-tuned BART | 0.4922 | 0.3381 | 0.3470 | 0.9187 | 0.2546 | 0.5818 | 0.6531 | 0.5802 |

## How metrics were really computed
- BERTScore with distilbert-base-uncased
- NER preservation used a regex keyword fallback because scispaCy en_core_sci_sm failed to load
- MEDCON is a custom lexicon of about 150 terms with substring matching
- Factual consistency is token overlap with the source note (a proxy)

## Known issues to fix in the paper
1. Paper medians (220 and 180 words) differ from measured 316 and 446
2. Scores use 50 of 750 test notes and not the full test set
3. Factual consistency (0.58 for fine-tuned) is missing from the paper
4. NER and MEDCON are keyword and lexicon proxies
5. Fine-tuned outputs often contain invalid JSON
6. TF-IDF is a separate baseline and is not fed into BART so avoid calling the system hybrid
7. Effective batch size was probably 16 (two GPUs) and not 8 and the cosine schedule was planned for 2,185 steps but about 1,095 ran. This is inferred and needs checking on Kaggle
8. peft was installed but never used

## Not recovered
Fine-tuned model weights (bart_large_medical_summarizer_final/) and full per-sample predictions. Check the Kaggle notebook Output tab or version history.
