# SequenceTagging

Assigning part-of-speech tags, phrase chunk labels and named-entity labels to the words in sentences. Two models are used, namely Logistic Regression and Conditional Random Fields for each of the 3 tasks.

Results:


| Tagger  | Accuracy | Precision | Recall | F1-Measure |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| TaggerLR POS | 0.770 | 0.770 | 0.770 | 0.770 |
| TaggerLR Chunk | 0.791 | 0.791 | 0.791 | 0.791 |
| TaggerLR NER | 0.914 | 0.914 | 0.914 | 0.914 |
| TaggerCRF POS | 0.885 | 0.885 | 0.885 | 0.885 |
| TaggerCRF Chunk | 0.917 | 0.917 | 0.917 | 0.917 |
| TaggerCRF NER | 0.943 | 0.943 | 0.943 | 0.943 |
