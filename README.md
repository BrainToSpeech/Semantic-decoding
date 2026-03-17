# Semantic-Decoding

Minimal SRM-based semantic decoding pipeline for podcast ECoG data.

This folder contains only the core code needed to:

1. Load podcast high-gamma neural data and text embeddings
2. Build subject-wise neural targets from word onsets 
3. Optionally fit SRM on selected subjects   
4. Train a text-to-neural decoder per subject
5. Compare performance with cosine similarity on the test split
 
## Run
 
```bash
python srm_cosine_pipeline.py --config config.example.json
```
 
## Input data format

The pipeline expects a pickle file with:

```python
{
  "significant_elecs": (n_channels,), 
  "word_data": {
    "subj": (n_channels,),
    "highgamma": (timepoints, n_channels),
    "onset": (n_words,),
    "word": (n_words,)  
  }
}
```
 
And a text embedding `.npy` file with shape:

```python 
(n_words, embedding_dim)
```

## Outputs
 
Results are saved per subject under the configured output directory:

- `test_metrics.json`
- `hyperparameters.json`

## Notes

- Text PCA is fit on the train split only.
- SRM standardization is computed per subject using train samples only.
- Final target z-scoring is also computed from the train split only.
- Test performance is reported with cosine similarity and MSE.
