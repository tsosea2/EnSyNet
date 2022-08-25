# EnSyNet

Anonymized using https://github.com/Smile-SA/anonymization

If you use this dataset, please cite our paper:


```bibtex
@InProceedings{sosea-caragea:2022:LREC,
  author    = {Sosea, Tiberiu  and  Caragea, Cornelia},
  title     = {EnsyNet: A Dataset for Encouragement and Sympathy Detection},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {June},
  year           = {2022},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {5444--5449},
  abstract  = {More and more people turn to Online Health Communities to seek social support during their illnesses. By interacting with peers with similar medical conditions, users feel emotionally and socially supported, which in turn leads to better adherence to therapy. Current studies in Online Health Communities focus only on the presence or absence of emotional support, while the available datasets are scarce or limited in terms of size. To enable development on emotional support detection, we introduce EnsyNet, a dataset of 6,500 sentences annotated with two types of support: encouragement and sympathy. We train BERT-based classifiers on this dataset, and apply our best BERT model in two large scale experiments. The results of these experiments show that receiving encouragements or sympathy improves users' emotional state, while the lack of emotional support negatively impacts patients' emotional state.},
  url       = {https://aclanthology.org/2022.lrec-1.583}
}


```

To reproduce paper results run:

```
sh run_ensynet_experiments.sh
```
