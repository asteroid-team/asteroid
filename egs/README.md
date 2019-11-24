### What is a recipe ? 
A recipe is a set of scripts that demonstrate how to use 
`asteroid` to build a source separation system.
Each directory corresponds to a dataset and each subdirectory 
corresponds to a system build on this dataset. 

### How is it organized ? 
The wham/ConvTasNet recipe contains the following elements 

```
├── conf.yml
├── data
├── eval.py
├── exp
├── local
│   ├── convert_sphere2wav.sh
│   ├── prepare_data.sh
│   └── preprocess_wham.py
├── logs
├── run.sh
├── train.py
└── utils
    ├── parse_options.sh
    └── prepare_python_env.sh

```

- Structure of the data folder? Dataset dependent probably? 
Or make something like wav.scp? 

### CLI & stage 
Coming

### How can I modify it? 
Good question. Info is coming.

