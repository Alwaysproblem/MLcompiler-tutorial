1. install pre-commit

```bash
conda create -n mlserving python=3.8 pre-commit==2.20.* -c conda-forge -y
conda install yapf==0.32.0 pylint==2.15.5 -c conda-forge -y
```

2. `pre-commit install`
3. before `git commit` make sure to run `pre-commit run`
