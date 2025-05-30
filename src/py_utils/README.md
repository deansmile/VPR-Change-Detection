# python_utils

a few small tools that I greatly rely on in the following area:
- robotics
- computer vision
- machine learning

### file structure

```
├── examples
│   └── demo_<tools>.ipynb
│
├── src
│   └── <tools>.py
│
├── test
│   └── test_<tools>>.py
│
├── __init__.py
├── LICENSE
├── README.md
└── requirements.txt
```

### Unittest

```bash
$ cd <directory of this repository>
$ python -m unittest
```

### Maintenance

```bash
$ cd <directory of this repository>
$ python -m black . --line-length=79

# sort all imports in alphabetical order
$ python -m isort .
```

### TODO:
- [ ] add github workflow
- [ ] Data Repo: functionality of attributes serialization
- [ ] Data Repo: implement tools to manage data repo tree
- [x] PCD: Implement pcd writer
