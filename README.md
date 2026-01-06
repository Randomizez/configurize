# Configurize

A Python library for hierarchical configuration management with inheritance, cross-references, and diffing support.

## Installation

```bash
pip install configurize
```

for development
```bash
pip install .
```

## Features

- **Hierarchical configs**: Nest configs within configs with automatic parent-child relationships
- **Inheritance**: Extend configs like regular Python classes
- **Cross-references**: Use `Ref()` to reference values from other parts of the config tree
- **Diffing**: Compare two configs and see what changed
- **Merge**: Apply updates from dicts or other configs
- **Validation**: Define `critical_keys` for consistency checks

## Quick Start

```python
from configurize import Config, Ref

class ModelConfig(Config):
    in_channels = 32
    out_channels = 64

class TrainerConfig(Config):
    train_iters = 100
    batch_size = 16

class Exp(Config):
    model = ModelConfig
    trainer = TrainerConfig

# Create and use
exp = Exp()
print(exp.model.in_channels)  # 32
print(exp.trainer.train_iters)  # 100
```

## Cross-References with Ref

Use `Ref()` to reference values from other parts of the config tree:

```python
class ModelConfig(Config):
    hidden_dim = Ref('..hidden_dim')  # Reference parent's hidden_dim

class Exp(Config):
    hidden_dim = 256
    model = ModelConfig

exp = Exp()
print(exp.model.hidden_dim)  # 256 (resolved from parent)
```

Reference syntax:
- `.attr` - self.attr
- `..attr` - parent.attr
- `...sub.attr` - grandparent.sub.attr

## Merging and Overrides

```python
exp = Exp()

# Merge from dict (supports dot notation)
exp.merge({'model.in_channels': 64, 'trainer.train_iters': 200})

# Temporary modifications
with exp.trainer.modify(batch_size=32):
    print(exp.trainer.batch_size)  # 32
print(exp.trainer.batch_size)  # 16 (restored)
```

## Diffing Configs

```python
exp1 = Exp()
exp2 = Exp()
exp2.model.in_channels = 128

diff = exp1.diff(exp2)
print(diff)  # Shows differences with colors
```

## CLI Tool

After installation, use `cfshow` to inspect and compare config files:

```bash
# Show a config
cfshow examples/train_example.py

# Compare two configs
cfshow base_exp.py new_exp.py

# Inspect a sub-config
cfshow my_exp.py --key=model
```

## Example

See [examples/train_example.py](examples/train_example.py) for a complete example showing:
- Nested config structure (logger, model, trainer)
- Builder pattern with `build_*` methods
- Running an experiment from config
