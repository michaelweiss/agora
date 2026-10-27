# agora

A lightweight agent framework with a blackboard architecture.

## Install

To install locally call the following from this directory:

```sh
python -m pip install -e .
```

## Use

Each agent must extend the `Agent` class. 

A `Controller` checks which agent should execute next.

Agents communicate via a `Blackboard`.

The following is a minimal example:

```python

```