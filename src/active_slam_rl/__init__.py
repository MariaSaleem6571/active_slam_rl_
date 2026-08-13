"""
active_slam_rl
==============

Reinforcement-learning-driven active SLAM for confined underwater environments.

This package implements the architecture from the thesis proposal:
registration (FS2D) -> volumetric mapping -> change detection -> loop closure
-> state encoding -> PPO policy -> reward, closed by 7 feedback loops.

See docs/ARCHITECTURE.md for the full explanation of every module and how the
math in the proposal maps onto the code.
"""

__version__ = "0.1.0"
