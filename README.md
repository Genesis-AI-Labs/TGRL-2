# TGRL-2
### Analysis of Experiment 1 (CartPole)

The first graph shows three approaches:
1. Baseline PPO (Blue)
2. GNN with Random Policy Data (Orange)
3. GNN with Partial Policy Data (Green)

Key observations:
- All three approaches eventually reach similar performance (~500 reward)
- The baseline PPO actually learns slightly faster initially
- Using GNN embeddings (both random and partial policy) doesn't provide significant advantages
- GNN with partial policy data shows slightly slower initial learning

Main issues:
1. CartPole is too simple - it's solved quickly by basic PPO, leaving little room for improvement
2. The GNN embeddings weren't trained with any meaningful objective
3. The discretization scheme might be too crude for the continuous state space

### Analysis of Experiment 2 (Acrobot)

The second graph shows:
1. Baseline PPO (Blue)
2. Value-Trained GNN + PPO (Orange)

Key observations:
- Initial learning is slower with the GNN approach
- Eventually both converge to similar performance
- More variance in the GNN approach's learning curve
- The final performance is about the same (-100 reward)

Main issues:
1. Value prediction training might not be capturing the most useful information
2. Only 10 epochs of GNN training might be insufficient
3. The embeddings aren't being updated during RL training

### Suggested Next Steps

The key is to:
1. Move to a more challenging environment where graph structure could actually help
2. Make the graph representation dynamic and updateable
3. Use more sophisticated state abstraction methods
4. Train the GNN with multiple relevant objectives
5. Better incorporate edge features (actions and rewards)
6. Consider exploration-specific metrics in the graph structure
