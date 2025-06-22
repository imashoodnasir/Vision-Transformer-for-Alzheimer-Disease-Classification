from bayes_opt import BayesianOptimization
from train import train_model
import warnings
warnings.filterwarnings("ignore")

# Objective function to optimize
def objective_function(lr, depth, mlp_ratio):
    lr = float(lr)
    depth = int(depth)
    mlp_ratio = float(mlp_ratio)

    try:
        acc = train_model(
            epochs=10,  # For search, use shorter training
            lr=lr,
            batch_size=32,
            device='cuda',
            depth=depth,
            mlp_ratio=mlp_ratio,
            return_val_acc=True  # Modify train_model to return val accuracy
        )
        return acc
    except Exception as e:
        print("Error during optimization:", e)
        return 0.0

# Define the search space
pbounds = {
    'lr': (1e-5, 5e-4),
    'depth': (4, 12),
    'mlp_ratio': (2.0, 6.0)
}

if __name__ == "__main__":
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    print("Starting Bayesian Hyperparameter Optimization...\n")
    optimizer.maximize(init_points=3, n_iter=7)

    print("\n✅ Best configuration found:")
    print(optimizer.max)
