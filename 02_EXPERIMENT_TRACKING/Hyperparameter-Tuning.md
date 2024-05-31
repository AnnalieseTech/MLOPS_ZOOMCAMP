# Adding Hyperparameter Tuning to Your Notebook with MLflow and Hyperopt

### Notes by [@AnnalieseTech](https://github.com/AnnalieseTech)

**Introduction**
- This tutorial covers how to add hyperparameter tuning to a notebook.
- It includes:
  - A complex example of hyperparameter tuning.
  - Exploring hyperparameter search results using the MLflow UI.
  - Determining the best model based on the results.
  - Demonstrating MLflow's autologging feature for efficient logging with minimal code.

**Setting Up Hyperparameter Tuning**
- Previous examples used a lasso model with hyperparameter alpha.
- Comparison of runs with different alpha values using MLflow UI.
- Limited insights from comparing just two runs.
- This example uses XGBoost for a more comprehensive hyperparameter tuning.

**Code Explanation**
- Import necessary libraries: XGBoost and Hyperopt.
- Hyperopt minimizes the objective function using Bayesian methods.
- The objective function:
  - Takes parameters and outputs a value to minimize.
  - Uses the `fmin` method to find the optimal hyperparameters.
  - Controls logic with the TPE algorithm.
  - Defines the search space using Hyperopt's `hp` library.
- Detailed explanation of the objective function, search space, and how Hyperopt uses them.

**Defining the Objective Function**
- Reads parameters and trains the XGBoost model on the training data.
- Uses the validation set to control the optimization.
- Stops optimization if no improvement after 50 iterations.
- Logs the root mean square error (RMSE) on the validation set.
- Returns the loss for Hyperopt to minimize.

**Defining the Search Space**
- Sets ranges for hyperparameters using methods like `quniform` and `loguniform`.
- Explanation of how `quniform` and `loguniform` work.

**Running the Optimization**
- Uses `fmin` to optimize the objective function.
- Checks if the directory and file paths are correct.
- Sets the maximum number of iterations to 50.
- Stores trial information in the `trials` object.

**Exploring Results in MLflow UI**
- Filters runs by the XGBoost model tag.
- Uses parallel coordinates plot to visualize hyperparameter values and their effects.
- Highlights correlations between hyperparameters and RMSE.
- Uses scatter plots and contour plots for further analysis.

**Selecting the Best Model**
- Sorts results by RMSE to find the best-performing model.
- Suggests considering training time, model size, and complexity.
- Chooses the model with the best trade-off between performance and complexity.

**Demonstrating Autologging**
- Introduces MLflow's autologging feature.
- Shows how to enable autologging with one line of code.
- Explains the additional information logged automatically, such as hyperparameters, metrics, and feature importance.

**Final Thoughts**
- Hyperparameter tuning and logging can significantly improve model performance.
- MLflow's autologging simplifies the logging process.
- Further exploration includes:
  - Logging models with MLflow manually.
  - Saving and retrieving models for predictions.
  - Controlling the model saving and retrieval process using the MLflow SDK.
