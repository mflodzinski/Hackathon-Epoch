import numpy as np
import pandas as pd

def score(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    """
    Computes the competition metric for predicting forest fire sizes.

    This metric is defined as the mean of:
        min(abs(log(predicted / true)), 10)
    over all valid entries, where invalid or missing predictions get a penalty of 10.

    Parameters
    ----------
    solution : pd.DataFrame
        A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"].
    submission : pd.DataFrame
        A DataFrame with columns ["ID", "STATE", "month", "total_fire_size"],
        where "ID" is typically formatted as "STATE_month".

    Returns
    -------
    float
        The mean of the clamped log errors over all entries.
    """
    # Merge submission with ground truth on (STATE, month)
    merged = solution.merge(
        submission, 
        on=["STATE", "month"], 
        how="left", 
        suffixes=("_true", "_pred")
    )

    # Identify missing or zero/negative predictions
    missing_pred_mask = merged["total_fire_size_pred"].isna()
    zero_pred_mask = merged["total_fire_size_pred"] <= 0

    # Default log errors to 10.0 (maximum penalty)
    log_errors = np.full(len(merged), 10.0)

    # Compute log errors for valid predictions only
    valid_pred_mask = ~missing_pred_mask & ~zero_pred_mask
    log_errors[valid_pred_mask] = np.abs(
        np.log(
            merged.loc[valid_pred_mask, "total_fire_size_pred"] /
            merged.loc[valid_pred_mask, "total_fire_size_true"]
        )
    )

    # Clamp the errors at 10
    final_scores = np.minimum(log_errors, 10)

    # Return the mean score; if no entries, return 10
    return np.mean(final_scores) if len(final_scores) > 0 else 10.0


def create_submission(
    submission_df: pd.DataFrame, 
    output_file: str = "submission.csv"
) -> None:
    """
    Creates a submission file from a DataFrame by adding an 'ID' column
    and saving it to CSV.

    Parameters
    ----------
    submission_df : pd.DataFrame
        A DataFrame with at least one column of predictions (e.g., 'TARGET').
        This function will add an 'ID' column to it.
    output_file : str, optional
        The path to the CSV file to create (default is "submission.csv").

    Returns
    -------
    None
    """
    # Create an ID column by assigning a unique integer to each row
    submission_df["ID"] = range(len(submission_df))

    # Reorder the DataFrame so that "ID" is the first column
    cols = ["ID"] + [col for col in submission_df.columns if col != "ID"]
    submission_df = submission_df[cols]

    # Optional: preview the DataFrame
    print(submission_df.head())

    # Save the DataFrame as a CSV file
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved as '{output_file}'.")
