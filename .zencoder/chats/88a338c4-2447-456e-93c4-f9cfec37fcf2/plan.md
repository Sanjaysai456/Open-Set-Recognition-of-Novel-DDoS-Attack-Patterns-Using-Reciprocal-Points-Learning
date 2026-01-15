# Bug Fix Plan

This plan guides you through systematic bug resolution. Please update checkboxes as you complete each step.

## Phase 1: Investigation

### [x] Bug Reproduction

- Understand the reported issue and expected behavior
- Reproduce the bug in a controlled environment
- Document steps to reproduce consistently
- Identify affected components and versions

### [x] Root Cause Analysis

- Debugged the `ValueError`: The production model's `StandardScaler` expects 77 features, but the reconstruction logic in `MyModel.predict` was only providing 20 features (likely because `self.feature_columns` in the saved production model only contains the 20 selected features).
- Identified a batch processing bug in `MyModel.predict`: it was only capable of processing a single row (or incorrectly broadcasting a single row's data).
- Found a mismatch in `config/schema.yaml`: `numerical_columns` only listed 49 out of 77 available numerical features.

## Phase 2: Resolution

### [x] Fix Implementation

- **Fix `MyModel.predict`**: Updated to handle batch predictions using `dataframe.index` and `pd.DataFrame(0.0, ...)` for robust feature reconstruction.
- **Update `config/schema.yaml`**: Included all 77 numerical features in `numerical_columns` to ensure the scaler is fitted on the full set of features and matches the data.
- **Improved `ModelEvaluation`**: Wrapped production model evaluation in a `try-except` block to gracefully handle incompatible models from S3, allowing the pipeline to replace them with corrected models.

### [x] Impact Assessment

- **Affected Areas**: `MyModel` prediction logic, `schema.yaml` configuration, and `ModelEvaluation` component.
- **Side Effects**: `DataValidation` now strictly requires all 77 numerical columns. This is the desired behavior for consistency with the model's expected input.
- **Backward Compatibility**: Existing models in S3 that were trained on fewer features will trigger a logged error during evaluation but will not crash the pipeline, thanks to the new `try-except` block in `ModelEvaluation`.

## Phase 3: Verification

### [ ] Testing & Verification

- Verify the bug is fixed with the original reproduction steps
- Write regression tests to prevent recurrence
- Test related functionality for side effects
- Perform integration testing if applicable

### [ ] Documentation & Cleanup

- Update relevant documentation
- Add comments explaining the fix
- Clean up any debug code
- Prepare clear commit message

## Notes

- Update this plan as you discover more about the issue
- Check off completed items using [x]
- Add new steps if the bug requires additional investigation
