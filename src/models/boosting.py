import xgboost as xgb

class XGB:
    def __init__(self, max_depth = 3, learning_rate = 0.05, n_estimators = 2000, gamma = 0, min_child_weight = 1, subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 0.004, tree_method = "gpu_hist", boosting_type = "gbdt") -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.boosting_type = boosting_type
    def xgb_model(self):
        model = xgb.XGBRegressor(
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            n_estimators = self.n_estimators,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            reg_alpha = self.reg_alpha,
            tree_method = self.tree_method,
            boosting_type = self.boosting_type
            )
        return model
