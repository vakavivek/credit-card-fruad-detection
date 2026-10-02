import optuna
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import os
import joblib

class FraudModel:

    def __init__(self):

        self.model = None
        self.best_params = None
        self.scale_pos_weight = None

    ###############################################################
    # Calculate Scale Pos Weight
    ###############################################################

    @staticmethod
    def calculate_scale_pos_weight(y_train):

        negative = (y_train == 0).sum()
        positive = (y_train == 1).sum()
        return negative / positive

    ###############################################################
    # Train Baseline Model
    ###############################################################

    def train_baseline(self,X_train,y_train):

        self.scale_pos_weight = self.calculate_scale_pos_weight(
            y_train
        )

        model = XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1
        )

        model.fit(X_train,y_train)

        self.model = model

        return model

    ###############################################################
    # Optuna Objective
    ###############################################################

    def objective(self,trial,X_train,y_train,X_valid,y_valid):

        params = {

            "n_estimators": trial.suggest_int("n_estimators",100,250),

            "max_depth": trial.suggest_int("max_depth",3,8),

            "learning_rate": trial.suggest_float("learning_rate",0.01,0.2,log=True),

            "subsample": trial.suggest_float("subsample",0.7,1.0),

            "colsample_bytree": trial.suggest_float("colsample_bytree",0.7,1.0),

            "min_child_weight": trial.suggest_int("min_child_weight",1,8),

            "gamma": trial.suggest_float("gamma",0,3),

            "reg_alpha": trial.suggest_float("reg_alpha",0, 3),

            "reg_lambda": trial.suggest_float( "reg_lambda",1,5),

            "scale_pos_weight": self.scale_pos_weight,

            "random_state":42,

            "eval_metric": "logloss",

            "tree_method": "hist",

            "n_jobs": -1

        }

        model = XGBClassifier(**params)

        model.fit(X_train,y_train, eval_set=[
                (
                    X_valid,
                    y_valid
                )
            ],

            verbose=False

        )

        pred_prob = model.predict_proba(
            X_valid
        )[:, 1]

        score = average_precision_score(

            y_valid,

            pred_prob

        )

        return score

    ###############################################################
    # Hyperparameter Tuning
    ###############################################################

    def tune_hyperparameters(self,X_train,y_train,X_valid,y_valid,n_trials=20):

        study = optuna.create_study(direction="maximize")

        study.optimize(

            lambda trial:

            self.objective(trial,X_train,y_train,X_valid,y_valid),

            n_trials=n_trials,

            show_progress_bar=True

        )

        self.best_params = study.best_params

        return study.best_params

    ###############################################################
    # Train Final Model
    ###############################################################

    def train_final_model(self,X_train,y_train):

        model = XGBClassifier(

            **self.best_params,

            scale_pos_weight=self.scale_pos_weight,

            random_state=42,

            eval_metric="logloss",

            tree_method="hist",

            n_jobs=-1

        )

        model.fit(X_train,y_train)

        self.model = model

        return model

    ###############################################################
    # Predict Probability
    ###############################################################

    def predict_proba(self,X):

        return self.model.predict_proba(X)[:, 1]

    ###############################################################
    # Predict
    ###############################################################

    def predict(self,X):

        return self.model.predict(X)
    
    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        model_path= os.path.join(path, "fraud_model.pkl")
        joblib.dump(self.model,model_path)

        print(f"Model Saved Successfully at {model_path}")

    def load(self, path="models/fraud_model.pkl"):
        self.model= joblib.load(path)
        print(f"Model loaded successfully from {path}")
        return self.model