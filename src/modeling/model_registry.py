from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models(random_state: int = 42):
    return {
        # --- Linear Models --
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced"
    ),
    "ExtraTrees": ExtraTreesClassifier(
         n_estimators=300,
        random_state=random_state,
        class_weight="balanced"
    ),
    "xgBoost": XGBClassifier(
        use_label_encoder = False,
        eval_metric = "logloss",
        random_state = random_state
    ),
    "LightGBM": LGBMClassifier(random_state=random_state),
    "AdaBoost": AdaBoostClassifier(random_state=random_state),
    "GaussianNB": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbours = 5),
    "SVM": SVC(probability = True, class_weight = "balanced", random_state = random_state)
    }