import os

import pandas as pd

DATASETS_CLASSIFICATION = {
    "Heart": ("heart", "imodels"),
    "Breast cancer": ("breast_cancer", "imodels"),
    "Haberman": ("haberman", "imodels"),
    "Ionosphere": ("ionosphere", "pmlb"),
    "Diabetes": ("diabetes", "pmlb"),
    "German credit": ("german", "pmlb"),
    "Juvenile": ("juvenile_clean", "imodels"),
    "Recidivism": ("compas_two_year_clean", "imodels"),
}

DATASETS_REGRESSION = {
    "Friedman1": ("friedman1", "synthetic"),
    "Friedman3": ("friedman3", "synthetic"),
    "Diabetes": ("diabetes", "sklearn"),
    "Geographical music": (),
    "Red wine": (),
    "Abalone": ("183", "openml"),
    "Satellite image": ("294_satellite_image", "pmlb"),
    "CA housing": ("california_housing", "sklearn"),
}


HEART_COLS_TRANSLATE = {
    0: "age",
    1: "sex",
    2: "chest_pain_type",
    3: "resting_blood_pressure",
    4: "serum_cholesterol",
    5: "fasting_blood_sugar",
    6: "resting_electrocardiographic_results",
    7: "maximum_heart_rate_achieved",
    8: "exercise_induced_angina",
    9: "oldpeak",
    10: "slope_of_the_peak",
    11: "number_of_major_vessels",
    12: "thal",
    13: "heart_disease",
}
HEART_COLS_REAL = [
    "age",
    "resting_blood_pressure",
    "serum_cholesterol",
    "maximum_heart_rate_achieved",
    "oldpeak",
    "number_of_major_vessels",
]
HEART_COLS_BINARY = [
    "sex",
    "fasting_blood_sugar",
    "exercise_induced_angina",
    "heart_disease",
]
HEART_COLS_ORDERED = ["slope_of_the_peak"]
HEART_COLS_NOMINAL = ["chest_pain_type", "resting_electrocardiographic_results", "thal"]

BREAST_AGE_TRANSLATE = {
    "10-19": 0,
    "20-29": 1,
    "30-39": 2,
    "40-49": 3,
    "50-59": 4,
    "60-69": 5,
    "70-79": 6,
    "80-89": 7,
    "90-99": 8,
}
BREAST_TUMOR_TRANSLATE = {
    "0-4": 0,
    "5-9": 1,
    "10-14": 2,
    "15-19": 3,
    "20-24": 4,
    "25-29": 5,
    "30-34": 6,
    "35-39": 7,
    "40-44": 8,
    "45-49": 9,
    "50-54": 10,
    "55-59": 11,
}
BREAST_INV_TRANSLATE = {
    "0-2": 0,
    "3-5": 1,
    "6-8": 2,
    "9-11": 3,
    "12-14": 4,
    "15-17": 5,
    "18-20": 6,
    "21-23": 7,
    "24-26": 8,
    "27-29": 9,
    "30-32": 10,
    "33-35": 11,
    "36-39": 12,
}

HABERMAN_COLS_TRANSLATE = {
    0: "age",
    1: "year_of_operation",
    2: "positive_axillary_nodes_detected",
    3: "survival",
}

IONOSPHERE_COLS_TRANSLATE = {
    **{i: f"attr_{i}" for i in range(34)},
    **{34: "ionosphere"},
}

GERMAN_COLS_TRANSLATE = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]
GERMAN_COLS_TRANSLATE = {i: v for i, v in enumerate(GERMAN_COLS_TRANSLATE)}


class DatasetClassification:

    def get_heart() -> pd.DataFrame:
        heart = pd.DataFrame(columns=list(range(len(HEART_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/heart.dat"), "r") as f:
            for line in f:
                params = line.strip().split(" ")
                heart = pd.concat([heart, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        heart = heart.rename(HEART_COLS_TRANSLATE, axis=1)
        for col in HEART_COLS_REAL:
            heart[col] = heart[col].astype(float)
        for col in HEART_COLS_BINARY+HEART_COLS_NOMINAL:
            heart[col] = heart[col].astype(float).astype(int)
        for col in HEART_COLS_ORDERED:
            heart[col] = heart[col].astype(float).astype(int).astype(str)

        return heart


    def get_breast_cancer() -> pd.DataFrame:
        breast_cancer = pd.DataFrame(columns=list(range(len(BREAST_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/dataset_13_breast-cancer.arff"), "r") as f:
            header = list()
            for line in f:
                if line.startswith("@attribute"):
                    header.append(line.split("'")[1])
                if line.startswith("%") or line.startswith("@"):
                    continue
                params = line.strip().replace("'", "").split(",")
                breast_cancer = pd.concat([breast_cancer, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        breast_cancer = breast_cancer.rename({i: v for i, v in enumerate(header)}, axis=1)
        breast_cancer = breast_cancer.replace("?", np.nan).dropna()
        breast_cancer["age"] = breast_cancer["age"].apply(lambda x: BREAST_AGE_TRANSLATE[x])
        breast_cancer["tumor-size"] = breast_cancer["tumor-size"].apply(lambda x: BREAST_TUMOR_TRANSLATE[x])
        breast_cancer["inv-nodes"] = breast_cancer["inv-nodes"].apply(lambda x: BREAST_INV_TRANSLATE[x])
        breast_cancer["node-caps"] = (breast_cancer["node-caps"] == "yes")*1
        breast_cancer["breast"] = (breast_cancer["breast"] == "right")*1
        breast_cancer["irradiat"] = (breast_cancer["irradiat"] == "yes")*1
        breast_cancer["Class"] = (breast_cancer["Class"] == "recurrence-events")*1
        breast_cancer = pd.get_dummies(breast_cancer, columns=["deg-malig", "menopause", "breast-quad"])

        return breast_cancer
    

    def get_haberman() -> pd.DataFrame:
        haberman = pd.DataFrame(columns=list(range(len(HABERMAN_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/haberman.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                haberman = pd.concat([haberman, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        haberman = haberman.rename(HABERMAN_COLS_TRANSLATE, axis=1)
        haberman = haberman.astype(int)
        
        return haberman


    def get_ionosphere() -> pd.DataFrame:
        ionosphere = pd.DataFrame(columns=list(range(len(IONOSPHERE_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/ionosphere.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                ionosphere = pd.concat([ionosphere, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        ionosphere = ionosphere.rename(IONOSPHERE_COLS_TRANSLATE, axis=1)
        ionosphere["ionosphere"] = ionosphere["ionosphere"] == "g"
        
        return ionosphere
    

    def get_diabetes() -> pd.DataFrame:
        diabetes = pd.read_csv(os.path.abspath("../data/classification/diabetes.csv"))
        return diabetes


    def get_german_credit() -> pd.DataFrame:
        german_credit = pd.DataFrame(columns=list(range(len(GERMAN_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/SouthGermanCredit.asc"), "r") as f:
            _ = f.readline().strip().split(" ") #skip header, which is in German
            for line in f:
                params = line.strip().split(" ")
                german_credit = pd.concat([german_credit, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)
        german_credit = german_credit.rename(GERMAN_COLS_TRANSLATE, axis=1)
        german_credit = german_credit.astype(int)