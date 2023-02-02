import os

import pandas as pd
import sklearn


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
    "Geographical music": ("4544_GeographicalOriginalofMusic", "pmlb"),
    "Red wine": ("wine_quality_red", "pmlb"),
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

GERMAN_COLS_TRANSLATE = {
    0: "status",
    1: "duration",
    2: "credit_history",
    3: "purpose",
    4: "amount",
    5: "savings",
    6: "employment_duration",
    7: "installment_rate",
    8: "personal_status_sex",
    9: "other_debtors",
    10: "present_residence",
    11: "property",
    12: "age",
    13: "other_installment_plans",
    14: "housing",
    15: "number_credits",
    16: "job",
    17: "people_liable",
    18: "telephone",
    19: "foreign_worker",
    20: "credit_risk"
}
GERMAN_COLS_MINUS = [
    "status",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

ABALONE_COLS_TRANSLATE = {
    0: "Sex", 1: "Length", 2: "Diameter", 3: "Height", 4: "Whole_weight",
    5: "Shucked_weight", 6: "Viscera_weight", 7: "Shell_weight", 8: "Rings"
}
ABALONE_SEX_TRANSLATE = {
    "M": 2, "F": 0, "I": 1
}

HOUSING_COLS_TRANSLATE = {
    0: "longitude",
    1: "latitude",
    2: "housingMedianAge",
    3: "totalRooms",
    4: "totalBedrooms",
    5: "population",
    6: "households",
    7: "medianIncome",
    8: "medianHouseValue"
}


class LocalDatasets:

    def get_heart_c() -> pd.DataFrame:
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


    def get_breast_cancer_c() -> pd.DataFrame:
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
    

    def get_haberman_c() -> pd.DataFrame:
        haberman = pd.DataFrame(columns=list(range(len(HABERMAN_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/haberman.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                haberman = pd.concat([haberman, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        haberman = haberman.rename(HABERMAN_COLS_TRANSLATE, axis=1)
        haberman = haberman.astype(int)
        
        return haberman


    def get_ionosphere_c() -> pd.DataFrame:
        ionosphere = pd.DataFrame(columns=list(range(len(IONOSPHERE_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/ionosphere.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                ionosphere = pd.concat([ionosphere, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)

        ionosphere = ionosphere.rename(IONOSPHERE_COLS_TRANSLATE, axis=1)
        ionosphere["ionosphere"] = ionosphere["ionosphere"] == "g"
        
        return ionosphere
    

    def get_diabetes_c() -> pd.DataFrame:
        diabetes = pd.read_csv(os.path.abspath("../data/classification/diabetes.csv"))
        return diabetes


    def get_german_credit_c() -> pd.DataFrame:
        german_credit = pd.DataFrame(columns=list(range(len(GERMAN_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/classification/SouthGermanCredit.asc"), "r") as f:
            _ = f.readline().strip().split(" ") #skip header, which is in German
            for line in f:
                params = line.strip().split(" ")
                german_credit = pd.concat([german_credit, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)
        german_credit = german_credit.rename(GERMAN_COLS_TRANSLATE, axis=1)
        german_credit = german_credit.astype(int)
        for col in GERMAN_COLS_MINUS:
            german_credit[col] = german_credit[col] - 1
        return german_credit


    def get_friedman_1_r() -> pd.DataFrame:
        X, y = sklearn.datasets.make_friedman1(200, 10)
        friedman1 = pd.DataFrame(X)
        friedman1["target"] = y
        return friedman1


    def get_friedman_3_r() -> pd.DataFrame:
        X, y = sklearn.datasets.make_friedman3(200)
        friedman3 = pd.DataFrame(X)
        friedman3["target"] = y
        return friedman3
    
    
    def get_diabetes_r() -> pd.DataFrame:
        diabetes_data = sklearn.datasets.load_diabetes(as_frame=True)
        diabetes = diabetes_data["data"]
        diabetes["diabetes"] = diabetes_data["target"]
        return diabetes
    
    
    def get_geographical_music_r() -> pd.DataFrame:
        geographical_music = pd.read_csv(os.path.abspath("../data/regression/geographical_music.tsv"), sep="\t")
        return geographical_music
    
    
    def get_red_wine_r() -> pd.DataFrame:
        red_wine = pd.read_csv(os.path.abspath("../data/regression/winequality-red.csv"), sep=";")
        return red_wine
    
    
    def get_abalone_r() -> pd.DataFrame:
        abalone = pd.DataFrame(columns=list(range(len(ABALONE_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/regression/abalone.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                abalone = pd.concat([abalone, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)
        abalone = abalone.rename(ABALONE_COLS_TRANSLATE, axis=1)
        abalone["Sex"] = abalone["Sex"].apply(lambda x: ABALONE_SEX_TRANSLATE[x])
        abalone = abalone.astype(float)
        return abalone
    
    
    def get_satellite_images_r() -> pd.DataFrame:
        satellite = pd.read_csv(os.path.abspath("../data/regression/satellite_image.tsv"), sep="\t")
        return satellite
    
    
    def get_ca_housing_r() -> pd.DataFrame:
        ca_housing = pd.DataFrame(columns=list(range(len(HOUSING_COLS_TRANSLATE))))
        with open(os.path.abspath("../data/regression/ca_housing.data"), "r") as f:
            for line in f:
                params = line.strip().split(",")
                ca_housing = pd.concat([ca_housing, pd.DataFrame.from_dict({i: [v] for i, v in enumerate(params)})], ignore_index=True)
        ca_housing = ca_housing.astype(float)
        ca_housing = ca_housing.rename(HOUSING_COLS_TRANSLATE, axis=1)
        return ca_housing