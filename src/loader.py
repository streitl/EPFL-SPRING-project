import pandas as pd

from sklearn.model_selection import train_test_split

from .preprocessing import clean, bin_features, one_hot_encode


def load_dataset(name):
    """
    Loads the dataset with the given name, and returns two DataFrames with the features and label.
    
    Arguments:
    - name: string name of the dataset to load, without the '.csv' extension
    
    Returns:
    - X: DataFrame with the features
    - y: DataFrame with the label, values are only 0 and 1
    """
    # Dict of dataset name to label target, used to convert 'label' column to 0/1 vector
    targets = {
        'adult':           " >50K",
        'annealing':       "3",
        'audiology-std':   1,
        'bank':            "yes",
        'bankruptcy':      "B",
        'car':             "unacc",
        'chess-krvk':      "draw",
        'chess-krvkp':     "won",
        'congress-voting': "democrat",
        'contrac':         1,
        'credit-approval': "+",
        'ctg':             1,
        'cylinder-bands':  "noband",
        'dermatology':     1,
        'german_credit':   1,
        'heart-cleveland': range(1, 5+1),
        'ilpd':            1,
        'mammo':           1,
        'mushroom':        "p",
        'wine':            2,
        'wine_qual':       range(6, 9+1)
    }
    # Dict of dataset name to column names, used to build the pandas DataFrame
    columns = {
        'adult': 
                ["age", "workclass", "fnlwgt", "education", "education_no",
                "marital_status", "occupation", "relationship", "race", "sex",
                "capital_gain", "capital_loss", "hours_per_week",
                "native_country", "label"],
        'annealing': 
                ["family", "product", "steel", "carbon", "hardness",
                "temper_rolling", "condition", "formability", "strength", "non",
                "surface_finish", "surface_quality", "enamelability", "bc", "bf",
                "bt", "bw", "bl", "m", "chrom", "phos", "cbond", "marvi",
                "exptl", "ferro", "corr", "blue", "lustre", "jurofm", "s", "p",
                "shape", "thick", "width", "len", "oil", "bore", "packing",
                "label"],
        'audiology-std':
                ["label", "age_gt_60", "air", "airBoneGap", "ar_c", "ar_u",
                "boneAbnormal", "history_dizziness", "history_fluctuating",
                "history_nausea", "history_noise", "history_recruitment",
                "history_ringing", "history_roaring", "history_vomiting",
                "late_wave_poor", "m_m_gt_2k", "m_m_sn", "m_m_sn_gt_1k",
                "m_m_sn_gt_2k", "m_sn_gt_1k", "m_sn_gt_2k", "m_sn_gt_3k",
                "m_sn_gt_4k", "m_sn_lt_1k", "middle_wave_poor", "mod_sn_gt_2k",
                "mod_sn_gt_3k", "mod_sn_gt_4k", "mod_sn_gt_500", "notch_4k",
                "notch_at_4k", "o_ar_c", "o_ar_u", "s_sn_gt_1k", "s_sn_gt_2k",
                "s_sn_gt_4k", "speech", "static_normal", "tymp", "wave_V_delayed",
                "waveform_ItoV_prolonged"],
        'bank': 
                ["age", "job", "marital", "education", "default", "housing",
                "loan", "contact", "month", "day_of_week", "duration",
                "campaign", "pdays", "previous", "poutcome", "emp.var.rate",
                "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
                "label"],
        'bankruptcy': 
                ["industrial_risk", "management_risk", "financial_flex",
                "credibility", "competitiveness", "operating_risk", "label"],
        'car': 
                ["buying", "maint", "doors", "persons", "lug_boot", "safety",
                "label"],
        'chess-krvk':
                ["White_King_file", "White_King_rank", "White_Rook_file",
                "White_Rook_rank", "Black_King_file", "Black_King_rank", "label"],
        'chess-krvkp': 
                ["bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr",
                "bkxwp", "blxwp", "bxqsq", "cntxt", "dsopp", "dwipd", "hdchk",
                "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx",
                "rkxwp", "rxmsq", "simpl", "skach", "skewr", "skrxp", "spcop",
                "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos",
                "wtoeg", "label"],
        'congress-voting': 
                ["label", "handicapped_infants", "water_project_cost_sharing",
                "adoption_of_the_budget_resolution", "physician_fee_freeze",
                "el_salvador_aid", "religious_groups_in_schools",
                "anti_satellite_test_ban", "aid_to_nicaraguan_contras",
                "mx_missile", "immigration", "synfuels_corporation_cutback",
                "education_spending", "superfund_right_to_sue", "crime",
                "duty_free_exports", "export_administration_act_south_africa"],
        'contrac': 
                ["age", "wifes_education", "husbands_education", "nchildren",
                "religion", "working", "husbands_occupation", "living_std",
                "media_exposure", "label"],
        'credit-approval': 
                ["A" + str(i) for i in range(1, 15+1)] + ["label"],
        'ctg': 
                ["FileName", "Date", "b", "e", "LBE", "LB", "AC", "FM", "UC",
                "ASTV", "mSTV", "ALTV", "mLTV", "DL", "DS", "DP", "DR", "Width",
                "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median",
                "Variance", "Tendency", "A", "B", "C", "D", "SH", "AD", "DE",
                "LD", "FS", "SUSP", "CLASS", "label"],
        'cylinder-bands': 
                ["timestamp", "cylinder_number", "customer", "job_number",
                "grain_screened", "ink_color", "proof_on_ctd_ink", "blade_mfg",
                "cylinder_division", "paper_type", "ink_type", "direct_steam",
                "solvent_type", "type_on_cylinder", "press_type", "press",
                "unit_number", "cylinder_size", "paper_mill_location",
                "plating_tank", "proof_cut", "viscosity", "caliper",
                "ink_temperature", "humifity", "roughness", "blade_pressure",
                "varnish_pct", "press_speed", "ink_pct", "solvent_pct",
                "ESA_Voltage", "ESA_Amperage", "wax", "hardener",
                "roller_durometer", "current_density", "anode_space_ratio",
                "chrome_content", "label"],
        'dermatology': 
                ["erythema", "scaling", "definite_borders", "itching",
                "koebner_phenomenon", "polygonal_papules", "follicular_papules",
                "oral_mucosal_involvement", "knee_and_elbow_involvement",
                "scalp_involvement", "family_history", "melanin_incontinence",
                "eosinophils_in_the_infiltrate", "PNL_infiltrate",
                "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis",
                "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges",
                "elongation_of_the_rete_ridges",
                "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule",
                "munro_microabcess", "focal_hypergranulosis",
                "disappearance_of_the_granular_layer",
                "vacuolisation_and_damage_of_basal_layer", "spongiosis",
                "saw_tooth_appearance_of_retes", "follicular_horn_plug",
                "perifollicular_parakeratosis",
                "inflammatory_monoluclear_inflitrate", "band_like_infiltrate",
                "age", "label"],
        'german_credit': 
                ["Status_of_checking_account", "Duration_in_months",
                "Credit_history", "Purpose", "Credit_amount", "Savings_account",
                "Employment_since", "Installment_rate", "status_and_sex",
                "Other_debtors", "residence_since", "Property", "Age",
                "Other_installment", "Housing", "existing_credits", "Job",
                "Number_of_dependents", "Have_Telephone", "foreign_worker",
                "label"],
        'heart-cleveland':
                ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "label"],
        'ilpd':
                ["age", "gender", "tb", "db", "alkphos", "sgpt", "sgot", "tp",
               "alb", "ag", "label"],
        'mammo':
                ["BIRADS", "age", "shape", "margin", "density", "label"],
        'mushroom':
                ["label", "cap_shape", "cap_surface", "cap_color", "bruises",
                "odor", "gill_attachment", "gill_spacing", "gill_size",
                "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above",
                "stalk_surface_below", "stalk_color_above", "stalk_color_below",
                "veil_type", "veil_color", "ring_number", "ring_type",
                "spore_print_color", "population", "habitat"],
        'wine':
                ["label", "Alcohol", "Malic_acid", "Ash", "Alcalinity_ash",
                "Magnesium", "Total_phenols", "Flavanoids",
                "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity",
                "Hue", "OD280_OD315", "Proline"],
        'wine_qual':
                ["fixed.acidity", "volatile.acidity", "citric.acid",
                "residual.sugar", "chlorides", "free.sulfur.dioxide",
                "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol",
                "label", "color"]
    }
    
    # Make sure that the two dictionaries have the same keys
    assert columns.keys() == targets.keys(), "something is wrong with columns and targets dictionaries" 
    
    # Load the data from the csv file
    df = pd.read_csv("data/%s.csv" % name, sep=",", header=None, names=columns[name])
        
    # We want the target label/s to be 1, and the other/s 0
    if type(targets[name]) == range:
        df['label'] = (df['label'].isin(targets[name])).astype(int)
    else:
        df['label'] = (df['label'] == targets[name]).astype(int)
    
    # Create the dataframe of features
    X = df.drop(columns=['label'])
    y = df['label']
    
    return X, y


def load_and_process(name, seed=100, nbins=3):
    """
    Applies the whole loading and processing pipeline.
    
    Arguments:
    - name:  Name of the dataset to process, without the '.csv' extension
    - seed:  Random seed to be used when splitting the data intro train and test
    - nbins: Number of bins to partition numerical data into
    
    Returns:
    - X_train: DataFrame with the train features, cleaned and processed
    - X_test:  DataFrame with the test features, cleaned and processed
    - y_train: DataFrame with the train labels, values are only 0 and 1
    - y_test:  DataFrame with the test labels, values are only 0 and 1
    """
    
    X, y = load_dataset(name)
    
    X = clean(X)
    
    X_train, X_test = train_test_split(X, train_size=0.9, random_state=seed)
    y_train, y_test = train_test_split(y, train_size=0.9, random_state=seed)

    X_train, X_test = bin_features(X_train, X_test, nbins=nbins)

    X_train = one_hot_encode(X_train)
    X_test = one_hot_encode(X_test)
    
    return X_train, X_test, y_train, y_test

