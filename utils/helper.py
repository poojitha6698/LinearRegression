import numpy as np


def prepare_input(
    age,
    sex,
    bmi,
    children,
    smoker,
    region
):

    sex_encoded = 1 if sex == 'male' else 0

    smoker_encoded = 1 if smoker == 'yes' else 0

    region_mapping = {
        'northeast': 0,
        'northwest': 1,
        'southeast': 2,
        'southwest': 3
    }

    region_encoded = region_mapping[region]

    input_data = np.array([
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded
    ]).reshape(1, -1)

    return input_data