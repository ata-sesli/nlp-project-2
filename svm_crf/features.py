import re


# --------------------------------------------------
# Hand-written token features
# --------------------------------------------------
#
# The features follow the three information sources from the course
# slides (POS/NER intro, slide 12):
#
# 1. The word itself (prior probability of word/tag)
# 2. Neighboring words (window of two words on each side)
# 3. Morphology and word shape (prefix, suffix, capitalization, digits)
#
# The same feature function can be reused by the SVM model, so that the
# only difference between SVM and CRF is the label dependency.

WINDOW = 2


def word_shape(token):
    """Map characters to X/x/d and collapse repeats: 'DE89' -> 'Xd'."""
    shape = re.sub(r"[A-Z]", "X", token)
    shape = re.sub(r"[a-z]", "x", shape)
    shape = re.sub(r"[0-9]", "d", shape)

    return re.sub(r"(.)\1+", r"\1", shape)


def token_features(token):
    """Features that describe one token on its own."""
    lower = token.lower()

    return {
        "lower": lower,
        "prefix3": lower[:3],
        "suffix3": lower[-3:],
        "shape": word_shape(token),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "is_digit": token.isdigit(),
        "has_digit": any(char.isdigit() for char in token),
        "has_at": "@" in token,
    }


def sentence_features(tokens):
    """Build one feature dictionary per token for a whole chunk."""
    base = [token_features(token) for token in tokens]
    features = []

    for index in range(len(tokens)):
        current = {"bias": 1.0}

        for offset in range(-WINDOW, WINDOW + 1):
            position = index + offset

            if position < 0:
                current[f"{offset}:BOS"] = True
                continue

            if position >= len(tokens):
                current[f"{offset}:EOS"] = True
                continue

            for name, value in base[position].items():
                current[f"{offset}:{name}"] = value

        features.append(current)

    return features
