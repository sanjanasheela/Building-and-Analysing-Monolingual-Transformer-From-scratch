"""
Synthetic Telugu comparative / basic-reasoning templates.

Examples are generated programmatically so every label is known.
Train vs eval leakage is avoided by holding out entity names and
attribute/relation families (Phase 3 spec).
"""

from __future__ import annotations

import random
from typing import Literal

RANDOM_SEED = 42

# Names used only in the training pool (expanded to ~35 names)
TRAIN_NAMES = [
    "రమేష్", "సురేష్", "అనిత", "విజయ్", "ప్రియ",
    "రాహుల్", "దివ్య", "కిరణ్", "పూజ", "వరుణ్",
    "లక్ష్మి", "నాగేశ్", "స్వాతి", "హరీష్", "మీనా",
    "అరుణ్", "కవిత", "సందీప్", "భాను", "తేజ",
    "ప్రసాద్", "మహేష్", "సుధాకర్", "లావణ్య", "జ్యోతి",
    "సత్య", "స్రవంతి", "రాఘవ", "భవానీ", "రాజేష్",
    "వంశీ", "కార్ತಿక్", "సుప్రియ", "రమ్య", "అఖిల్"
]

# Names never used in training examples (expanded to ~15 held-out names)
HELD_OUT_NAMES = [
    "శ్రీను", "మాధవి", "గోపి", "సునీత", "చైతన్య",
    "అనుష", "రవి", "పద్మ", "యశ్వంత్", "కీర్తి",
    "భార్గవ్", "సమీర", "వినయ్", "దీపిక", "నవీన్"
]

TRAIN_ITEMS = ["పెన్ను", "పుస్తకం", "బ్యాగ్", "లాప్‌టాప్", "సైకిల్", "గడియారం"]
HELD_OUT_ITEMS = ["గొడుగు", "కుర్చీ", "ఫోన్", "టోపీ", "బాటిల్"]

# Cities used only in the training pool (expanded to ~35 cities)
TRAIN_CITIES = [
    "హైదరాబాద్", "బెంగళూరు", "చెన్నై", "ముంబై", "విశాఖపట్నం",
    "విజయవాడ", "తిరుపతి", "వరంగల్", "కరీంనగర్", "గుంటూరు",
    "కాకినాడ", "రాజమహేంద్రవరం", "నెల్లూరు", "కర్నూలు", "నిజామాబాద్",
    "ఖమ్మం", "ఎల్దుర్తి", "మహబూబ్ నగర్", "నల్గొండ", "సికింద్రాబాద్",
    "పుణే", "అహ్మదాబాద్", "కోల్‌కతా", "జైపూర్", "లక్నో",
    "పాట్నా", "భోపాల్", "చండీగఢ్", "కొచ్చి", "తిరువనంతపురం",
    "మైసూర్", "మంగళూరు", "కోయంబత్తూర్", "మధురై", "నాగ్‌పూర్"
]

# Cities never used in training examples (expanded to ~15 held-out cities)
HELD_OUT_CITIES = [
    "అమరావతి", "ఒంగోలు", "శ్రీకాకుళం", "ఏలూరు", "మచిలీపట్నం",
    "ప్రొద్దుటూరు", "హిందూపూర్", "చిత్తూరు", "మదనపల్లె", "అనంతపురం",
    "బాపట్ల", "నర్సరావుపేట", "గూడూరు", "పాలకొల్లు", "తణుకు"
]
# Attribute families used in train vs held-out splits.
TRAIN_CHAIN_ATTRS = ("height", "age", "money", "weight")
HELD_OUT_CHAIN_ATTRS = ("points", "temperature", "distance")

CATEGORIES = [
    "Direct comparisons",
    "Numerical comparisons",
    "2-hop comparisons",
    "3-hop comparisons",
    "4-hop comparisons",
    "Greater/smaller/equal cases",
]

Pool = Literal["train", "held_out"]

_CHAIN_LANG = {
    "height": {
        "fact": "{a} {b} కంటే పొడవు.",
        "chain_intro": "ఎత్తు",
        "q_extreme_high": "{people} లో ఎవరు అత్యంత పొడవు?",
        "q_extreme_low": "{people} లో ఎవరు అత్యంత పొట్టి?",
        "q_pair": "{a} మరియు {b} లో ఎవరు పొడవు?",
        "q_yesno_gt": "{a} {b} కంటే పొడవుగా ఉన్నారా?",
        "reason_gt": "పొడవు",
        "high_word": "అత్యంత పొడవు",
        "low_word": "అత్యంత పొట్టి",
    },
    "age": {
        "fact": "{a} వయసు {b} కంటే ఎక్కువ.",
        "chain_intro": "వయసు",
        "q_extreme_high": "{people} లో ఎవరు పెద్దవారు?",
        "q_extreme_low": "{people} లో ఎవరు చిన్నవారు?",
        "q_pair": "{a} మరియు {b} లో ఎవరు పెద్దవారు?",
        "q_yesno_gt": "{a} వయసు {b} కంటే ఎక్కువా?",
        "reason_gt": "వయసు",
        "high_word": "పెద్దవారు",
        "low_word": "చిన్నవారు",
    },
    "money": {
        "fact": "{a} వద్ద {b} కంటే ఎక్కువ డబ్బు ఉంది.",
        "chain_intro": "డబ్బు",
        "q_extreme_high": "{people} లో ఎవరి వద్ద ఎక్కువ డబ్బు ఉంది?",
        "q_extreme_low": "{people} లో ఎవరి వద్ద తక్కువ డబ్బు ఉంది?",
        "q_pair": "{a} మరియు {b} లో ఎవరి వద్ద ఎక్కువ డబ్బు ఉంది?",
        "q_yesno_gt": "{a} వద్ద {b} కంటే ఎక్కువ డబ్బు ఉందా?",
        "reason_gt": "డబ్బు",
        "high_word": "ఎక్కువ డబ్బు",
        "low_word": "తక్కువ డబ్బు",
    },
    "weight": {
        "fact": "{a} బరువు {b} కంటే ఎక్కువ.",
        "chain_intro": "బరువు",
        "q_extreme_high": "{people} లో ఎవరు ఎక్కువ బరువు?",
        "q_extreme_low": "{people} లో ఎవరు తక్కువ బరువు?",
        "q_pair": "{a} మరియు {b} లో ఎవరు ఎక్కువ బరువు?",
        "q_yesno_gt": "{a} బరువు {b} కంటే ఎక్కువా?",
        "reason_gt": "బరువు",
        "high_word": "ఎక్కువ బరువు",
        "low_word": "తక్కువ బరువు",
    },
    "points": {
        "fact": "{a} పాయింట్లు {b} కంటే ఎక్కువ.",
        "chain_intro": "పాయింట్లు",
        "q_extreme_high": "{people} లో ఎవరికి ఎక్కువ పాయింట్లు ఉన్నాయి?",
        "q_extreme_low": "{people} లో ఎవరికి తక్కువ పాయింట్లు ఉన్నాయి?",
        "q_pair": "{a} మరియు {b} లో ఎవరికి ఎక్కువ పాయింట్లు ఉన్నాయి?",
        "q_yesno_gt": "{a} పాయింట్లు {b} కంటే ఎక్కువా?",
        "reason_gt": "పాయింట్లు",
        "high_word": "ఎక్కువ పాయింట్లు",
        "low_word": "తక్కువ పాయింట్లు",
    },
    "temperature": {
        "fact": "{a} నగరంలో ఉష్ణోగ్రత {b} నగరం కంటే ఎక్కువ.",
        "chain_intro": "ఉష్ణోగ్రత",
        "q_extreme_high": "{people} లో ఏ నగరంలో ఉష్ణోగ్రత ఎక్కువ?",
        "q_extreme_low": "{people} లో ఏ నగరంలో ఉష్ణోగ్రత తక్కువ?",
        "q_pair": "{a} మరియు {b} లో ఏ నగరంలో ఉష్ణోగ్రత ఎక్కువ?",
        "q_yesno_gt": "{a} నగరంలో ఉష్ణోగ్రత {b} కంటే ఎక్కువా?",
        "reason_gt": "ఉష్ణోగ్రత",
        "high_word": "ఎక్కువ ఉష్ణోగ్రత",
        "low_word": "తక్కువ ఉష్ణోగ్రత",
    },
    "distance": {
        "fact": "{a} దూరం {b} కంటే ఎక్కువ.",
        "chain_intro": "దూరం",
        "q_extreme_high": "{people} లో ఏది ఎక్కువ దూరం?",
        "q_extreme_low": "{people} లో ఏది తక్కువ దూరం?",
        "q_pair": "{a} మరియు {b} లో ఏది ఎక్కువ దూరం?",
        "q_yesno_gt": "{a} దూరం {b} కంటే ఎక్కువా?",
        "reason_gt": "దూరం",
        "high_word": "ఎక్కువ దూరం",
        "low_word": "తక్కువ దూరం",
    },
}


def _names_for(pool: Pool) -> list[str]:
    return list(TRAIN_NAMES if pool == "train" else HELD_OUT_NAMES)


def _items_for(pool: Pool) -> list[str]:
    return list(TRAIN_ITEMS if pool == "train" else HELD_OUT_ITEMS)


def _cities_for(pool: Pool) -> list[str]:
    return list(TRAIN_CITIES if pool == "train" else HELD_OUT_CITIES)


def _attrs_for(pool: Pool) -> tuple[str, ...]:
    return TRAIN_CHAIN_ATTRS if pool == "train" else HELD_OUT_CHAIN_ATTRS


def _join_people(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} మరియు {names[1]}"
    return ", ".join(names[:-1]) + f" మరియు {names[-1]}"


def _sample(category: str, pool: Pool) -> dict:
    names = _names_for(pool)
    items = _items_for(pool)
    cities = _cities_for(pool)
    n = random.sample(names, k=min(5, len(names)))
    item = random.choice(items)
    city = random.choice(cities)

    if category == "Direct comparisons":
        return _direct(n, item, city)
    if category == "Numerical comparisons":
        return _numerical(n)
    if category in ("2-hop comparisons", "3-hop comparisons", "4-hop comparisons"):
        hops = int(category[0])
        attr = random.choice(_attrs_for(pool))
        n_ent = hops + 1
        if attr in ("temperature", "distance"):
            entities = random.sample(cities, k=n_ent)
        else:
            entities = n[:n_ent]
        return _chain(entities, attr, hops)
    if category == "Greater/smaller/equal cases":
        return _gse(n)
    raise ValueError(f"Unknown category: {category}")


def _pack(instruction: str, output: str, category: str, hop_count: int, answer_option: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ],
        "category": category,
        "hop_count": hop_count,
        "answer_option": answer_option,
    }


def _direct(n: list[str], item: str, city: str) -> dict:
    variant = random.choice(["has_item", "lives", "taller", "older"])
    if variant == "has_item":
        instruction = (
            f"{n[0]} దగ్గర ఒక {item} ఉంది, కానీ {n[1]} దగ్గర లేదు. "
            f"ఎవరి దగ్గర {item} ఉంది?"
        )
        output = f"{n[0]} దగ్గర {item} ఉంది కాబట్టి సమాధానం {n[0]}."
        return _pack(instruction, output, "Direct comparisons", 1, n[0])
    if variant == "lives":
        instruction = (
            f"{n[0]} {city} లో నివసిస్తున్నారు, {n[1]} నివసించరు. "
            f"{n[0]} మరియు {n[1]} లో ఎవరు {city} లో ఉన్నారు?"
        )
        output = f"{n[0]} {city} లో నివసిస్తున్నారు కాబట్టి సమాధానం {n[0]}."
        return _pack(instruction, output, "Direct comparisons", 1, n[0])
    if variant == "taller":
        a, b = n[0], n[1]
        if random.choice([True, False]):
            a, b = b, a
        instruction = f"{a} {b} కంటే పొడవు. ఎవరు పొడవు?"
        output = f"{a} {b} కంటే పొడవు కాబట్టి సమాధానం {a}."
        return _pack(instruction, output, "Direct comparisons", 1, a)
    a, b = n[0], n[1]
    instruction = f"{a} వయసు {b} కంటే ఎక్కువ. ఎవరు పెద్దవారు?"
    output = f"{a} వయసు ఎక్కువ కాబట్టి సమాధానం {a}."
    return _pack(instruction, output, "Direct comparisons", 1, a)


def _numerical(n: list[str]) -> dict:
    lo = random.randint(10, 80)
    hi = random.randint(lo + 5, 400)
    ask_more = random.choice([True, False])
    if random.choice([True, False]):
        instruction = (
            f"{n[0]} వద్ద {lo} రూపాయలు ఉన్నాయి, {n[1]} వద్ద {hi} రూపాయలు ఉన్నాయి. "
            + ("ఎవరి వద్ద ఎక్కువ డబ్బు ఉంది?" if ask_more else "ఎవరి వద్ద తక్కువ డబ్బు ఉంది?")
        )
        if ask_more:
            output = f"{hi} > {lo} కాబట్టి సమాధానం {n[1]}."
            ans = n[1]
        else:
            output = f"{lo} < {hi} కాబట్టి సమాధానం {n[0]}."
            ans = n[0]
        return _pack(instruction, output, "Numerical comparisons", 1, ans)

    instruction = (
        f"ఒక పెట్టెలో {lo} ఆపిల్స్ ఉన్నాయి, మరొక దానిలో {hi} ఆపిల్స్ ఉన్నాయి. "
        + ("దేనిలో ఎక్కువ ఉన్నాయి?" if ask_more else "దేనిలో తక్కువ ఉన్నాయి?")
    )
    if ask_more:
        output = f"{hi} > {lo} కాబట్టి సమాధానం రెండవ పెట్టె."
        ans = "రెండవ పెట్టె"
    else:
        output = f"{lo} < {hi} కాబట్టి సమాధానం మొదటి పెట్టె."
        ans = "మొదటి పెట్టె"
    return _pack(instruction, output, "Numerical comparisons", 1, ans)


def _chain(ordered: list[str], attr: str, hops: int) -> dict:
    """ordered[0] > ordered[1] > ... along `attr`."""
    lang = _CHAIN_LANG[attr]
    facts = [lang["fact"].format(a=ordered[i], b=ordered[i + 1]) for i in range(len(ordered) - 1)]
    fact_text = " ".join(facts)
    people = _join_people(ordered)
    chain_sym = " > ".join(ordered)
    category = f"{hops}-hop comparisons"

    qtype = random.choice(["extreme_high", "extreme_low", "pair", "yesno"])
    if qtype == "extreme_high":
        instruction = f"{fact_text} {lang['q_extreme_high'].format(people=people)}"
        output = (
            f"క్రమం {chain_sym}. కాబట్టి {ordered[0]} కి {lang['high_word']}. "
            f"సమాధానం {ordered[0]}."
        )
        ans = ordered[0]
    elif qtype == "extreme_low":
        instruction = f"{fact_text} {lang['q_extreme_low'].format(people=people)}"
        output = (
            f"క్రమం {chain_sym}. కాబట్టి {ordered[-1]} కి {lang['low_word']}. "
            f"సమాధానం {ordered[-1]}."
        )
        ans = ordered[-1]
    elif qtype == "pair":
        ask_high = random.choice([True, False])
        instruction = f"{fact_text} {lang['q_pair'].format(a=ordered[0], b=ordered[-1])}"
        if ask_high:
            output = (
                f"అనుక్రమ సంబంధం: {chain_sym}. "
                f"{ordered[0]} {ordered[-1]} కంటే ఎక్కువ. సమాధానం {ordered[0]}."
            )
            ans = ordered[0]
        else:
            instruction = (
                f"{fact_text} {ordered[0]} మరియు {ordered[-1]} లో ఎవరు తక్కువ?"
            )
            output = (
                f"అనుక్రమ సంబంధం: {chain_sym}. "
                f"{ordered[-1]} {ordered[0]} కంటే తక్కువ. సమాధానం {ordered[-1]}."
            )
            ans = ordered[-1]
    else:
        ask_true = random.choice([True, False])
        if ask_true:
            instruction = f"{fact_text} {lang['q_yesno_gt'].format(a=ordered[0], b=ordered[-1])}"
            output = (
                f"అనుక్రమ సంబంధం: {chain_sym}. అవును, {ordered[0]} ఎక్కువ. సమాధానం అవును."
            )
            ans = "అవును"
        else:
            instruction = f"{fact_text} {lang['q_yesno_gt'].format(a=ordered[-1], b=ordered[0])}"
            output = (
                f"అనుక్రమ సంబంధం: {chain_sym}. కాదు, {ordered[0]} ఎక్కువ. సమాధానం కాదు."
            )
            ans = "కాదు"
    return _pack(instruction, output, category, hops, ans)


def _gse(n: list[str]) -> dict:
    variant = random.choice(["nums_gt", "nums_eq", "people_eq"])
    if variant == "nums_gt":
        a = random.randint(10, 50)
        b = random.randint(51, 120)
        ask_big = random.choice([True, False])
        instruction = (
            f"{a} మరియు {b} సంఖ్యలను పోల్చండి. "
            + ("ఏది పెద్దది?" if ask_big else "ఏది చిన్నది?")
        )
        if ask_big:
            output = f"{b} > {a} కాబట్టి సమాధానం {b}."
            ans = str(b)
        else:
            output = f"{a} < {b} కాబట్టి సమాధానం {a}."
            ans = str(a)
        return _pack(instruction, output, "Greater/smaller/equal cases", 1, ans)
    if variant == "nums_eq":
        v = random.randint(10, 100)
        equal = random.choice([True, False])
        other = v if equal else v + random.randint(3, 20)
        instruction = f"ఇచ్చిన రెండు సంఖ్యలు {v} మరియు {other} సమానమా లేదా అసమానమా?"
        if equal:
            output = f"{v} = {other} కాబట్టి సమాధానం సమానం."
            ans = "సమానం"
        else:
            output = f"{v} ≠ {other} కాబట్టి సమాధానం అసమానం."
            ans = "అసమానం"
        return _pack(instruction, output, "Greater/smaller/equal cases", 1, ans)
    v = random.randint(140, 180)
    instruction = (
        f"{n[0]} ఎత్తు {v} సెం.మీ., {n[1]} ఎత్తు కూడా {v} సెం.మీ. "
        f"{n[0]} మరియు {n[1]} లో ఎవరు పొడవు, లేదా సమానమా?"
    )
    output = f"ఇద్దరి ఎత్తు {v} సెం.మీ. కాబట్టి సమాధానం సమానం."
    return _pack(instruction, output, "Greater/smaller/equal cases", 1, "సమానం")


def generate_single_sample(category: str, pool: Pool = "train") -> dict:
    """Create one labelled Telugu reasoning example in `pool`."""
    return _sample(category, pool)


def format_prompt(user_text: str) -> str:
    """Prompt shown to the model at train and eval time (answer omitted)."""
    return f"ప్రశ్న: {user_text}\nజవాబు:"


def format_example(sample: dict) -> str:
    """Full supervised sequence: question + gold answer."""
    user = sample["messages"][0]["content"]
    assistant = sample["messages"][1]["content"]
    return f"{format_prompt(user)} {assistant}"
