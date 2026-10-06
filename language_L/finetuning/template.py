"""
Synthetic Nepali comparative / basic-reasoning templates.

Examples are generated programmatically so every label is known.
Train vs eval leakage is avoided by holding out entity names and
attribute/relation families (Phase 3 spec).
"""

from __future__ import annotations

import random
from typing import Literal

RANDOM_SEED = 42

ANSWER_MARKER = "उत्तर"
QUESTION_TAG = "प्रश्न:"

# Names used only in the training pool.
TRAIN_NAMES = [
    "राम", "सीता", "हरि", "गीता", "कृष्ण",
    "माया", "सुरेश", "अनीता", "विजय", "प्रिया",
    "राहुल", "दिव्या", "किरण", "पूजा", "वरुण",
    "लक्ष्मी", "नागेश", "स्वाती", "हरीश", "मीना",
    "अरुण", "कविता", "सन्दीप", "भानु", "तेज",
    "प्रसाद", "महेश", "ज्योति", "सत्य", "राजेश",
    "वंशी", "कार्तिक", "राम्या", "अखिल", "सञ्जय",
]

# Names never used in training examples.
HELD_OUT_NAMES = [
    "श्रीन", "माधवी", "गोपी", "सुनीता", "चैतन्य",
    "अनुषा", "रवि", "पद्मा", "यशवन्त", "कीर्ति",
    "भार्गव", "समीर", "विनय", "दीपिका", "नवीन",
]

TRAIN_ITEMS = ["कलम", "किताब", "झोला", "ल्यापटप", "साइकल", "घडी"]
HELD_OUT_ITEMS = ["छाता", "कुर्सी", "फोन", "टोपी", "बोतल"]

TRAIN_CITIES = [
    "काठमाडौं", "पोखरा", "ललितपुर", "भक्तपुर", "विराटनगर",
    "वीरगञ्ज", "धनगढी", "नेपालगञ्ज", "हेटौंडा", "जनकपुर",
    "बुटवल", "धरान", "इटहरी", "गोरखा", "चितवन",
    "नारायणघाट", "दमक", "तुल्सीपुर", "सुर्खेत", "भैरहवा",
    "बिरेन्द्रनगर", "राजविराज", "लहान", "कलैया", "गौर",
    "भिमफेदी", "त्रिशूली", "बनेपा", "धुलिखेल", "पनौती",
]

HELD_OUT_CITIES = [
    "बागलुङ", "पाल्पा", "स्याङ्जा", "म्याग्दी", "पर्वत",
    "तनहुँ", "लमजुङ", "मनाङ", "मुस्ताङ", "कास्की",
    "नुवाकोट", "धादिङ", "मकवानपुर", "बारा", "पर्सा",
]

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
        "fact": "{a} {b} भन्दा अग्लो हो।",
        "q_extreme_high": "{people} मध्ये को सबैभन्दा अग्लो हो?",
        "q_extreme_low": "{people} मध्ये को सबैभन्दा होचो हो?",
        "q_pair": "{a} र {b} मध्ये को अग्लो हो?",
        "q_yesno_gt": "{a} {b} भन्दा अग्लो हो?",
        "high_word": "सबैभन्दा अग्लो",
        "low_word": "सबैभन्दा होचो",
    },
    "age": {
        "fact": "{a} को उमेर {b} भन्दा बढी हो।",
        "q_extreme_high": "{people} मध्ये को जेठो हो?",
        "q_extreme_low": "{people} मध्ये को कान्छो हो?",
        "q_pair": "{a} र {b} मध्ये को जेठो हो?",
        "q_yesno_gt": "{a} को उमेर {b} भन्दा बढी हो?",
        "high_word": "जेठो",
        "low_word": "कान्छो",
    },
    "money": {
        "fact": "{a} सँग {b} भन्दा बढी पैसा छ।",
        "q_extreme_high": "{people} मध्ये कससँग बढी पैसा छ?",
        "q_extreme_low": "{people} मध्ये कससँग कम पैसा छ?",
        "q_pair": "{a} र {b} मध्ये कससँग बढी पैसा छ?",
        "q_yesno_gt": "{a} सँग {b} भन्दा बढी पैसा छ?",
        "high_word": "बढी पैसा",
        "low_word": "कम पैसा",
    },
    "weight": {
        "fact": "{a} को तौल {b} भन्दा बढी हो।",
        "q_extreme_high": "{people} मध्ये को बढी तौलको हो?",
        "q_extreme_low": "{people} मध्ये को कम तौलको हो?",
        "q_pair": "{a} र {b} मध्ये को बढी तौलको हो?",
        "q_yesno_gt": "{a} को तौल {b} भन्दा बढी हो?",
        "high_word": "बढी तौल",
        "low_word": "कम तौल",
    },
    "points": {
        "fact": "{a} का अङ्क {b} भन्दा बढी छन्।",
        "q_extreme_high": "{people} मध्ये कसका बढी अङ्क छन्?",
        "q_extreme_low": "{people} मध्ये कसका कम अङ्क छन्?",
        "q_pair": "{a} र {b} मध्ये कसका बढी अङ्क छन्?",
        "q_yesno_gt": "{a} का अङ्क {b} भन्दा बढी छन्?",
        "high_word": "बढी अङ्क",
        "low_word": "कम अङ्क",
    },
    "temperature": {
        "fact": "{a} सहरमा तापक्रम {b} सहरभन्दा बढी छ।",
        "q_extreme_high": "{people} मध्ये कुन सहरमा तापक्रम बढी छ?",
        "q_extreme_low": "{people} मध्ये कुन सहरमा तापक्रम कम छ?",
        "q_pair": "{a} र {b} मध्ये कुन सहरमा तापक्रम बढी छ?",
        "q_yesno_gt": "{a} सहरमा तापक्रम {b} भन्दा बढी छ?",
        "high_word": "बढी तापक्रम",
        "low_word": "कम तापक्रम",
    },
    "distance": {
        "fact": "{a} को दूरी {b} भन्दा बढी हो।",
        "q_extreme_high": "{people} मध्ये कुनको दूरी बढी हो?",
        "q_extreme_low": "{people} मध्ये कुनको दूरी कम हो?",
        "q_pair": "{a} र {b} मध्ये कुनको दूरी बढी हो?",
        "q_yesno_gt": "{a} को दूरी {b} भन्दा बढी हो?",
        "high_word": "बढी दूरी",
        "low_word": "कम दूरी",
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
        return f"{names[0]} र {names[1]}"
    return ", ".join(names[:-1]) + f" र {names[-1]}"


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
            f"{n[0]} सँग एउटा {item} छ, तर {n[1]} सँग छैन। "
            f"कससँग {item} छ?"
        )
        output = f"{n[0]} सँग {item} छ, त्यसैले {ANSWER_MARKER} {n[0]}।"
        return _pack(instruction, output, "Direct comparisons", 1, n[0])
    if variant == "lives":
        instruction = (
            f"{n[0]} {city} मा बस्छन्, {n[1]} बस्दैनन्। "
            f"{n[0]} र {n[1]} मध्ये को {city} मा छ?"
        )
        output = f"{n[0]} {city} मा बस्छन्, त्यसैले {ANSWER_MARKER} {n[0]}।"
        return _pack(instruction, output, "Direct comparisons", 1, n[0])
    if variant == "taller":
        a, b = n[0], n[1]
        if random.choice([True, False]):
            a, b = b, a
        instruction = f"{a} {b} भन्दा अग्लो हो। को अग्लो हो?"
        output = f"{a} {b} भन्दा अग्लो हो, त्यसैले {ANSWER_MARKER} {a}।"
        return _pack(instruction, output, "Direct comparisons", 1, a)
    a, b = n[0], n[1]
    instruction = f"{a} को उमेर {b} भन्दा बढी हो। को जेठो हो?"
    output = f"{a} को उमेर बढी हो, त्यसैले {ANSWER_MARKER} {a}।"
    return _pack(instruction, output, "Direct comparisons", 1, a)


def _numerical(n: list[str]) -> dict:
    lo = random.randint(10, 80)
    hi = random.randint(lo + 5, 400)
    ask_more = random.choice([True, False])
    if random.choice([True, False]):
        instruction = (
            f"{n[0]} सँग {lo} रुपैयाँ छ, {n[1]} सँग {hi} रुपैयाँ छ। "
            + ("कससँग बढी पैसा छ?" if ask_more else "कससँग कम पैसा छ?")
        )
        if ask_more:
            output = f"{hi} > {lo} त्यसैले {ANSWER_MARKER} {n[1]}।"
            ans = n[1]
        else:
            output = f"{lo} < {hi} त्यसैले {ANSWER_MARKER} {n[0]}।"
            ans = n[0]
        return _pack(instruction, output, "Numerical comparisons", 1, ans)

    instruction = (
        f"एउटा बाकसमा {lo} स्याउ छन्, अर्कोमा {hi} स्याउ छन्। "
        + ("कुनमा बढी छन्?" if ask_more else "कुनमा कम छन्?")
    )
    if ask_more:
        output = f"{hi} > {lo} त्यसैले {ANSWER_MARKER} दोस्रो बाकस।"
        ans = "दोस्रो बाकस"
    else:
        output = f"{lo} < {hi} त्यसैले {ANSWER_MARKER} पहिलो बाकस।"
        ans = "पहिलो बाकस"
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
            f"क्रम {chain_sym}। त्यसैले {ordered[0]} {lang['high_word']} हो। "
            f"{ANSWER_MARKER} {ordered[0]}।"
        )
        ans = ordered[0]
    elif qtype == "extreme_low":
        instruction = f"{fact_text} {lang['q_extreme_low'].format(people=people)}"
        output = (
            f"क्रम {chain_sym}। त्यसैले {ordered[-1]} {lang['low_word']} हो। "
            f"{ANSWER_MARKER} {ordered[-1]}।"
        )
        ans = ordered[-1]
    elif qtype == "pair":
        ask_high = random.choice([True, False])
        instruction = f"{fact_text} {lang['q_pair'].format(a=ordered[0], b=ordered[-1])}"
        if ask_high:
            output = (
                f"क्रमबद्ध सम्बन्ध: {chain_sym}। "
                f"{ordered[0]} {ordered[-1]} भन्दा बढी हो। {ANSWER_MARKER} {ordered[0]}।"
            )
            ans = ordered[0]
        else:
            instruction = (
                f"{fact_text} {ordered[0]} र {ordered[-1]} मध्ये को कम हो?"
            )
            output = (
                f"क्रमबद्ध सम्बन्ध: {chain_sym}। "
                f"{ordered[-1]} {ordered[0]} भन्दा कम हो। {ANSWER_MARKER} {ordered[-1]}।"
            )
            ans = ordered[-1]
    else:
        ask_true = random.choice([True, False])
        if ask_true:
            instruction = f"{fact_text} {lang['q_yesno_gt'].format(a=ordered[0], b=ordered[-1])}"
            output = (
                f"क्रमबद्ध सम्बन्ध: {chain_sym}। हो, {ordered[0]} बढी हो। {ANSWER_MARKER} हो।"
            )
            ans = "हो"
        else:
            instruction = f"{fact_text} {lang['q_yesno_gt'].format(a=ordered[-1], b=ordered[0])}"
            output = (
                f"क्रमबद्ध सम्बन्ध: {chain_sym}। होइन, {ordered[0]} बढी हो। {ANSWER_MARKER} होइन।"
            )
            ans = "होइन"
    return _pack(instruction, output, category, hops, ans)


def _gse(n: list[str]) -> dict:
    variant = random.choice(["nums_gt", "nums_eq", "people_eq"])
    if variant == "nums_gt":
        a = random.randint(10, 50)
        b = random.randint(51, 120)
        ask_big = random.choice([True, False])
        instruction = (
            f"{a} र {b} सङ्ख्यालाई तुलना गर्नुहोस्। "
            + ("कुन ठूलो हो?" if ask_big else "कुन सानो हो?")
        )
        if ask_big:
            output = f"{b} > {a} त्यसैले {ANSWER_MARKER} {b}।"
            ans = str(b)
        else:
            output = f"{a} < {b} त्यसैले {ANSWER_MARKER} {a}।"
            ans = str(a)
        return _pack(instruction, output, "Greater/smaller/equal cases", 1, ans)
    if variant == "nums_eq":
        v = random.randint(10, 100)
        equal = random.choice([True, False])
        other = v if equal else v + random.randint(3, 20)
        instruction = f"दिइएका दुई सङ्ख्या {v} र {other} बराबर हुन् कि असमान?"
        if equal:
            output = f"{v} = {other} त्यसैले {ANSWER_MARKER} बराबर।"
            ans = "बराबर"
        else:
            output = f"{v} ≠ {other} त्यसैले {ANSWER_MARKER} असमान।"
            ans = "असमान"
        return _pack(instruction, output, "Greater/smaller/equal cases", 1, ans)
    v = random.randint(140, 180)
    instruction = (
        f"{n[0]} को उचाइ {v} से.मि. छ, {n[1]} को उचाइ पनि {v} से.मि. छ। "
        f"{n[0]} र {n[1]} मध्ये को अग्लो हो, वा बराबर?"
    )
    output = f"दुवैको उचाइ {v} से.मि. छ, त्यसैले {ANSWER_MARKER} बराबर।"
    return _pack(instruction, output, "Greater/smaller/equal cases", 1, "बराबर")


def generate_single_sample(category: str, pool: Pool = "train") -> dict:
    """Create one labelled Nepali reasoning example in `pool`."""
    return _sample(category, pool)


def format_prompt(user_text: str) -> str:
    """Prompt shown to the model at train and eval time (answer omitted)."""
    return f"प्रश्न: {user_text}\nजवाफ:"


def format_example(sample: dict) -> str:
    """Full supervised sequence: question + gold answer."""
    user = sample["messages"][0]["content"]
    assistant = sample["messages"][1]["content"]
    return f"{format_prompt(user)} {assistant}"
