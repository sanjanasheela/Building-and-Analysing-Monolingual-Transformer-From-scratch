import re

def fix_ravattu_brackets(text):
    """
    Removes spurious left brackets caused by OCR confusing the Telugu Ra-vattu.
    """
    # Regex Breakdown:
    # [\(\[\{]        -> Matches a stray left parenthesis, square bracket, or curly brace
    # (               -> Starts Capture Group 1 (we will keep everything inside this group)
    # ['\"‘“\s]*      -> Allows for optional quotes or spaces between the bracket and letter
    # [\u0c15-\u0c39] -> Matches any standard Telugu consonant
    # \u0c4d\u0c30    -> Matches the Virama (్) + Ra (ర) which creates the Ra-vattu (్ర)
    # )               -> Ends Capture Group 1
    
    pattern = r"[\(\[\{](['\"‘“\s]*[\u0c15-\u0c39]\u0c4d\u0c30)"
    
    # We replace the matched string with whatever was in Capture Group 1
    # This effectively deletes the bracket and leaves the Telugu text untouched.
    cleaned_text = re.sub(pattern, r"\1", text)
    
    return cleaned_text
