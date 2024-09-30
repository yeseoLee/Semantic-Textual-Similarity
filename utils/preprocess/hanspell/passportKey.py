import re
import requests


def get_passport_key():
    """네이버에서 '네이버 맞춤법 검사기' 페이지에서 passportKey를 획득

    - 네이버에서 '네이버 맞춤법 검사기'를 띄운 후
    html에서 passportKey를 검색하면 값을 찾을 수 있다.

    - 찾은 값을 spell_checker.py 48 line에 적용한다.
    """

    url = "https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=네이버+맞춤법+검사기"
    res = requests.get(url)

    html_text = res.text

    match = re.search(r'passportKey=([^&"}]+)', html_text)
    if match:
        passport_key = match.group(1)
        return passport_key
    else:
        return False


def fix_spell_checker_py_code(file_path, passportKey):
    """획득한 passportkey를 spell_checker.py파일에 적용"""

    pattern = r'"passportKey": ".*"'

    with open(file_path, "r", encoding="utf-8") as input_file:
        content = input_file.read()
        modified_content = re.sub(pattern, f'"passportKey": "{passportKey}"', content)

    with open(file_path, "w", encoding="utf-8") as output_file:
        output_file.write(modified_content)

    return


# before run
def init():
    spell_checker_file_path = "./hanspell/spell_checker.py"

    passport_key = get_passport_key()
    if passport_key:
        fix_spell_checker_py_code(spell_checker_file_path, passport_key)
    else:
        print("passportKey를 찾을 수 없습니다.")
