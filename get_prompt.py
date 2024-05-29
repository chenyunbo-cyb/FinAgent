import re

import prompt_config
# from prompt_config import PROMPT_TEMPLATES


def get_prompt_template(type: str, name: str):
    '''
    从prompt_config中加载模板内容
    type: "basic_info_qa", "core_indicator_statistics", "structured_info_extraction", "fund_analysis_report"的其中一种。
    '''

    # from . import prompt_config
    import importlib
    importlib.reload(prompt_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)


prompt_template = get_prompt_template("basic_info_qa",  "default")
# print(prompt_template)