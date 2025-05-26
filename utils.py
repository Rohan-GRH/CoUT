import re
from typing import List, Literal, Union

import yaml
from pydantic import BaseModel


class Example(BaseModel):
    question: str
    answer: str


class Config(BaseModel):
    system_prompt: str
    format: str
    fewshot: List[Example]


def load_config(task: str, config: Literal["baseline", "cot", "cod", "CoUT"]) -> Config:
    
    # 处理bigbench子任务，提取子任务名称
    if task.startswith("bigbench_"):
        # 提取bigbench后面的所有内容作为子任务名称
        subtask = task[len("bigbench_"):]
        config_path = f"./configs/bigbench_{subtask}_{config}.yaml"
    else:
        config_path = f"./configs/{task}_{config}.yaml"
    
    with open(config_path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))


def compose_request(config: Config, shot: int, question: str) -> str:
    request = config.system_prompt + "\n"
    if shot is None:
        shot = len(config.fewshot)
    if shot != 0:
        fewshot = [config.format.format(question=ex.question, answer=ex.answer) for ex in config.fewshot[:shot]]
        request += "\n".join(fewshot) + "\n"
    request += config.format.format(question=question, answer="")
    return request


def nth_percentile(values: list[float], percentile: float) -> float:
    values = sorted(values)
    index = min(round(percentile * len(values)), len(values)) - 1
    return values[index]


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def trimmed_average(values: list[float], percentile: float) -> float:
    values = sorted(values)
    count = round(len(values) * percentile)
    trimmed = values[count : len(values) - count]
    return average(trimmed)


def extract_number_from_string(s: str) -> Union[int, float]:
    match = re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?", s)
    if match:
        number_str = match.group().replace(",", "")  # Remove commas
        return float(number_str) if "." in number_str else int(number_str)
    return None