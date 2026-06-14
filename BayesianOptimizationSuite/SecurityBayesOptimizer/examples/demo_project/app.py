import hashlib
import subprocess

import yaml


DEBUG = True
API_KEY = "sk_live_1234567890abcdefSECRET"


def run_command(user_input):
    subprocess.run("echo " + user_input, shell=True)


def unsafe_eval(expr):
    return eval(expr)


def weak_hash(value):
    return hashlib.md5(value.encode()).hexdigest()


def parse_config(text):
    return yaml.load(text)
