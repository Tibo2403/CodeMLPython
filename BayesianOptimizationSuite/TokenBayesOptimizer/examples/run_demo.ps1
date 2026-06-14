$env:PYTHONPATH = "$PSScriptRoot\..\src"
python -m token_bayes_optimizer.cli --iterations 35 --quality-floor 0.82 --tasks "$PSScriptRoot\tasks.jsonl" --json-output token_optimization_result.json --csv-output token_optimization_history.csv --report-output token_optimization_report.md --html-output token_optimization_report.html
