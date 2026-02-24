python scripts/run_benchmark.py --test data/benchmark/attack_bench/test/none/persistent_exfiltrate_health/ --memory-backend none  --model gemini-3.1-pro-preview --num-workers 4
python scripts/run_benchmark.py --test data/benchmark/attack_bench/test/explicit/persistent_exfiltrate_health/ --memory-backend explicit  --model gemini-3.1-pro-preview --num-workers 4
python scripts/run_benchmark.py --test data/benchmark/attack_bench/test/mem0/persistent_exfiltrate_health/ --memory-backend mem0  --model gemini-3.1-pro-preview --num-workers 4
python scripts/run_benchmark.py --test data/benchmark/attack_bench/test/rag/persistent_exfiltrate_health/ --memory-backend rag  --model gemini-3.1-pro-preview --num-workers 4
python scripts/run_benchmark.py --test data/benchmark/attack_bench/test/context/persistent_exfiltrate_health/ --memory-backend context  --model gemini-3.1-pro-preview --num-workers 4
