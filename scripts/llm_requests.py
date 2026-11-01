"""Proposal section 3, step 4: free paraphrases and language parsing.

1. An open instruction-tuned language model rewrites every scenario request twice
   (from template 0), keeping all requirements.
2. Every text (three templates and two paraphrases per scenario) is parsed twice:
   by the rule-based parser (request_text.parse) and by the language model asked
   for JSON (llm_text.parse_messages).
3. Parsing accuracy is reported per parser and text source. A paraphrase that the
   parsers cannot read back correctly either lost meaning or is hard to parse;
   the human audit (step 5) separates the two on a sample.

Download the model once on the login node (compute nodes may have no internet):
    export HF_HOME=$SCR/hf_cache
    python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B-Instruct')"
Then:
    sbatch --partition=gpumedium --time=01:00:00 --gres=gpu:1 jobs/run.sh configs/base.yaml \
        scripts/llm_requests.py --config configs/base.yaml

Writes $data_dir/processed/benchmark/requests_llm.jsonl and
reports/benchmark/requests_llm.md / requests_llm.json in the repository.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trajrisk import llm_text as lt
from trajrisk import request_text as rt

REPO = Path(__file__).resolve().parents[1]


def load_cfg(path):
    os.environ["TRAJRISK_CONFIG"] = str(path)
    from trajrisk.config import load_config
    try:
        return load_config(path)
    except TypeError:
        return load_config()


class Generator:
    def __init__(self, name, seed):
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                          device_map="cuda")
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.seed = seed

    @torch.no_grad()
    def __call__(self, conversations, max_new_tokens, sample, batch_size=32):
        out = []
        for i in range(0, len(conversations), batch_size):
            prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                       for c in conversations[i:i + batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(self.seed + i)
            kw = dict(do_sample=True, temperature=0.9, top_p=0.95) if sample else dict(do_sample=False)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **kw)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--paraphrases", type=int, default=2)
    parser.add_argument("--max-scenarios", type=int, default=None, help="for a quick test")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    store = Path(cfg["paths"]["data_dir"]) / "processed" / "benchmark"
    scen = {s["scenario_id"]: s for s in map(json.loads, open(store / "scenarios.jsonl"))}
    texts = [json.loads(line) for line in open(store / "requests.jsonl")]
    if args.max_scenarios:
        keep = set(list(scen)[:args.max_scenarios])
        scen = {k: v for k, v in scen.items() if k in keep}
        texts = [t for t in texts if t["scenario_id"] in keep]

    t0 = time.time()
    gen = Generator(args.model, cfg["project"]["seed"])

    # 1. paraphrases of template 0
    base = [t for t in texts if t["variant"] == 0]
    rows = [{"scenario_id": t["scenario_id"], "split": t["split"], "source": "template",
             "variant": t["variant"], "text": t["text"]} for t in texts]
    for p in range(args.paraphrases):
        gen.seed = cfg["project"]["seed"] + 1000 * (p + 1)
        raw = gen([lt.paraphrase_messages(t["text"]) for t in base], max_new_tokens=96, sample=True)
        rows += [{"scenario_id": t["scenario_id"], "split": t["split"], "source": "llm",
                  "variant": p, "text": lt.clean_paraphrase(r)} for t, r in zip(base, raw)]

    # 2. both parsers on every text
    raw = gen([lt.parse_messages(r["text"]) for r in rows], max_new_tokens=128, sample=False)
    for r, out in zip(rows, raw):
        s = scen[r["scenario_id"]]
        rule = rt.parse(r["text"])
        llm = lt.parse_llm_output(out)
        r.update({"parsed_rule": rule, "parsed_llm": llm, "llm_valid_json": llm["valid_json"]})
        r.update({f"rule_{k}": v for k, v in rt.compare(rule, s).items()})
        r.update({f"llm_{k}": v for k, v in rt.compare(llm, s).items()})

    with open(store / "requests_llm.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # 3. report
    df = pd.DataFrame(rows)
    cols = ["hard_exact", "soft_order", "all_correct"]
    table = {}
    for src in ("template", "llm"):
        part = df[df["source"] == src]
        table[src] = {"texts": int(len(part)),
                      **{f"rule_{c}": float(part[f"rule_{c}"].mean()) for c in cols},
                      **{f"llm_{c}": float(part[f"llm_{c}"].mean()) for c in cols},
                      "llm_valid_json": float(part["llm_valid_json"].mean())}
    info = {"model": args.model, "revision": gen.revision, "paraphrases_per_scenario": args.paraphrases,
            "paraphrase_sampling": {"temperature": 0.9, "top_p": 0.95}, "parsing": "greedy",
            "seconds": round(time.time() - t0), "accuracy": table}
    out = REPO / "reports" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "requests_llm.json").write_text(json.dumps(info, indent=2))

    L = ["# Proposal section 3, step 4: free paraphrases and language parsing", "",
         f"Model: {args.model} (revision {gen.revision}); {args.paraphrases} paraphrases per "
         f"scenario (temperature 0.9, top-p 0.95); parsing with greedy decoding; "
         f"{info['seconds']} s in total.", "",
         "| Text source | Texts | Parser | Hard constraints exact | Preference order | All correct |",
         "|---|---|---|---|---|---|"]
    for src, v in table.items():
        for p in ("rule", "llm"):
            L.append(f"| {src} | {v['texts']:,} | {'rule-based' if p == 'rule' else 'language model'} | "
                     f"{100 * v[f'{p}_hard_exact']:.1f}% | {100 * v[f'{p}_soft_order']:.1f}% | "
                     f"{100 * v[f'{p}_all_correct']:.1f}% |")
    L += ["", f"Language-model parser returned valid JSON for "
              f"{100 * df['llm_valid_json'].mean():.1f}% of texts.", "", "## Example paraphrases", ""]
    for sid in list(scen)[:6]:
        L.append(f"- template: {next(r['text'] for r in rows if r['scenario_id'] == sid and r['variant'] == 0 and r['source'] == 'template')}")
        L += [f"  - paraphrase: {r['text']}" for r in rows if r["scenario_id"] == sid and r["source"] == "llm"]
    fails = df[(df["source"] == "llm") & ~df["llm_all_correct"]].head(10)
    if len(fails):
        L += ["", "## Paraphrases the language-model parser did not read back correctly (first 10)", ""]
        L += [f"- {r.text}  ->  {r.parsed_llm['hard']} {r.parsed_llm['soft']}" for r in fails.itertuples()]
    (out / "requests_llm.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
