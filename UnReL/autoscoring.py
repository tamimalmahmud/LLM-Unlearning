from __future__ import annotations
from typing import List, Dict, Tuple, Any, Optional
import os, re, json, time, math, datetime as dt, random
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ---------- determinism ----------
os.environ.setdefault("PYTHONHASHSEED","0")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
random.seed(0)
try:
    import numpy as np
    np.random.seed(0)
except Exception:
    np=None
torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

def pick_device(pref: str="cuda")->str:
    return "cuda" if pref=="cuda" and torch.cuda.is_available() else "cpu"

# ---------- Luhn (credit card) ----------
def luhn_valid(num: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", num)]
    if not (13 <= len(digits) <= 19): return False
    s, alt = 0, False
    for d in reversed(digits):
        s += (d*2 - 9) if alt and d>=5 else (d*2 if alt else d)
        alt = not alt
    return s % 10 == 0

class AutoScoringForgetProbability:
    """
    Global-format risk scorer:
      P_forget = w_r * risk_norm + w_f * fresh_norm
    risk = semantic(personal/medical/financial) + token counts + global PII signals
    freshness = exponential decay from best-available date
    """

    DEFAULT_WEIGHTS = dict(w_r=0.7, w_f=0.3)
    DEFAULT_MIX = dict(gamma_sem=0.5, gamma_cnt=0.5)
    HALF_LIFE_DAYS = 180
    MIN_YEAR = 1900
    MAX_YEAR = 2100

    CATEGORY_LEX = {
        "medical": [
            r"\bmedical\b", r"\bpatient\b", r"\bdiagnosis\b", r"\btreatment\b",
            r"\bclinical\b", r"\bprescription\b", r"\bhealth\b", r"\bcondition\b"
        ],
        "personal": [
            r"\bname\b", r"\baddress\b", r"\bcity\b", r"\bcountry\b",
            r"\bpostal(?:\s*code)?\b", r"\bpostcode\b", r"\bzip\b",
            r"\bdate of birth\b", r"\bdob\b", r"\bcontact\b"
        ],
        "financial": [
            r"\baccount\b", r"\biban\b", r"\bcard\b", r"\bcredit\b", r"\bdebit\b",
            r"\binvestment\b", r"\btransaction\b", r"\bbalance\b", r"\bpayment\b"
        ],
    }

    # Global PII / identifiers (no country specifics)
    PII_REGEXES = {
        "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I),
        "phone_e164": re.compile(r"\+?[1-9]\d{7,14}\b"),  # E.164-ish (8–15 digits, leading nonzero)
        "ipv4": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
        "ipv6": re.compile(r"\b(?:[A-F0-9]{1,4}:){2,7}[A-F0-9]{1,4}\b", re.I),
        "url": re.compile(r"\bhttps?://[^\s]+", re.I),
        "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
        # generic IDs near key hints (no specific country): capture 6–20 alnum tokens near keywords
        "id_near_key": re.compile(
            r"(?i)\b(id|identity|identifier|document|license|licence|passport|tax|tin|ssn|national|gov|uid|user\s*id)\b"
            r"[^A-Za-z0-9]{0,20}([A-Z0-9][A-Z0-9\-]{5,19})"
        ),
        # credit card candidate (validate with Luhn)
        "cc_candidate": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    }

    DATE_FIELDS = ["date","created_at","timestamp","published_at","updated_at","ingested_at"]

    def __init__(self, device="cuda", weights=None, mix=None, half_life_days=None,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = pick_device(device)
        self.weights = (weights or self.DEFAULT_WEIGHTS).copy()
        self.mix = (mix or self.DEFAULT_MIX).copy()
        self.half_life_days = half_life_days or self.HALF_LIFE_DAYS
        self.lmbda = math.log(2) / self.half_life_days
        self.model = SentenceTransformer(model_name, device=self.device)

        # semantic prototypes
        self.category_prototypes = {
            "medical": self._embed_mean(["medical patient diagnosis treatment clinical healthcare"]),
            "personal": self._embed_mean(["personal identity private address contact profile"]),
            "financial": self._embed_mean(["finance banking account transaction credit debit investment money"]),
        }
        # compile category lex
        self._compiled_lex = {c:[re.compile(p, re.I) for p in pats] for c,pats in self.CATEGORY_LEX.items()}

    # ---- helpers
    def _embed_mean(self, texts: List[str]) -> torch.Tensor:
        emb = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return emb if emb.ndim==1 else emb.mean(0)

    def _semantic_category_scores(self, text: str) -> Dict[str,float]:
        emb = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        out={}
        for cat, proto in self.category_prototypes.items():
            sim = util.cos_sim(emb, proto).item()    # [-1,1]
            out[cat] = 0.5*(sim+1.0)                 # -> [0,1]
        return out

    @staticmethod
    def _soft_count_to_score(count: int, alpha=0.25, cap=30) -> float:
        c = min(max(count,0), cap)
        return 1.0 - math.exp(-alpha*c)

    def _category_token_counts(self, text: str) -> Dict[str,int]:
        counts={}
        for c, pats in self._compiled_lex.items():
            counts[c] = sum(len(p.findall(text)) for p in pats)
        return counts

    # Flatten ALL text from JSON (keys & values)
    def _flatten_json_texts(self, obj) -> str:
        out=[]
        if isinstance(obj, dict):
            for k,v in obj.items():
                out.append(str(k))
                out.append(self._flatten_json_texts(v))
        elif isinstance(obj, list):
            for it in obj:
                out.append(self._flatten_json_texts(it))
        else:
            if obj is None: pass
            elif isinstance(obj, str): out.append(obj)
            else: out.append(str(obj))
        return " ".join(s for s in out if s)

    # Global PII signal
    def _pii_signal(self, text: str) -> float:
        counts = 0
        # simple regex hits
        for name, rgx in self.PII_REGEXES.items():
            if name == "cc_candidate":
                # validate each candidate with Luhn
                for m in rgx.findall(text):
                    if luhn_valid(m): counts += 1
            elif name == "id_near_key":
                counts += len(rgx.findall(text))
            else:
                counts += len(rgx.findall(text))
        return self._soft_count_to_score(counts, alpha=0.20, cap=40)

    # dates
    @staticmethod
    def _parse_date_any(s: str) -> Optional[dt.date]:
        s=s.strip()
        for fmt in ("%Y-%m-%d","%Y/%m/%d","%Y.%m.%d","%d-%m-%Y","%d/%m/%Y"):
            try: return dt.datetime.strptime(s, fmt).date()
            except Exception: pass
        if re.fullmatch(r"(19\d{2}|20\d{2})", s):
            return dt.date(int(s),1,1)
        return None

    def _extract_best_date(self, text: str, meta: Dict[str,Any], filepath: str) -> Optional[dt.date]:
        for k in self.DATE_FIELDS:
            if k in meta and meta[k]:
                d=self._parse_date_any(str(meta[k]))
                if d: return d
        # full dates in text
        dc=[]
        for m in re.finditer(r"\b(19\d{2}|20\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])\b", text):
            y,M,d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try: dc.append(dt.date(y,M,d))
            except Exception: pass
        if dc: return max(dc)
        # year only
        yrs=[int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
        yrs=[y for y in yrs if self.MIN_YEAR<=y<=self.MAX_YEAR]
        if yrs: return dt.date(max(yrs),1,1)
        # fallback: file mtime
        try:
            return dt.date.fromtimestamp(os.path.getmtime(filepath))
        except Exception:
            return None

    def _freshness_score(self, text: str, meta: Dict[str,Any], filepath: str, today: Optional[dt.date]=None) -> float:
        today = today or dt.date.today()
        d = self._extract_best_date(text, meta, filepath)
        if not d: return 0.6
        age_days = max((today - d).days, 0)
        return math.exp(- (math.log(2)/self.half_life_days) * age_days)

    def _merged_risk(self, text: str) -> Dict[str,float]:
        sem = self._semantic_category_scores(text)
        cnt = self._category_token_counts(text)
        cnt_score = {c: self._soft_count_to_score(cnt[c], alpha=0.3, cap=30) for c in cnt}
        gamma_sem = float(self.mix.get("gamma_sem",0.5)); gamma_cnt=float(self.mix.get("gamma_cnt",0.5))
        s = gamma_sem+gamma_cnt; gamma_sem/=s; gamma_cnt/=s
        per_cat = {c: gamma_sem*sem[c] + gamma_cnt*cnt_score.get(c,0.0) for c in sem}
        pii = self._pii_signal(text)
        cat_risk = max(per_cat.values()) if per_cat else 0.0
        overall = 0.7*cat_risk + 0.3*pii
        return {"per_category": per_cat, "pii": pii, "overall": overall}

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        if not values: return values
        lo,hi = min(values), max(values)
        if hi-lo < 1e-12: return [0.5 for _ in values]
        return [(x-lo)/(hi-lo) for x in values]

    # ---------- main API ----------
    def calculate_for_local_dataset(self, datasets_dir: str) -> List[Dict[str,Any]]:
        items=[]
        for filename in sorted(os.listdir(datasets_dir)):
            if filename.startswith(".") or filename.lower()=="running_time.txt":
                continue
            path = os.path.join(datasets_dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            try:
                if ext==".txt":
                    with open(path,"r",encoding="utf-8") as f:
                        content = f.read()
                    items.append(dict(file=filename, text=content, meta={}, path=path))
                elif ext==".csv":
                    df = pd.read_csv(path)
                    texts=[]
                    for _,row in df.iterrows():
                        fields=[str(row.get(k,"")) for k in ["question","answer","text","content","name","email","phone","id"]]
                        texts.append(" ".join(fields))
                    meta={}
                    for col in self.DATE_FIELDS:
                        if col in df.columns and pd.notnull(df[col]).any():
                            meta[col]=str(df[col].dropna().iloc[-1]); break
                    items.append(dict(file=filename, text=" ".join(texts), meta=meta, path=path))
                elif ext==".json":
                    with open(path,"r",encoding="utf-8") as f:
                        try:
                            data=json.load(f)
                        except json.JSONDecodeError:
                            f.seek(0)
                            data=[json.loads(line) for line in f if line.strip()]
                    combined = self._flatten_json_texts(data)
                    # date from first dict if present
                    meta={}
                    iters = [data] if isinstance(data, dict) else data
                    for entry in iters:
                        if isinstance(entry, dict):
                            for k in self.DATE_FIELDS:
                                if entry.get(k):
                                    meta[k]=str(entry[k]); break
                        if meta: break
                    items.append(dict(file=filename, text=combined, meta=meta, path=path))
                else:
                    continue
            except Exception as e:
                print(f"[WARN] Skipping {filename}: {e}")

        raw_risk, raw_fresh, bundle = [], [], []
        for it in items:
            text, meta, path = it["text"], it["meta"], it["path"]
            risk = self._merged_risk(text)
            fresh = self._freshness_score(text, meta, path)
            raw_risk.append(risk["overall"]); raw_fresh.append(fresh)
            bundle.append((it, risk, fresh))

        risk_std  = self._minmax(raw_risk)
        fresh_std = self._minmax(raw_fresh)

        w_r = float(self.weights.get("w_r",0.7)); w_f=float(self.weights.get("w_f",0.3))
        s=w_r+w_f; w_r/=s; w_f/=s

        results=[]
        for i, ((it, risk, fresh), Rn, Fn) in enumerate(zip(bundle, risk_std, fresh_std)):
            p = float(min(max(w_r*Rn + w_f*Fn, 0.0), 1.0))
            results.append({"file": it["file"], "probability": round(p,3)})

        return results

if __name__ == "__main__":
    start_time = time.time()
    datasets_dir = "./CalculateL_x/forgetdata"

    scorer = AutoScoringForgetProbability(
        device="cuda",
        weights=dict(w_r=0.7, w_f=0.3),
        mix=dict(gamma_sem=0.5, gamma_cnt=0.5),
        half_life_days=180,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    results = scorer.calculate_for_local_dataset(datasets_dir)

    # sort low → high probability (UnReL order)
    results = sorted(results, key=lambda r: (r["probability"], r["file"]))
    n = len(results)
    if n == 0:
        print("No files found.")
        raise SystemExit(0)

    # ----- UnReL shard assignment by quartiles -----
    q1 = max(1, int(math.ceil(n * 0.25)))
    q2 = max(q1 + 1, int(math.ceil(n * 0.50)))
    q3 = max(q2 + 1, int(math.ceil(n * 0.75)))
    if n < 4:
        q1, q2, q3 = 1, 2, 3

    def unrel_shard(i: int) -> int:
        if i < q1: return 1
        if i < q2: return 2
        if i < q3: return 3
        return 4

    import random
    indices = list(range(n))
    random.shuffle(indices)

    shard_size = math.ceil(n / 4)
    random_shards = [0] * n
    for s in range(4):
        start = s * shard_size
        end = min(start + shard_size, n)
        for idx in indices[start:end]:
            random_shards[idx] = s + 1  # S1..S4

    # ----- Create R_shard2 -----
    # Merge S1&S2 → S1,  S3&S4 → S2
    R_shard2 = []
    for s in random_shards:
        if s in (1, 2):
            R_shard2.append(1)
        else:
            R_shard2.append(2)

    # ----- Print results -----
    print(f"{'file':<20} | {'probability':>12} | {'UnReL_shard':>12} | {'R_shard4':>12} | {'R_shard2':>10}")
    print("-" * 80)
    for i, r in enumerate(results):
        print(
            f"{r['file']:<20} | "
            f"{r['probability']:>12.3f} | "
            f"{('S'+str(unrel_shard(i))):>12} | "
            f"{('S'+str(random_shards[i])):>12} | "
            f"{('S'+str(R_shard2[i])):>10}"
        )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds.")
