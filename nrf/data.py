"""Synthetic tool-calling benchmark with noisy, partially verified corrective feedback.

Each example is a natural-language request inside one API family (domain). The agent must pick one
of four tools of that family: <domain>.{search,create,update,delete}_<entity>.

Splits (group split by API family, re-drawn per seed):
  seed       small clean set (train families)            -> trusted, weight 1
  adapt      noisy corrective feedback (train families)  -> what the methods must weight
  test_seen  locked test, train families
  test_held  locked test, held-out families never seen in seed/adapt (unseen tools + unseen phrasing)

Noise types (rate parameter `noise` = target mean label-error rate of the feedback):
  uniform  every source/domain has the same error rate, random wrong label
  source   error rate depends on the source only (global annotator reliability)
  context  every source is good on most families but systematically wrong on a few
           (source x domain reliability); global source accuracy is roughly equal across sources,
           so a global source prior cannot separate good from bad feedback.

The true label of an adapt item is only *observable* when the verifier ran (`verified`). Methods
must use `observable_view()`, which hides `y_true` of unverified items.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

VERBS = ["search", "create", "update", "delete"]

# domain -> (entity, fields, values, domain-specific phrasings per verb)
DOMAINS = {
    "crm": ("customer", ["email", "phone", "address"], ["an@mail.vn", "0903 112 233", "12 Le Loi"],
            {"create": ["Onboard {N} as a new {E}."], "update": ["{N} moved; refresh the {E}'s {F} to {V}."],
             "delete": ["{N} closed their account, so remove the {E}."], "search": ["Is {N} one of our {E}s?"]}),
    "calendar": ("event", ["time", "room", "title"], ["3pm Friday", "Room B2", "Sprint review"],
                 {"create": ["Book a new {E} '{N}' for {V}."], "update": ["Reschedule the {E} {N} to {V}."],
                  "delete": ["Cancel the {E} {N}; it is not happening."], "search": ["When is the {E} {N}?"]}),
    "inventory": ("product", ["stock", "price", "supplier"], ["40 units", "120000 VND", "ACME"],
                  {"create": ["List {N} as a new {E} in the catalogue."], "update": ["Restock {N}: {F} is now {V}."],
                   "delete": ["Discontinue the {E} {N} and take it off the system."],
                   "search": ["How many units of {N} do we have?"]}),
    "hr": ("employee", ["title", "salary", "team"], ["Senior Engineer", "30M VND", "Platform"],
           {"create": ["Hire {N}: create their {E} profile."], "update": ["Promote {N}; set {F} to {V}."],
            "delete": ["Offboard {N} and delete the {E} profile."], "search": ["Who is the {E} {N} reporting to?"]}),
    "tickets": ("ticket", ["priority", "assignee", "status"], ["P1", "Minh", "blocked"],
                {"create": ["Open a {E} about {N}."], "update": ["Escalate the {E} {N}: {F} {V}."],
                 "delete": ["The {E} {N} was spam, delete it."], "search": ["Is there already a {E} about {N}?"]}),
    "files": ("document", ["owner", "folder", "title"], ["Lan", "/finance", "Q3 report"],
              {"create": ["Upload a new {E} named {N}."], "update": ["Rename the {E} {N}; {F} should be {V}."],
               "delete": ["Trash the {E} {N}."], "search": ["Where did we store the {E} {N}?"]}),
    "billing": ("invoice", ["amount", "due date", "currency"], ["2.5M VND", "30/11", "USD"],
                {"create": ["Issue an {E} to {N}."], "update": ["Adjust the {E} for {N}: {F} {V}."],
                 "delete": ["Void the {E} for {N}."], "search": ["Has {N} been sent an {E} yet?"]}),
    "shipping": ("order", ["destination", "carrier", "weight"], ["Da Nang", "GHN", "2kg"],
                 {"create": ["Place an {E} for {N}."], "update": ["Reroute {N}'s {E}: {F} {V}."],
                  "delete": ["Cancel {N}'s {E} before it ships."], "search": ["Track the {E} for {N}."]}),
    "library": ("book", ["shelf", "edition", "author"], ["C3", "2nd", "Nam Cao"],
                {"create": ["Catalogue {N} as a new {E}."], "update": ["Shelve {N} elsewhere: {F} {V}."],
                 "delete": ["Withdraw the {E} {N} from the collection."], "search": ["Is the {E} {N} available?"]}),
    "clinic": ("appointment", ["doctor", "time", "room"], ["Dr. Hoa", "9am Monday", "P204"],
               {"create": ["Schedule an {E} for {N}."], "update": ["Push {N}'s {E} to {V}."],
                "delete": ["{N} won't come; cancel the {E}."], "search": ["Does {N} have an {E} this week?"]}),
    "fleet": ("vehicle", ["driver", "plate", "mileage"], ["Tuan", "51A-123.45", "42000 km"],
              {"create": ["Register {N} as a new {E}."], "update": ["Log {N}'s new {F}: {V}."],
               "delete": ["Scrap the {E} {N} and drop its record."], "search": ["Which {E} is {N} driving?"]}),
    "school": ("student", ["class", "grade", "guardian"], ["10A2", "8.5", "Mrs. Thu"],
               {"create": ["Enrol {N} as a new {E}."], "update": ["Move {N} to {F} {V}."],
                "delete": ["{N} transferred out; delete the {E}."], "search": ["What {F} is the {E} {N} in?"]}),
}

GENERIC = {
    "search": ["Find the {E} record for {N}.", "Look up {N} in the {E} list.", "Show me the {F} of {E} {N}.",
               "Check whether {N} is already registered as a {E}.", "Pull up {N}'s {E} details.",
               "Which {E} has {F} {V}?"],
    "create": ["Add a new {E} called {N}.", "Register {N} as a new {E} with {F} {V}.",
               "Create a {E} entry for {N}.", "Set up a fresh {E} record for {N}.",
               "{N} is a new {E}; please put them in the system."],
    "update": ["Change the {F} of {E} {N} to {V}.", "Update {N}'s {F} to {V}.",
               "Correct the {F} on the {E} record for {N}.", "Set {F} = {V} for {E} {N}.",
               "Edit {E} {N} so the {F} reads {V}."],
    "delete": ["Delete the {E} record for {N}.", "Remove {N} from the {E} list.",
               "Erase {E} {N} permanently.", "Drop {N}'s {E} entry.", "Purge the {E} named {N}."],
}
# compositional requests whose surface words point to another verb
HARD = {
    "search": ["Just tell me whether {N} exists as a {E}; do not change or delete anything.",
               "Before we add anyone, look up whether {N} is already a {E}."],
    "create": ["We can't find {N} anywhere, so add them as a new {E}.",
               "Make another {E} record exactly like the old one, but for {N}."],
    "update": ["Find {E} {N} and change the {F} to {V}.", "Don't delete {N}; just fix the {F} to {V}."],
    "delete": ["Look up {N} and remove that {E} entirely.", "Instead of updating {N}, delete the {E} outright."],
}
# long-tail verb lexicon: phrase frequencies are Zipfian (order re-shuffled per seed), so rare phrasings
# need more data -> noisy feedback has real value beyond the verified subset.
LEXICON = {
    "search": ["find", "look up", "search for", "locate", "show me", "pull up", "fetch", "retrieve", "get me",
               "check on", "query", "display", "give me the details of", "tell me about", "what do we have on",
               "bring up", "view", "inspect", "dig up", "track down", "peek at", "read out", "look for",
               "see if we have", "report on"],
    "create": ["add", "create", "register", "insert", "set up", "make", "enter", "log", "record", "start",
               "enroll", "sign up", "put in", "file", "submit", "generate", "introduce", "append", "spin up",
               "draft", "establish", "open up", "launch", "write up", "bring in"],
    "update": ["update", "change", "modify", "edit", "correct", "fix", "adjust", "amend", "revise", "alter",
               "tweak", "patch", "overwrite", "refresh", "bump", "switch", "reassign", "relabel", "rewrite",
               "touch up", "rework", "fine-tune", "realign", "reconfigure", "redo"],
    "delete": ["delete", "remove", "erase", "drop", "purge", "discard", "get rid of", "wipe", "destroy",
               "eliminate", "terminate", "retire", "scrap", "trash", "clear out", "take down", "strike",
               "expunge", "nuke", "throw away", "dispose of", "kill", "wipe out", "decommission", "zap"],
}
OBJ = {"search": ["the {E} {N}", "{N}'s {E} record", "the {E} named {N}", "{E} {N}"],
       "create": ["a new {E} for {N}", "{N} as a new {E}", "a {E} record for {N}", "a fresh {E} called {N}"],
       "update": ["the {E} {N}", "{N}'s {E} record", "the {E} named {N}", "the {F} of {E} {N}"],
       "delete": ["the {E} {N}", "{N}'s {E} record", "the {E} named {N}", "{E} {N}"]}
TAIL = {"search": ["", "", " and tell me the {F}", " so I can check the {F}"],
        "create": ["", "", " with {F} {V}", " ({F}: {V})"],
        "update": [" so the {F} is {V}", " to have {F} {V}", " ({F} -> {V})", ""],
        "delete": ["", "", " from the system", " for good"]}
PRE = ["", "", "Please ", "Hey, ", "Could you ", "Can you ", "I need you to ", "Quickly ", "Urgent: ",
       "When you get a chance, "]
SUF = [".", ".", " please.", " asap.", " thanks!", " before the 5pm sync.", ", as discussed with {N2}.",
       ", then ping {N2}.", "; {N2} asked for it."]
# distractor clauses mentioning other actions (ground truth is still the main verb)
DISTRACT = [" We will update the dashboard later.", " Don't create anything else.",
            " No need to search the archive.", " Nothing should be deleted besides that."]

NAMES = ["Alice Tran", "Bao Nguyen", "Chi Le", "Dung Pham", "Hai Vo", "Khanh Do", "Lan Bui", "Minh Ho",
         "Nga Dang", "Phuc Ngo", "Quang Ly", "Thao Vu", "Uyen Mai", "Vinh Ta", "Yen Cao", "Project Atlas",
         "Order 7781", "Kim Hoang", "Son Lam", "Trang Duong"]


def tool_names(domain: str) -> list[str]:
    ent = DOMAINS[domain][0]
    return [f"{domain}.{v}_{ent}" for v in VERBS]


def _pick(rng, pool):
    return pool[rng.integers(len(pool))]


def _make_example(rng, domain, verb, lex_p, p_hard=0.1, p_domain=0.15, p_generic=0.1, p_distract=0.15):
    ent, fields, values, spec = DOMAINS[domain]
    fi = rng.integers(len(fields))
    names = rng.choice(len(NAMES), 2, replace=False)
    fmt = dict(E=ent, N=NAMES[names[0]], N2=NAMES[names[1]], F=fields[fi], V=values[fi])
    r = rng.random()
    if r < p_hard:
        text = _pick(rng, HARD[verb]).format(**fmt)
    elif r < p_hard + p_domain:
        text = _pick(rng, spec[verb]).format(**fmt)
    elif r < p_hard + p_domain + p_generic:
        text = _pick(rng, GENERIC[verb]).format(**fmt)
    else:
        vp = LEXICON[verb][rng.choice(len(LEXICON[verb]), p=lex_p[verb])]
        pre = _pick(rng, PRE)
        if pre in ("", "Urgent: "):
            vp = vp[0].upper() + vp[1:]
        text = (pre + vp + " " + _pick(rng, OBJ[verb]) + _pick(rng, TAIL[verb]) + _pick(rng, SUF)).format(**fmt)
    if rng.random() < p_distract:
        text += _pick(rng, DISTRACT)
    text = re.sub(r"\b([aA]) (?=[aeiouAEIOU])", r"\1n ", text)
    return dict(domain=domain, text=text, y_true=VERBS.index(verb))


def _sample(rng, domains, n, lex_p, **kw):
    rows = [_make_example(rng, domains[rng.integers(len(domains))], VERBS[rng.integers(4)], lex_p, **kw)
            for _ in range(n)]
    return pd.DataFrame(rows)


# organisation-specific conventions per API family (not inferable from tool names; must be learned from feedback)
POLICIES = {"none": {}, "soft_delete": {"delete": "update"}, "versioned": {"update": "create"}}


def _apply_policy(df, policy):
    surf = df.y_true.to_numpy()
    y = surf.copy()
    for i, d in enumerate(df.domain):
        rule = POLICIES[policy.get(d, "none")]
        v = VERBS[surf[i]]
        if v in rule:
            y[i] = VERBS.index(rule[v])
    df["y_surface"] = surf
    df["y_true"] = y
    df["policy_item"] = y != surf
    return df


def _zipf_probs(rng, n, s=1.1):
    p = 1.0 / np.arange(1, n + 1) ** s
    p = p[rng.permutation(n)]
    return p / p.sum()


def _cell_error_rates(rng, noise_type, noise, n_sources, train_domains, n_weak=3):
    """error[s][d] for adapt feedback."""
    err = {}
    if noise_type == "uniform":
        for s in range(n_sources):
            err[s] = {d: noise for d in train_domains}
    elif noise_type == "source":
        k = np.linspace(0.25, 1.75, n_sources)
        rng.shuffle(k)
        for s in range(n_sources):
            err[s] = {d: float(min(0.95, noise * k[s])) for d in train_domains}
    elif noise_type == "context":
        f = n_weak / len(train_domains)
        lo = noise / 3.0
        hi = min(0.95, (noise - (1 - f) * lo) / f)
        for s in range(n_sources):
            weak = set(rng.choice(train_domains, n_weak, replace=False).tolist())
            err[s] = {d: (hi if d in weak else lo) for d in train_domains}
    else:
        raise ValueError(noise_type)
    return err


def make_benchmark(seed: int, noise: float = 0.4, noise_type: str = "context", verify_rate: float = 0.25,
                   n_seed: int = 80, n_adapt: int = 640, n_test_seen: int = 300, n_test_held: int = 200,
                   n_sources: int = 6, n_heldout_domains: int = 4, systematic: float = 0.8,
                   use_policies: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    doms = list(DOMAINS)
    rng.shuffle(doms)
    held, train_d = doms[:n_heldout_domains], doms[n_heldout_domains:]
    lex_p = {v: _zipf_probs(rng, len(LEXICON[v])) for v in VERBS}

    seed_df = _sample(rng, train_d, n_seed, lex_p)
    adapt = _sample(rng, train_d, n_adapt, lex_p)
    test_seen = _sample(rng, train_d, n_test_seen, lex_p)
    test_held = _sample(rng, held, n_test_held, lex_p)
    names = list(POLICIES)
    policy = ({d: names[rng.integers(len(names))] for d in train_d} if use_policies else {})
    for df in (seed_df, adapt, test_seen, test_held):
        _apply_policy(df, policy)

    err = _cell_error_rates(rng, noise_type, noise, n_sources, train_d)
    # systematic confusion per (source, domain, true verb): the wrong label an annotator tends to give
    confuse = {(s, d, v): int(rng.choice([u for u in range(4) if u != v]))
               for s in range(n_sources) for d in train_d for v in range(4)}
    src = rng.integers(n_sources, size=n_adapt)
    y_fb = adapt.y_true.to_numpy().copy()
    for i in range(n_adapt):
        d, v, s = adapt.domain[i], adapt.y_true[i], int(src[i])
        if rng.random() < err[s][d]:
            if noise_type != "uniform" and rng.random() < systematic:
                y_fb[i] = confuse[(s, d, v)]
            else:
                y_fb[i] = int(rng.choice([u for u in range(4) if u != v]))
    adapt["source"] = src
    adapt["y_fb"] = y_fb
    adapt["verified"] = rng.random(n_adapt) < verify_rate
    test_seen["split"] = "seen"
    test_held["split"] = "held"
    test = pd.concat([test_seen, test_held], ignore_index=True)
    meta = dict(seed=seed, noise=noise, noise_type=noise_type, verify_rate=verify_rate, train_domains=train_d,
                heldout_domains=held, policy=policy, realized_noise=float((adapt.y_fb != adapt.y_true).mean()),
                verified_frac=float(adapt.verified.mean()))
    return dict(seed=seed_df, adapt=adapt, test=test, meta=meta)


def observable_view(adapt: pd.DataFrame) -> pd.DataFrame:
    """What a learner may see: no ground truth except where the verifier ran."""
    obs = adapt.drop(columns=["y_true", "y_surface", "policy_item"], errors="ignore").copy()
    obs["y_ver"] = np.where(adapt.verified, adapt.y_true, -1)
    return obs
