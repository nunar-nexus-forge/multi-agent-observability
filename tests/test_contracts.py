from __future__ import annotations

from ma_trace.contracts import Contract, state_fingerprint
from ma_trace.events import ContractEvent


def test_state_fingerprint_is_order_independent_for_dicts():
    assert state_fingerprint({"a": 1, "b": 2}) == state_fingerprint({"b": 2, "a": 1})
    assert state_fingerprint([1, 2]) != state_fingerprint([2, 1])


def test_contract_verify_against_recording():
    c = Contract.begin("agent", "move", {"pos": 0}, seed=7)
    c.set_post_state({"pos": 1})
    c.set_result("ok")
    rec = ContractEvent(agent="agent", action="move", pre_hash=c.pre_hash, post_hash=c.post_hash, seed=7)
    c.recorded = rec
    check = c.verify()
    assert check is not None and check.ok and check.pre_match and check.post_match
    ev = c.to_event()
    assert ev.verified is True and ev.result == "ok" and ev.seed == 7

    c2 = Contract.begin("agent", "move", {"pos": 0})
    c2.set_post_state({"pos": 99})
    c2.recorded = rec
    check2 = c2.verify()
    assert check2 is not None and not check2.ok and check2.pre_match and check2.post_match is False
    assert c2.expected_post_hash == c.post_hash
