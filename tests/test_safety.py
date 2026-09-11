from backend.safety import KillSwitch


def test_killswitch_latches_and_calls_emergency_once():
    calls = []
    ks = KillSwitch(lambda: calls.append("stop"))

    assert not ks.engaged
    assert ks.engage("camera failure", "test") is True
    assert ks.engaged
    assert calls == ["stop"]

    # A second error updates status but must not repeatedly hammer emergency calls.
    assert ks.engage("second failure", "test") is False
    assert calls == ["stop"]
    assert ks.snapshot()["reason"] == "second failure"


def test_killswitch_requires_explicit_reset():
    ks = KillSwitch(lambda: None)
    ks.engage("fault")
    try:
        ks.require_safe()
        assert False, "require_safe should reject commands while engaged"
    except RuntimeError:
        pass

    ks.reset()
    ks.require_safe()
    assert not ks.engaged
