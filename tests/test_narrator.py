from portfolio_guardrails.monitor import narrative_facts, run_monitoring
from portfolio_guardrails.narrator import LLMNarrator, TemplateNarrator, ungrounded_numbers

FACTS = {"nav": 301114.27, "vol": 0.078612, "n_trades": 6, "as_of": "2026-09-30", "factor_betas": {"MKT": 0.47}}


def test_formats_of_a_true_fact_are_accepted():
    text = "NAV is SGD 301,114 (about 301k). Volatility 7.9% or 7.86%. 6 orders. MKT beta 0.47, as of 30 Sep 2026."
    assert ungrounded_numbers(text, FACTS) == []


def test_invented_number_is_caught():
    assert ungrounded_numbers("Volatility is 7.9% and should reach 9.5% next year.", FACTS) == ["9.5"]


def test_llm_draft_using_only_facts_is_kept():
    narr = LLMNarrator(complete=lambda prompt: "Six orders are proposed; ex-ante volatility is 7.9%.")
    out = narr.narrate(FACTS)
    assert out.source == "llm" and out.rejected_numbers == []


def test_llm_draft_with_a_made_up_number_falls_back_to_template(config, market, account):
    units, cash, as_of = account
    res = run_monitoring(config, market, units, cash, as_of)
    facts = narrative_facts(res)
    narr = LLMNarrator(complete=lambda prompt: "Volatility is fine. Expect returns of 12.7% next year.")
    out = narr.narrate(facts)
    assert out.source == "template" and out.rejected_numbers == ["12.7"]
    assert out.text == TemplateNarrator().narrate(facts).text


def test_failing_model_never_blocks_the_report(config, market, account):
    def down(prompt):
        raise TimeoutError("model unavailable")

    units, cash, as_of = account
    res = run_monitoring(config, market, units, cash, as_of, narrator=LLMNarrator(complete=down))
    assert res.narrative.source == "template"


def test_template_passes_its_own_grounding_check(config, market, account):
    # If the deterministic template could not pass the check, the check would be wrong.
    units, cash, as_of = account
    res = run_monitoring(config, market, units, cash, as_of)
    facts = narrative_facts(res)
    assert ungrounded_numbers(res.narrative.text, facts) == []
