"""Dynamic Skills loading and agent isolation tests (local only, no network calls)."""
from core.skill_loader import SkillManager


def _sm():
    sm = SkillManager(root_dir="skills", max_prompt_chars=5000)
    sm.load()
    return sm


def test_three_finance_skills_load_no_errors():
    sm = _sm()
    s = sm.summary()
    assert s["count"] == 3, s
    assert not s["errors"], s["errors"]


def test_research_skill_injected_for_research_agent():
    sm = _sm()
    p = sm.prompt_for("check this company's annual report and valuation", "research")
    assert p  # research Skill matches the research agent + its keywords


def test_agents_isolation():
    sm = _sm()
    msg = "suitability and risk level"
    assert sm.prompt_for(msg, "compliance")        # compliance Skill is injected for the compliance agent
    assert not sm.prompt_for(msg, "research")      # the research agent never sees the compliance Skill


def test_skills_english_input():
    sm = _sm()
    assert sm.prompt_for("find me the research report and valuation", "research")
    assert sm.prompt_for("suitability and risk level assessment", "compliance")
    assert not sm.prompt_for("suitability and risk level assessment", "research")
