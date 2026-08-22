"""动态 Skills 加载与 agents 隔离测试(纯本地,不发起网络调用)。"""
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
    p = sm.prompt_for("帮我看下这家公司的研报和估值", "research")
    assert p  # 投研 Skill 命中 research agent + 关键词


def test_agents_isolation():
    sm = _sm()
    msg = "适当性和风险等级怎么评估"
    assert sm.prompt_for(msg, "compliance")        # 合规 Skill 注入 compliance agent
    assert not sm.prompt_for(msg, "research")       # research agent 拿不到 compliance Skill(隔离)
