.PHONY: crawl parse chunk index ask ui eval evalset ablate scenarios scenarios-check test pipeline

# 数据管线
crawl:   ## 爬取飞书帮助中心文章
	uv run python -m kb_agent.crawler

parse:   ## HTML → Markdown 清洗
	uv run python -m kb_agent.parser

chunk:   ## 标题层级分块 + 父子文档
	uv run python -m kb_agent.chunker

index:   ## 构建向量 + BM25 双路索引
	uv run python -m kb_agent.indexer

# 使用
ask:     ## 命令行问答
	uv run python -m kb_agent.cli

ui:      ## Streamlit 界面（python -m 保证项目根在模块路径中）
	uv run python -m streamlit run kb_agent/ui.py

# 评测
evalset: ## 构造评测集（需要 API key）
	uv run python -m kb_agent.eval.build_evalset

eval:    ## 跑评测并输出报告（需要 API key）
	uv run python -m kb_agent.eval.run_eval

scenarios-check: ## 校验人工场景题集与原文依据（不调用 API）
	uv run python -m kb_agent.eval.run_scenarios --validate-only

scenarios: ## 跑用户场景评测（需要 API key；保留每轮结果、节点轨迹和延迟）
	uv run python -m kb_agent.eval.run_scenarios

ablate:  ## 消融实验（检索部分无需 API key）
	uv run python -m kb_agent.eval.run_ablation

test:    ## 单元测试
	uv run pytest tests/ -v

# 一键重建全链路
pipeline: crawl parse chunk index
	@echo "数据管线完成，运行 make ask 或 make ui 开始问答"
