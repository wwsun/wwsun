---
title: "AI Agents 101: Everything You Need to Know About Agents"
source: https://medium.com/@sahin.samia/ai-agents-101-everything-you-need-to-know-about-agents-265fba8b9267
author:
  - "[[Sahin Ahmed]]"
  - "[[Data Scientist]]"
published: 2025-01-08
created: 2025-08-22
description: Imagine having an AI assistant that can not only answer your questions but also plan your entire vacation, negotiate deals for your business, or write and debug your code — all autonomously. This…
tags:
  - clippings
  - agent
---
## Introduction 简介


想象一下拥有一个 AI 助手，它不仅能回答你的问题，还能规划你的整个假期、为你的业务谈判交易，或者编写和调试你的代码——这一切都是自主完成的。这不是对遥远未来的憧憬；这就是当今智能体的现实。在突破性基础模型的驱动下，这些智能体正在改变我们与技术互动的方式，推动着 AI 能力的边界。

> **从本质上讲，智能体不仅仅是软件。它们感知环境，对任务进行推理，并采取行动来实现用户定义的目标。无论是处理复杂查询的客服机器人、收集和分析数据的研究助手，还是在繁忙街道上导航的自动驾驶汽车，智能体正在成为各行各业不可或缺的工具。**


智能体 AI 的兴起是一个游戏规则改变者，它能够完成以前被认为过于复杂而无法自动化的任务。但是，强大的能力伴随着巨大的复杂性。挑战不仅在于构建能够有效规划和执行行动的智能体，还在于确保它们能够反思并从其表现中学习。

在这篇博客中，我们将深入探讨 AI 代理的世界——它们是什么、为什么重要，以及它们如何工作。我们将探索驱动它们的工具、推动它们的规划，以及使它们能够随时间改进的机制。无论您是技术爱好者、开发人员还是商业领袖，这次深入了解 AI 代理的构造和潜力的旅程都会让您看到它们的变革性可能。

![](https://miro.medium.com/v2/resize:fit:640/format:webp/0*KXwH_o6ibcDz3vZl.png)

source: https://www.simform.com/blog/ai-agent/ 来源：https://www.simform.com/blog/ai-agent/

## What Are AI Agents? 什么是 AI 代理？


最简单地说，AI 代理是能够感知环境并采取行动来实现特定目标的系统。Stuart Russell 和 Peter Norvig 在他们的开创性著作《人工智能：现代方法》中，将代理定义为"任何可以被视为通过传感器感知环境并通过执行器对环境采取行动的东西"。这个定义突出了代理的双重特性——它们观察、推理和行动。

在现代 AI 背景下，这些 agents 由先进的基础模型驱动，可以处理大量数据，使它们能够在最少人工干预的情况下执行复杂任务。它们融合感知和行动的能力使其成为创建智能自主系统愿景的核心。

## Everyday Examples 日常示例


AI agents 已经成为我们日常生活的一部分，往往以我们认为理所当然的方式存在。以下是一些例子：

- **ChatGPT and Virtual Assistants:** These agents can generate text, answer questions, and even hold engaging conversations. Tools like Siri and Alexa extend this further by integrating with devices and performing actions like setting reminders or controlling smart home systems.  
	ChatGPT 和虚拟助手：这些 agents 可以生成文本、回答问题，甚至进行引人入胜的对话。Siri 和 Alexa 等工具通过与设备集成并执行设置提醒或控制智能家居系统等操作，进一步扩展了这一功能。
- **Self-Driving Cars:** Autonomous vehicles perceive their environment using sensors like cameras and LIDAR to navigate roads, avoid obstacles, and make split-second decisions.  
	自动驾驶汽车：自动驾驶车辆使用摄像头和激光雷达等传感器感知环境，在道路上导航、避开障碍物并做出瞬间决策。
- **Automated Customer Service Bots:** These agents handle customer queries, troubleshoot issues, and even recommend products, providing 24/7 support with high efficiency.  
	自动化客户服务机器人：这些代理处理客户查询、排除故障，甚至推荐产品，提供全天候 24 小时的高效支持。
- **Research and Coding Agents:** Systems like AutoGPT and SWE-agent assist in gathering information, analyzing data, and even writing or debugging code.  
	研究和编程代理：像 AutoGPT 和 SWE-agent 这样的系统协助收集信息、分析数据，甚至编写或调试代码。

These examples showcase the versatility of agents and their potential to revolutionize industries.  
这些例子展示了代理的多样性及其在各行各业带来革命性变化的潜力。

## Core Characteristics 核心特征

AI 智能体由三个关键特征定义：它们的环境、工具和行动：

**Environment 环境**  
An agent’s environment is the context or space in which it operates. This could be:  
智能体的环境是指它运行的上下文或空间。这可能是：

- A digital space like the internet or a database (e.g., for research agents).  
	数字空间，如互联网或数据库（例如，用于研究代理）。
- A physical world, such as roads for self-driving cars or a factory floor for robotic agents.  
	物理世界，如自动驾驶汽车的道路或机器人代理的工厂车间。
- A structured system like a game board or a file system.  
	结构化系统，如游戏棋盘或文件系统。

**Tools 工具**  
The tools an agent has access to determine its capabilities. For instance:  
智能体可以访问的工具决定了它的能力。例如：

- A text-based agent like ChatGPT might have tools like web browsing, a code interpreter, or APIs.  
	像 ChatGPT 这样基于文本的智能体可能拥有网络浏览、代码解释器或 API 等工具。
- A coding agent like SWE-agent uses tools to navigate repositories, search files, and edit code.  
	像 SWE-agent 这样的编程智能体使用工具来导航代码库、搜索文件和编辑代码。
- A data analytics agent might rely on SQL query generators or knowledge retrievers to interact with structured data.  
	数据分析智能体可能依赖 SQL 查询生成器或知识检索器来与结构化数据交互。

**Actions 动作**  
Actions are what agents can do based on their environment and tools. Examples include:  
动作是智能体根据其环境和工具能够执行的操作。例如包括：

- Retrieving and processing information (e.g., querying a database).  
	检索和处理信息（例如，查询数据库）。
- Interacting with external systems (e.g., sending emails or making API calls).  
	与外部系统交互（例如，发送电子邮件或进行 API 调用）。
- Modifying their environment (e.g., editing files or navigating a route).  
	修改其环境（例如，编辑文件或导航路线）。

These characteristics combine to make AI agents powerful problem solvers, capable of reasoning through tasks and executing them with a level of autonomy that’s changing the way we think about automation.  
这些特征相结合，使 AI 智能体成为强大的问题解决者，能够对任务进行推理并以一定的自主性执行任务，这正在改变我们对自动化的思考方式。

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*cDG6f6uyPIDPXUY2chexKA.png)

## Tools: Empowering AI Agents工具：赋能 AI 智能体

Tools are the cornerstone of an AI agent’s capabilities, enabling it to perceive and interact with its environment effectively. They significantly enhance the agent’s ability to process complex tasks and extend its functionality beyond the limitations of its core model. Tools can be broadly categorized into three key types: **Knowledge Augmentation**, **Capability Extension**, and **Write Actions**.  
工具是 AI 智能体能力的基石，使其能够有效地感知环境并与环境交互。它们显著增强了智能体处理复杂任务的能力，并将其功能扩展到核心模型局限性之外。工具可以大致分为三种关键类型：知识增强、能力扩展和写入操作。

## 1\. Knowledge Augmentation1. 知识增强

These tools help agents gather, retrieve, and process information, enriching their understanding of the environment. They ensure agents can access the most relevant and up-to-date data, both private and public. Examples include:  
这些工具帮助智能体收集、检索和处理信息，丰富它们对环境的理解。它们确保智能体能够访问最相关和最新的数据，包括私有和公共数据。例如：

- **Web Browsing:** Allows agents to access the internet for real-time data, preventing information staleness.  
	网页浏览：允许智能体访问互联网获取实时数据，防止信息过时。
- **Data Retrieval:** Includes APIs for fetching text, images, or structured data like SQL queries.  
	数据检索：包括用于获取文本、图像或结构化数据（如 SQL 查询）的 API。
- **APIs:** Connect the agent to external systems, such as inventory databases, Slack retrievals, or email readers.  
	API：将代理连接到外部系统，例如库存数据库、Slack 检索或电子邮件阅读器。

## 2\. Capability Extension 2. 能力扩展

These tools address inherent limitations of AI models, enabling them to perform specific tasks with greater accuracy and efficiency. Examples include:  
这些工具解决了 AI 模型的固有局限性，使其能够以更高的准确性和效率执行特定任务。示例包括：

- **Calculator:** Enhances mathematical precision, especially for complex calculations.  
	计算器：提高数学精度，特别是对于复杂计算。
- **Translator:** Facilitates multilingual communication by translating between languages the model isn’t trained for.  
	翻译器：通过在模型未经训练的语言之间进行翻译，促进多语言交流。
- **Code Interpreter:** Allows agents to write, execute, and debug code, making them powerful assistants for developers and data analysts.  
	代码解释器：允许智能体编写、执行和调试代码，使其成为开发人员和数据分析师的强大助手。

## 3\. Write Actions 3. 写入操作

Write tools empower agents to modify their environment directly, allowing for automation and real-world impact. Examples include:  
写入工具使智能体能够直接修改其环境，实现自动化并产生现实世界的影响。示例包括：

- **Database Updates:** Agents can retrieve or modify records in a database, such as updating customer accounts.  
	数据库更新：智能体可以检索或修改数据库中的记录，例如更新客户账户。
- **Email Automation:** Enables agents to send, respond to, and manage emails autonomously.  
	电子邮件自动化：使智能体能够自主发送、回复和管理电子邮件。
- **System Control:** Provides agents the ability to interact with operating systems, such as editing files or managing workflows.  
	系统控制：为智能体提供与操作系统交互的能力，例如编辑文件或管理工作流程。

## Balancing Tool Inventories平衡工具清单

While tools dramatically expand an agent’s capabilities, they also add complexity.  
虽然工具极大地扩展了智能体的能力，但也增加了复杂性。

Giving an agent too many tools can:  
给智能体过多的工具可能会：

- Overload its decision-making.  
	使其决策过载。
- Increase the likelihood of errors in tool use.  
	增加工具使用中出现错误的可能性。
- Make tool selection more difficult.  
	让工具选择变得更加困难。

Striking the right balance requires experimentation:  
找到合适的平衡需要实验：

- Perform ablation studies to assess the necessity of each tool.  
	进行消融研究来评估每个工具的必要性。
- Optimize tool descriptions and usage prompts to improve understanding.  
	优化工具描述和使用提示以提高理解度。
- Monitor tool usage patterns and refine the inventory for efficiency.  
	监控工具使用模式并优化库存以提高效率。

## The Role of Tools in AI Agent Success工具在 AI 智能体成功中的作用

The tools available to an agent define the scope of tasks it can accomplish. A well-curated tool inventory ensures the agent is equipped to excel in its environment while minimizing risks and inefficiencies. With the right tools, agents can go beyond simple queries to perform complex, multi-step tasks, driving real-world impact in diverse applications.  
智能体可用的工具定义了它能完成任务的范围。精心策划的工具库存确保智能体具备在其环境中表现优异的能力，同时最大限度地减少风险和低效率。有了合适的工具，智能体可以超越简单的查询，执行复杂的多步骤任务，在各种应用中产生现实世界的影响。

## 在 Python 中创建具有网络搜索和计算器工具的智能体示例：

```c
!pip install -qU langchain langchain_community langchain_experimental duckduckgo-search
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.llms import OpenAI
from langchain_experimental.tools import PythonREPLTool
import os

def create_search_calculator_agent(openai_api_key):
    """
    Creates a LangChain agent with web search and calculator capabilities.
    
    Args:
        openai_api_key (str): Your OpenAI API key
        
    Returns:
        Agent: Initialized LangChain agent
    """
    # Initialize the language model
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )
    
    # Initialize the tools
    search = DuckDuckGoSearchRun()
    python_repl = PythonREPLTool()
    
    tools = [
        Tool(
            name="Web Search",
            func=search.run,
            description="Useful for searching the internet to find information on recent or current events and general topics."
        ),
        Tool(
            name="Calculator",
            func=python_repl.run,
            description="Useful for performing mathematical calculations. Input should be a valid Python mathematical expression."
        )
    ]
    
    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

# Example usage
if __name__ == "__main__":
    # Replace with your OpenAI API key
    OPENAI_API_KEY = " paste your key"
    
    # Create the agent
    agent = create_search_calculator_agent(OPENAI_API_KEY)
    
    # Example queries
    queries = [
        "What is the population of Tokyo and calculate it divided by 1000?",
    ]
    
    # Test the agent
    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.run(query)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")
```
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*KJDE58mAFtYmfkust8fm-w.png)

## Planning in AI Agents AI Agent 中的规划

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*nJtGFF7bpqgTD3NuY_JfUg.png)

Planning is a fundamental capability of AI agents, enabling them to break down complex tasks into manageable actions and execute them efficiently. It involves reasoning about the goals, constraints, and resources available, then creating a roadmap (plan) to accomplish the desired task. Effective planning is critical for agents to operate autonomously and adapt to dynamic environments.  
规划是 AI Agent 的一项基础能力，使它们能够将复杂任务分解为可管理的行动并高效执行。它涉及对目标、约束和可用资源进行推理，然后创建一个路线图（计划）来完成所需的任务。有效的规划对于 Agent 自主运行和适应动态环境至关重要。

## Core Components of Planning规划的核心组件

**Plan Generation 计划生成**

- The process of creating a sequence of actions to achieve a task.  
	为完成任务而创建一系列行动的过程。
- Requires understanding the task’s **goal** (what needs to be achieved) and **constraints** (e.g., time, cost, or resource limitations).  
	需要理解任务的目标（需要实现什么）和约束条件（例如，时间、成本或资源限制）。
- Example: For a query like *“Plan a budget-friendly two-week trip to Europe,”* the agent might:  
	示例：对于"规划一次经济实惠的两周欧洲之旅"这样的查询，智能体可能会：
- Identify the user’s budget.  
	确定用户的预算。
- Suggest destinations.推荐目的地。
- Determine flight and accommodation options.  
	确定航班和住宿选择。

**Plan Validation 计划验证**

- Ensures the generated plan is feasible, logical, and within constraints.  
	确保生成的计划是可行的、合乎逻辑的，并且在约束条件内。
- Validation can involve:验证可能涉及：
- **Heuristics:** Simple rules to eliminate invalid plans (e.g., rejecting plans with more steps than the agent can execute).  
	启发式规则：简单的规则来排除无效计划（例如，拒绝步骤数量超过代理执行能力的计划）。
- **AI Evaluators:** Using another model to assess the plan’s quality.  
	AI 评估器：使用另一个模型来评估计划的质量。
- Example: A travel plan that exceeds the user’s budget would be flagged and revised.  
	示例：超出用户预算的旅行计划会被标记并修改。

**Execution 执行**

- Involves performing the actions outlined in the plan.  
	涉及执行计划中概述的操作。
- Actions can involve:操作可能包括：
- Using tools (e.g., APIs, databases, or calculators).  
	使用工具（例如，API、数据库或计算器）。
- Gathering feedback from the environment (e.g., web search results or code execution outputs).  
	从环境中收集反馈（例如，网络搜索结果或代码执行输出）。
- Example: After validating a plan, the agent books flights, reserves hotels, and sends an itinerary.  
	示例：在验证计划后，智能体预订航班、预订酒店并发送行程安排。

**Reflection and Error Correction  
反思和错误修正**

- Post-action evaluation to determine if the task was successfully completed.  
	行动后评估以确定任务是否成功完成。
- If the task fails, the agent identifies errors, updates its plan, and retries.  
	如果任务失败，代理会识别错误，更新其计划，并重新尝试。
- Example: If a booking tool fails to process a request, the agent retries with alternative tools or methods.  
	示例：如果预订工具无法处理请求，代理会使用替代工具或方法重新尝试。

## Approaches to Planning 规划方法

**Hierarchical Planning 分层规划**

- Plans are created in layers, starting with high-level goals and breaking them into smaller, actionable steps.  
	计划是分层创建的，从高层次目标开始，然后分解为更小的可操作步骤。
- Example:示例：
- High-level: “Plan a trip to Europe.”  
	高层次："计划一次欧洲之旅。"
- Subtasks: Book flights → Reserve hotels → Create a daily itinerary.  
	子任务：预订航班 → 预订酒店 → 制定每日行程。

**Step-by-Step Planning 逐步规划**

- The agent reasons through each step sequentially, deciding the next action based on the previous step’s outcome.  
	智能体按顺序推理每个步骤，根据前一步的结果决定下一个行动。
- Often used with techniques like **chain-of-thought prompting** to maintain focus on the task.  
	通常与链式思维提示等技术结合使用，以保持对任务的专注。

**Parallel Planning 并行规划**

- Allows the agent to execute multiple steps simultaneously to save time.  
	允许代理同时执行多个步骤以节省时间。
- Example: Searching for hotels and flights at the same time.  
	示例：同时搜索酒店和航班。

**Dynamic Planning 动态规划**

- Plans adapt in real-time based on new information or changes in the environment.  
	计划根据新信息或环境变化实时调整。
- Example: If an API fails during a task, the agent updates its plan to use an alternative method.  
	示例：如果在执行任务期间 API 失败，智能体会更新其计划以使用替代方法。

## Challenges in Planning 规划中的挑战

**Complexity of Multi-Step Tasks  
多步骤任务的复杂性**

- Accuracy decreases as the number of steps increases due to error propagation.  
	由于错误传播，准确性随着步骤数量的增加而降低。
- Example: If an agent’s accuracy is 95% per step, after 10 steps, overall accuracy might drop to ~60%.  
	例如：如果智能体每步的准确率为 95%，经过 10 步后，整体准确率可能下降至约 60%。

**Goal Misalignment 目标不一致**

- The agent might generate a plan that doesn’t meet the user’s goals or violates constraints.  
	智能体可能生成不符合用户目标或违反约束条件的计划。
- Example: Planning a luxury trip when the user specified a budget-friendly option.  
	例如：当用户明确要求经济实惠的选择时，却规划了奢华旅行。

**Tool Dependency 工具依赖**

- Plans rely heavily on tools, and any failure in tool usage can derail the task.  
	计划严重依赖工具，任何工具使用失败都可能导致任务偏离轨道。
- Example: Using an invalid API call or passing incorrect parameters to a tool.  
	示例：使用无效的 API 调用或向工具传递错误参数。

**Resource Efficiency 资源效率**

- Plans with unnecessary steps waste resources like API calls, compute time, and cost.  
	包含不必要步骤的计划会浪费资源，如 API 调用、计算时间和成本。

## Strategies for Better Planning更好规划的策略

**Decoupling Planning from Execution  
将规划与执行解耦**

- First, generate a plan.首先，生成一个计划。
- Validate the plan.验证计划。
- Execute the validated plan.  
	执行已验证的计划。

**Intent Classification 意图分类**

- Understand the user’s intent to create more accurate and relevant plans.  
	理解用户的意图以创建更准确和相关的计划。
- Example: Distinguishing between a query for “buying shoes online” versus “researching shoe trends.”  
	示例：区分"在线购买鞋子"与"研究鞋类趋势"的查询。

**Reflection-Driven Iteration  
反思驱动迭代**

- Use self-reflection prompts (e.g., “What could go wrong?”) to refine plans before execution.  
	使用自我反思提示（例如，"可能出现什么问题？"）在执行前完善计划。

**Multi-Agent Collaboration  
多智能体协作**

- Assign different roles to specialized agents (e.g., one for planning, another for validation) for more robust outcomes.  
	为不同的专门智能体分配不同角色（例如，一个负责规划，另一个负责验证），以获得更稳健的结果。

## Example: AI Agent Planning示例：AI 智能体规划

**Task:** Find and summarize the top research papers on AI for the last year.  
任务：查找并总结去年关于 AI 的顶级研究论文。

**Plan:计划：**

1. Use the **web search tool** to retrieve the top AI conferences.  
	使用网络搜索工具检索顶级 AI 会议。
2. Query **academic databases** for papers presented at these conferences.  
	查询学术数据库中在这些会议上发表的论文。
3. Use an **LLM tool** to summarize the abstracts of the top 5 papers.  
	使用 LLM 工具总结前 5 篇论文的摘要。
4. Compile and return the summary to the user.  
	编译并向用户返回摘要。

**Execution:执行：**

- **Step 1:** Retrieve conference names.  
	步骤 1：检索会议名称。
- **Step 2:** Search papers from each conference.  
	步骤 2：从每个会议中搜索论文。
- **Step 3:** Summarize papers.  
	步骤 3：总结论文。
- **Step 4:** Return results.步骤 4：返回结果。

**Reflection:反思：**  
If the retrieved papers are outdated, refine the search criteria and retry.  
如果检索到的论文已过时，请优化搜索标准并重试。

## The Future of AI Agent PlanningAI 智能体规划的未来

- **Integration with Memory Systems:** Enhanced planning by retaining context and past decisions.  
	与记忆系统集成：通过保留上下文和过往决策来增强规划能力。
- **Tool-Aware Planning:** Improved capabilities with deeper knowledge of tool functionalities.  
	工具感知规划：通过深入了解工具功能来提升能力。
- **Human-AI Collaboration:** Hybrid workflows where humans validate or enhance plans.  
	人机协作：人类验证或增强计划的混合工作流程。

Planning is the backbone of intelligent AI agents, transforming them from reactive systems into proactive problem-solvers capable of tackling complex, real-world tasks.  
规划是智能 AI 代理的支柱，将其从被动系统转变为能够处理复杂现实任务的主动问题解决者。

## Python example of creating an Agent with planning capability:创建具有规划能力的 Agent 的 Python 示例：

Code source:[https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb)  
代码来源：https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb

```c
%%capture --no-stderr
%pip install --quiet -U langgraph langchain-community langchain-openai tavily-python
import getpass
import os

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
#defining the tools
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3)]

#define the execution agent

from langchain import hub
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("ih/ih-react-agent-executor")
prompt.pretty_print()

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

#define the agent state
import operator
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
#define the planner
from pydantic import BaseModel, Field
class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Plan)

#define replanner 
from typing import Union

class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).with_structured_output(Act)

#create the graph
from typing import Literal
from langgraph.graph import END

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
from IPython.display import Image, display

display(Image(app.get_graph(xray=True).draw_mermaid_png()))
```
![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*adjw0P_XGhZjLUiVZdKKvg.png)

```c
config = {"recursion_limit": 10}
inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*97s215n1HU0TTbq_sbgELQ.png)

## Reflection: Learning from Mistakes in AI Agents反思：从 AI Agent 的错误中学习

Reflection is a critical process in AI agents, enabling them to learn from mistakes, adapt their strategies, and improve performance over time. By analyzing their actions and outcomes, agents can identify errors, refine their plans, and ensure that tasks are completed successfully. Reflection also helps agents become more resilient to failures and better equipped to handle complex, multi-step tasks.  
反思是 AI Agent 中的一个关键过程，使它们能够从错误中学习、调整策略并随着时间的推移提高性能。通过分析它们的行为和结果，Agent 可以识别错误、完善计划，并确保任务能够成功完成。反思还有助于 Agent 对失败变得更有韧性，并更好地处理复杂的多步骤任务。

## What Is Reflection in AI Agents?AI Agents 中的反思是什么？

Reflection is the process where an agent evaluates its own performance at various stages of task execution. It involves:  
反思是指智能体在任务执行的各个阶段评估自身表现的过程。它包括：

- Assessing the correctness of actions taken.  
	评估所采取行动的正确性。
- Verifying whether goals are being achieved.  
	验证目标是否正在实现。
- Identifying and correcting errors.  
	识别和纠正错误。
- Iterating to refine future actions.  
	迭代以改进未来的行动。

Reflection is often interwoven with error correction, creating a feedback loop where the agent learns and improves with each iteration.  
反思通常与错误纠正交织在一起，创建一个反馈循环，使智能体在每次迭代中学习和改进。

## Key Points in the Reflection Process反思过程中的关键要点

Reflection can occur at multiple stages of an agent’s workflow:  
反思可以在智能体工作流程的多个阶段发生：

**Before Task Execution 任务执行前**

- Evaluate the feasibility of a generated plan.  
	评估生成计划的可行性。
- Identify potential risks or limitations.  
	识别潜在的风险或限制。
- Example: Before executing a plan to book a trip, the agent checks if the budget constraints are realistic.  
	示例：在执行预订旅行计划之前，智能体会检查预算约束是否现实。

**During Execution 执行过程中**

- Monitor the outcomes of each action to ensure they align with the plan.  
	监控每个操作的结果，确保它们与计划保持一致。
- Identify deviations or failures early.  
	及早识别偏差或故障。
- Example: If a database query returns no results, the agent reflects on whether the query parameters were correct.  
	例如：如果数据库查询没有返回结果，智能体会反思查询参数是否正确。

**After Task Completion 任务完成后**

- Determine if the task was successfully completed.  
	确定任务是否成功完成。
- Analyze any failures and their causes.  
	分析任何失败及其原因。
- Example: After completing a coding task, the agent checks whether the generated code passes all test cases.  
	示例：完成编码任务后，代理会检查生成的代码是否通过所有测试用例。

## Mechanisms for Reflection反思机制

1. **Self-Critique 自我批评**
- The agent critiques its own actions using prompts or heuristics.  
	代理使用提示或启发式方法对自己的行为进行批判。
- Example: After generating an output, the agent asks, “Did this result achieve the goal? If not, why?”  
	示例：生成输出后，智能体会询问："这个结果是否达到了目标？如果没有，为什么？"

**Error Analysis 错误分析**

- The agent identifies specific points of failure and their underlying causes.  
	智能体识别具体的失败点及其根本原因。
- Example: For a failed SQL query, the agent reflects on whether the table names or column names were incorrect.  
	示例：对于失败的 SQL 查询，智能体会反思表名或列名是否不正确。

**Replanning 重新规划**

- The agent adjusts its plan based on identified errors and retries the task.  
	智能体根据识别出的错误调整其计划并重新尝试任务。
- Example: If an API call fails due to a missing parameter, the agent modifies the call and tries again.  
	示例：如果 API 调用由于缺少参数而失败，智能体会修改调用并再次尝试。

**External Evaluation 外部评估**

- Another agent or model evaluates the output, providing feedback for improvement.  
	另一个智能体或模型评估输出，提供改进反馈。
- Example: A coding assistant’s output is evaluated by a separate testing agent.  
	示例：编程助手的输出由独立的测试智能体进行评估。

## Reflection Frameworks 反思框架

1. **ReAct Framework (Reasoning + Acting)  
	ReAct 框架（推理 + 行动）**
![](https://miro.medium.com/v2/resize:fit:640/format:webp/0*rqTV7b6v08p0E3yF.png)

Link to paper:[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)  
论文链接：https://arxiv.org/abs/2210.03629

- Combines reasoning (planning and reflection) with actions at each step.  
	将推理（规划和反思）与每步的行动相结合。
- Encourages agents to alternate between planning, executing, and reflecting iteratively.  
	鼓励智能体在规划、执行和反思之间迭代交替。
- Example:示例：
```c
Thought: I need to find the top news articles about AI. 
Action: Perform a web search. Observation: The search returned irrelevant results. Thought: The query needs to be refined for better results.
```

**Reflexion Framework Reflexion 框架**

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*y3LUfe0loSWYWB_gtk7B0A.png)

Link to paper: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)  
论文链接：https://arxiv.org/abs/2303.11366

- Separates reflection into two components:  
	将反思分为两个组成部分：
- **Evaluator:** Assesses whether the task was completed successfully.  
	评估器：评估任务是否成功完成。
- **Self-Reflection Module:** Identifies and analyzes mistakes, then provides suggestions for improvement.  
	自我反思模块：识别和分析错误，然后提供改进建议。
- Example: If the agent fails to retrieve relevant data, it reflects that the search term was too generic and revises the query.  
	示例：如果代理无法检索到相关数据，它会反思搜索词过于宽泛，并修订查询。

## Benefits of Reflection 反思的好处

**Improved Accuracy 提高准确性**

- By analyzing errors, agents can refine their actions and reduce mistakes in future iterations.  
	通过分析错误，智能体可以改进其行动并减少未来迭代中的错误。

**Resilience to Failure 故障恢复能力**

- Reflection allows agents to recover from unexpected failures or incorrect assumptions.  
	反思允许智能体从意外故障或错误假设中恢复。

**Better Resource Efficiency  
更好的资源效率**

- Detecting errors early in the process prevents the agent from wasting time or resources on flawed plans.  
	在流程早期检测错误可防止智能体在有缺陷的计划上浪费时间或资源。

**Continuous Learning 持续学习**

- Reflection creates a loop where agents learn from their experiences and improve over time.  
	反思创建了一个循环，让智能体从经验中学习并随着时间推移不断改进。

## Challenges in Reflection 反思中的挑战

**Latency and Cost 延迟和成本**

- Generating reflective insights increases token usage and response time, especially in multi-step tasks.  
	生成反思性洞察会增加 token 使用量和响应时间，特别是在多步骤任务中。
- Mitigation: Use reflection selectively, focusing on critical tasks or steps.  
	缓解措施：有选择性地使用反思，专注于关键任务或步骤。

**Complexity of Multi-Step Tasks  
多步骤任务的复杂性**

- Errors in earlier steps can cascade, making it harder to pinpoint the root cause of failure.  
	早期步骤中的错误可能会级联传播，使得难以准确定位失败的根本原因。
- Mitigation: Introduce intermediate checkpoints for reflection.  
	缓解措施：引入中间检查点进行反思。

**Reflection Quality 反思质量**

- Agents may generate overly generic or unhelpful reflections.  
	智能体可能生成过于泛化或无用的反思。
- Mitigation: Enhance reflection prompts with clear instructions and examples.  
	缓解措施：通过清晰的指令和示例来增强反思提示。

## The Future of Reflection in AI AgentsAI Agents 中反思的未来

- **Enhanced Self-Critique:** Advanced models that can critique their actions with greater depth and specificity.  
	增强的自我批评：能够更深入、更具体地批评自身行为的高级模型。
- **Memory Integration:** Reflection systems that retain knowledge of past mistakes to prevent recurrence.  
	记忆集成：能够保留过往错误知识以防止重复发生的反思系统。
- **Multi-Agent Collaboration:** Agents evaluating each other’s actions to increase robustness.  
	多智能体协作：智能体相互评估彼此的行为以提高鲁棒性。

Reflection is a cornerstone of effective AI agents, enabling them to learn, adapt, and excel in complex environments. By systematically evaluating their actions and outcomes, agents can achieve higher accuracy, efficiency, and reliability in their tasks.  
反思是高效 AI 智能体的基石，使它们能够在复杂环境中学习、适应和表现出色。通过系统性地评估自身的行为和结果，智能体可以在其任务中实现更高的准确性、效率和可靠性。

## Failure Modes in AI AgentsAI 智能体的失败模式

AI agents, while powerful, are not immune to errors. Failures can occur at various stages of their operation, often due to the complexity of planning, execution, or tool usage. Understanding and addressing these failure modes is critical for building robust and reliable agents.  
AI 智能体虽然强大，但并非不会出错。失效可能发生在其操作的各个阶段，通常是由于规划、执行或工具使用的复杂性所致。理解和解决这些失效模式对于构建鲁棒可靠的智能体至关重要。

## 1\. Planning Failures 1. 规划失败

Planning is a challenging task, especially for multi-step workflows. Common failure modes in planning include:  
规划是一项具有挑战性的任务，尤其是对于多步骤工作流程。规划中常见的失败模式包括：

**Using Invalid Tools or Parameters:  
使用无效的工具或参数：**

- The agent may generate a plan that includes tools not available in its inventory or call tools with incorrect or missing parameters.  
	Agent 可能会生成一个包含其工具库中不可用工具的计划，或使用不正确或缺失的参数调用工具。
- **Example:** Calling a function with the wrong argument types (e.g., passing a string where a number is expected).  
	示例：使用错误的参数类型调用函数（例如，在需要数字的地方传递字符串）。

**Failing to Achieve Goals or Adhere to Constraints:  
未能实现目标或遵守约束：**

- Plans might not accomplish the user’s goals or violate specified constraints.  
	计划可能无法完成用户的目标或违反指定的约束。
- **Example:** Planning a trip outside a given budget or booking a flight for the wrong destination.  
	示例：规划超出给定预算的旅行或预订到错误目的地的航班。

**Misjudging Task Completion:  
错误判断任务完成：**

- The agent might incorrectly assume that a task has been completed when it has not.  
	智能体可能会错误地认为某项任务已经完成，而实际上并没有完成。
- **Example:** Assigning hotel rooms to fewer people than required but considering the task finished.  
	示例：为少于所需人数的人员分配酒店房间，但认为任务已经完成。

## 2.Tool Failures 2.工具故障

Agents often depend on external tools, and any errors in tool usage can lead to failures. These include:  
代理通常依赖外部工具，工具使用中的任何错误都可能导致失败。这些包括：

**Incorrect Outputs:错误的输出：**

- Tools may provide incorrect or incomplete results due to bugs or misconfiguration.  
	工具可能因为错误或配置不当而提供不正确或不完整的结果。
- **Example:** A SQL query generator returning a syntactically incorrect query.  
	示例：SQL 查询生成器返回语法错误的查询。

**Translation Errors:翻译错误：**

- If a translator module is used to map high-level plans into tool-specific actions, it can introduce errors.  
	如果使用翻译器模块将高级计划映射为特定工具的操作，可能会引入错误。
- **Example:** Mapping a plan step to an incorrect API endpoint.  
	示例：将计划步骤映射到错误的 API 端点。

## 3\. Efficiency Issues 3. 效率问题

Even if the agent accomplishes its task, it may do so inefficiently, leading to wasted resources and higher costs.  
即使代理完成了任务，也可能以低效的方式完成，导致资源浪费和成本增加。

**Excessive Steps:步骤过多：**

- The agent may take unnecessary steps to achieve the goal, increasing time and cost.  
	代理可能采取不必要的步骤来实现目标，增加时间和成本。
- **Example:** Performing redundant web searches or making multiple API calls for the same data.  
	示例：执行冗余的网络搜索或为相同数据进行多次 API 调用。

**High Latency:高延迟：**

- Tasks might take longer to execute than expected, reducing the agent’s utility in time-sensitive scenarios.  
	任务的执行时间可能比预期更长，降低了智能代理在时间敏感场景中的实用性。
- **Example:** A customer support agent taking too long to respond to a query.  
	示例：客户支持代理对查询响应时间过长。

**Cost Overruns:成本超支：**

- Using expensive tools or making inefficient API calls can lead to higher operational costs.  
	使用昂贵的工具或进行低效的 API 调用可能导致更高的运营成本。
- **Example:** Frequent use of an expensive language model API for trivial tasks.  
	示例：频繁使用昂贵的语言模型 API 来处理琐碎任务。

## Evaluation Metrics for failuers失败评估指标

To detect and address failures, it’s important to evaluate agents using specific metrics:  
为了检测和解决失败问题，使用特定指标评估智能体非常重要：

**Validity of Plans and Tool Calls:  
计划和工具调用的有效性：**

- Check whether the agent’s plans are executable and its tool calls are valid.  
	检查代理的计划是否可执行，其工具调用是否有效。
- **Metric:** Percentage of valid plans and tool calls.  
	指标：有效计划和工具调用的百分比。

**Frequency of Invalid or Inefficient Actions:  
无效或低效操作的频率：**

- Measure how often the agent selects the wrong tool, uses invalid parameters, or takes unnecessary steps.  
	衡量智能体选择错误工具、使用无效参数或执行不必要步骤的频率。
- **Metric:** Count of invalid tool calls or redundant steps per task.  
	指标：每项任务中无效工具调用或冗余步骤的数量。

**Analysis of Failure Patterns:  
失败模式分析：**

- Identify recurring issues in specific types of tasks or with particular tools.  
	识别特定类型任务或特定工具中反复出现的问题。
- **Metric:** Categorization and frequency of common failure modes.  
	指标：常见故障模式的分类和频率。

**Tool Effectiveness:工具有效性：**

- Evaluate how well each tool contributes to task success.  
	评估每个工具对任务成功的贡献程度。
- **Metric:** Success rate of actions involving specific tools.  
	指标：涉及特定工具的操作成功率。

## Example of Failure Analysis故障分析示例

**Task:** Retrieve the top-selling products for the last quarter and generate a sales report.  
任务：检索上个季度的热销产品并生成销售报告。

**Failure Scenario:失败场景：**

**Planning Failure:规划失败：**

- The agent generates a plan to use a “fetch\_data” tool, but this tool isn’t in its inventory.  
	智能体生成了一个使用"fetch\_data"工具的计划，但该工具不在其工具库中。
- Result: The plan cannot be executed.  
	结果：计划无法执行。

**Tool Failure:工具故障：**

- The agent uses a database query tool, but the query contains syntax errors.  
	智能体使用了数据库查询工具，但查询包含语法错误。
- Result: The database returns an error.  
	结果：数据库返回错误。

**Efficiency Issue:效率问题：**

- The agent performs three redundant searches to fetch the same data.  
	代理执行三次冗余搜索来获取相同的数据。
- Result: Increased latency and cost.  
	结果：增加了延迟和成本。

## Strategies to Address Failures解决故障的策略

**Improving Prompts and Plans:  
改进提示词和计划：**

- Use better examples and more detailed instructions to guide the agent during planning.  
	使用更好的示例和更详细的指令来指导智能体进行规划。

**Enhancing Tool Descriptions:  
增强工具描述：**

- Provide clear documentation for tools, including their inputs, outputs, and limitations.  
	为工具提供清晰的文档，包括它们的输入、输出和限制。

**Validation Checks:验证检查：**

- Introduce validation steps for plans and tool calls before execution.  
	在执行前为计划和工具调用引入验证步骤。

**Monitoring and Logging:监控和日志记录：**

- Record all actions, tool calls, and outputs for analysis and debugging.  
	记录所有操作、工具调用和输出，用于分析和调试。

**Reflection and Correction:  
反思和纠正：**

- Use reflection mechanisms to identify and correct errors dynamically during execution.  
	使用反思机制在执行过程中动态识别和纠正错误。

Failures in AI agents can stem from planning errors, tool usage issues, or inefficiencies. By identifying and addressing these failure modes through robust evaluation and error correction mechanisms, developers can enhance the reliability and performance of agents, ensuring they deliver value in real-world applications.  
AI 智能体的失败可能源于规划错误、工具使用问题或效率低下。通过鲁棒的评估和错误纠正机制识别和解决这些失败模式，开发者可以增强智能体的可靠性和性能，确保它们在实际应用中提供价值。

## Security Considerations in AI AgentsAI 智能体的安全考虑

AI agents are powerful tools capable of performing complex tasks autonomously. However, their capabilities also introduce significant security risks. Addressing these risks is critical to ensuring the safe and reliable operation of AI agents in real-world environments.  
AI 智能体是能够自主执行复杂任务的强大工具。然而，它们的能力也带来了重大的安全风险。解决这些风险对于确保 AI 智能体在现实环境中的安全可靠运行至关重要。

## Key Security Risks 主要安全风险

**1\. Malicious Actions 1\. 恶意行为**

AI agents with access to powerful tools and sensitive data can be exploited for malicious purposes:  
具有强大工具访问权限和敏感数据访问权限的 AI 智能体可能被恶意利用：

- **Unauthorized Data Access:** Agents could inadvertently or maliciously access and expose private or sensitive data.  
	未经授权的数据访问：智能体可能无意或恶意地访问和暴露私人或敏感数据。
- **Harmful Outputs:** Misuse of generative capabilities could result in misinformation, biased outputs, or offensive content.  
	有害输出：滥用生成能力可能导致错误信息、有偏见的输出或攻击性内容。
- **Automation Risks:** Agents executing write actions, such as database modifications or file edits, could be manipulated to delete critical information or make harmful changes.  
	自动化风险：执行写入操作的智能体，如数据库修改或文件编辑，可能被操纵来删除关键信息或进行有害更改。
- **Code Injection Attacks:** If agents have access to code execution tools, attackers could inject malicious code for execution.  
	代码注入攻击：如果智能体能够访问代码执行工具，攻击者可能会注入恶意代码并执行。

**2\. Vulnerabilities to Manipulation  
2\. 易受操纵的漏洞**

Agents can be manipulated into performing unintended actions through adversarial attacks:  
智能体可能通过对抗性攻击被操纵执行非预期的行为：

- **Prompt Injection:** Malicious actors craft inputs that manipulate the agent’s behavior, leading to unintended or harmful outcomes.  
	提示词注入：恶意行为者精心设计输入来操纵智能体的行为，导致非预期或有害的结果。
- **Data Poisoning:** Feeding misleading or malicious data during training or fine-tuning can bias agent behavior.  
	数据投毒：在训练或微调过程中提供误导性或恶意数据可能会使智能体行为产生偏差。
- **Social Engineering:** Crafting deceptive inputs to trick agents into revealing sensitive information or taking unauthorized actions.  
	社会工程：设计欺骗性输入来诱使代理泄露敏感信息或执行未授权操作。

**3\. Over-Reliance on External Tools  
3\. 过度依赖外部工具**

Agents relying on external tools and APIs introduce additional attack surfaces:  
依赖外部工具和 API 的代理会引入额外的攻击面：

- **API Exploits:** Unauthorized or malformed API calls could compromise system security.  
	API 漏洞利用：未授权或格式错误的 API 调用可能会危及系统安全。
- **Third-Party Vulnerabilities:** If external tools or APIs are compromised, the agent may unintentionally propagate the attack.  
	第三方漏洞：如果外部工具或 API 被攻破，代理可能会无意中传播攻击。

## Mitigation Strategies 缓解策略

**1\. Defensive Prompt Engineering  
1\. 防御性提示工程**

- Craft prompts that explicitly limit the agent’s scope of operation and ensure safe behavior:  
	制作明确限制代理操作范围并确保安全行为的提示词：
- **Constraints:** Include instructions to avoid specific sensitive actions (e.g., “Do not perform write actions without explicit approval”).  
	约束条件：包含避免特定敏感操作的指令（例如，"未经明确批准不得执行写入操作"）。
- **Validation Prompts:** Ask the agent to validate its actions before executing them (e.g., “Is this action safe and aligned with user intent?”).  
	验证提示：要求代理在执行操作前验证其行为（例如，"此操作是否安全且符合用户意图？"）。
- **Layered Prompts:** Use structured prompts that introduce multiple layers of checks and confirmations.  
	分层提示：使用结构化提示来引入多层检查和确认机制。

**2\. Access Control 2\. 访问控制**

- Implement strict permissions to control what tools and data the agent can access:  
	实施严格的权限控制，以管控代理可以访问的工具和数据：
- **Role-Based Access:** Assign specific permissions to the agent based on its task.  
	基于角色的访问控制：根据代理的任务为其分配特定权限。
- **Tool Inventory Restriction:** Limit the number of tools the agent has access to, reducing potential misuse.  
	工具清单限制：限制代理可访问的工具数量，减少潜在的误用。
- **Environment Sandboxing:** Isolate the agent’s operations in a controlled sandbox to prevent unauthorized system-level actions.  
	环境沙箱化：将智能体的操作隔离在受控沙箱中，以防止未经授权的系统级操作。

**3\. Input and Output Validation  
3\. 输入输出验证**

- **Sanitize Inputs:** Ensure user inputs are properly sanitized to prevent injection attacks or manipulation.  
	输入清理：确保用户输入得到适当清理，以防止注射攻击或操纵。
- **Validate Outputs:** Review agent-generated actions or responses for compliance with expected behavior.  
	输出验证：检查智能体生成的操作或响应是否符合预期行为。

**4\. Logging and Monitoring  
4\. 日志记录和监控**

- Maintain detailed logs of all agent actions, tool calls, and outputs:  
	维护所有智能体操作、工具调用和输出的详细日志：
- **Real-Time Monitoring:** Use dashboards to track the agent’s behavior and detect anomalies.  
	实时监控：使用仪表板跟踪智能体的行为并检测异常。
- **Audit Trails:** Keep records for post-incident analysis and accountability.  
	审计追踪：保留记录用于事后分析和问责。

**5\. Human-in-the-Loop Oversight  
5\. 人在回路监督**

- Integrate human review for critical or high-risk actions:  
	为关键或高风险操作集成人工审查：
- **Approval Gates:** Require explicit human approval for sensitive tasks, such as financial transactions or database modifications.  
	审批关卡：对敏感任务（如金融交易或数据库修改）要求明确的人工审批。
- **Fallback Mechanisms:** Allow humans to intervene and correct agent actions in real time.  
	回退机制：允许人工实时干预和纠正智能体操作。

**6\. Model and Tool Hardening  
6\. 模型和工具加固**

- Regularly update and fine-tune the agent model to improve robustness against adversarial inputs.  
	定期更新和微调 AI 智能体模型，以提高对对抗性输入的鲁棒性。
- Conduct security testing for external tools and APIs to minimize vulnerabilities.  
	对外部工具和 API 进行安全测试，以最大限度地减少漏洞。

## Example: Securing an AI Agent示例：保护 AI 智能体

**Scenario:** A customer support agent capable of accessing user account data and resolving issues autonomously.  
场景：能够访问用户账户数据并自主解决问题的客户支持代理。

**Risks:风险：**

1. Malicious users attempting to access other customers’ data.  
	恶意用户试图访问其他客户的数据。
2. Prompt injection to trigger unauthorized actions.  
	提示注入以触发未经授权的操作。

**Mitigation Measures:缓解措施：**

Restrict database access to read-only for non-administrative tasks.  
对非管理任务限制数据库访问为只读。

Use defensive prompts like:  
使用防御性提示，例如：

- “Verify user authentication before retrieving account details.”  
	"在检索账户详细信息之前验证用户认证。"
- “Do not reveal sensitive data like passwords or full payment information.”  
	"不要透露密码或完整付款信息等敏感数据。"

Log all actions, such as data retrievals and responses, for monitoring.  
记录所有操作，如数据检索和响应，以便监控。

Require human approval for actions involving refunds or account deletions.  
对于涉及退款或账户删除的操作需要人工批准。

## Commenly used agentic frameworks:常用的智能体框架：

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*5ZH6vIsIp7tfrn3bGA1k-A.png)

## Conclusion 结论

AI agents represent a transformative step in the evolution of artificial intelligence, combining powerful reasoning, planning, and action capabilities to autonomously solve complex problems. From automating routine tasks to tackling sophisticated workflows, these agents are poised to revolutionize industries, drive productivity, and unlock new possibilities across diverse domains.  
AI 智能体代表了人工智能演进中的一个变革性步骤，它结合了强大的推理、规划和行动能力，能够自主解决复杂问题。从自动化日常任务到处理复杂的工作流程，这些智能体有望彻底改变各个行业，推动生产力提升，并在不同领域释放新的可能性。

However, with great power comes great responsibility. The development and deployment of AI agents require a nuanced understanding of their capabilities, limitations, and potential risks. Planning, tool selection, and reflection are critical components for building effective agents, while robust security measures ensure that these systems operate safely and ethically.  
然而，能力越大，责任越大。AI 智能体的开发和部署需要对其能力、局限性和潜在风险有细致入微的理解。规划、工具选择和反思是构建有效智能体的关键组成部分，而强有力的安全措施则确保这些系统能够安全、合乎道德地运行。

As the field of agentic AI continues to evolve, embracing collaboration between human and machine will be key to leveraging their full potential. Whether you’re a developer, researcher, or business leader, investing in the understanding and integration of AI agents today can pave the way for a smarter, more efficient tomorrow.  
随着智能体 AI 领域的持续发展，拥抱人机协作将是充分发挥其潜力的关键。无论您是开发者、研究人员还是企业领导者，今天在 AI 智能体的理解和集成方面进行投资，都能为更智能、更高效的明天铺平道路。

The possibilities are vast, but so is the responsibility to build agents that are not only powerful but also safe, transparent, and aligned with human values. By combining innovation with accountability, we can harness the true potential of AI agents to create a better future.  
可能性是巨大的，但构建不仅强大而且安全、透明、与人类价值观一致的智能体的责任同样重大。通过将创新与责任相结合，我们可以发挥 AI 智能体的真正潜力，创造更美好的未来。

## Useful references to read further:进一步阅读的有用参考资料：

- **AI Agents That Matter 重要的 AI 智能体**  
	*Analyzing benchmarks and evaluation practices for real-world AI agent applications.* Available at: [arXiv:2407.01502](https://arxiv.org/abs/2407.01502).  
	分析现实世界 AI 智能体应用的基准测试和评估实践。可在以下网址获取：arXiv:2407.01502。
- **An In-depth Survey of Large Language Model-based Artificial Intelligence Agents  
	基于大型语言模型的人工智能代理深度调研**  
	*Exploring core components such as planning, memory, and tool use in LLM-based agents.* Available at: [arXiv:2309.14365](https://arxiv.org/abs/2309.14365).  
	探索基于 LLM 的代理中的核心组件，如规划、记忆和工具使用。可在 arXiv:2309.14365 获取。
- Blog by CHIP HUYEN: [https://huyenchip.com/2025/01/07/agents.html](https://huyenchip.com/2025/01/07/agents.html)  
	CHIP HUYEN 的博客：https://huyenchip.com/2025/01/07/agents.html
- Building effective agents By ANTHROPIC:[https://www.anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)  
	构建有效代理 由 Anthropic 提供：https://www.anthropic.com/research/building-effective-agents
- LLM Multi-Agent Systems: Challenges and Open Problems:[https://arxiv.org/abs/2402.03578](https://arxiv.org/abs/2402.03578)  
	LLM 多代理系统：挑战与开放问题：https://arxiv.org/abs/2402.03578
- The Rise and Potential of Large Language Model Based Agents: [https://arxiv.org/abs/2309.07864](https://arxiv.org/abs/2309.07864)  
	基于大型语言模型的智能体的崛起与潜力：https://arxiv.org/abs/2309.07864
