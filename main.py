from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = DuckDuckGoSearchRun()

# --- STATE ---
class AgentState(TypedDict):
    task: str
    steps: List[str]
    results: List[str]
    final_report: str

# --- AGENT 1: PLANNER ---
def planner(state: AgentState) -> AgentState:
    print("\n🧠 Planner думает...")

    response = llm.invoke(
        f"Ты планировщик задач. Разбей следующую задачу на 3 конкретных шага.\n"
        f"Отвечай только списком шагов, каждый с новой строки, без нумерации.\n\n"
        f"Задача: {state['task']}"
    )

    steps = [s.strip() for s in response.content.strip().split("\n") if s.strip()]
    print(f"📋 Шаги: {steps}")

    return {**state, "steps": steps}

# --- AGENT 2: EXECUTOR (с tool call) ---
llm_with_tools = llm.bind_tools([search_tool])

def executor(state: AgentState) -> AgentState:
    print("\n⚙️ Executor работает...")

    results = []

    for step in state["steps"]:
        print(f"\n🔧 Выполняю шаг: {step}")
        messages = [
            HumanMessage(
                content=f"Ты исполнитель задач. Используй поиск чтобы найти реальную информацию для выполнения этого шага.\n\nШаг: {step}"
            )
        ]

        # Первый вызов LLM — он решает использовать tool или нет
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Если LLM решил вызвать tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"🔍 Ищу в интернете: {tool_call['args']['query']}")

                # Реально выполняем поиск
                search_result = search_tool.invoke(tool_call['args']['query'])

                # Кладём результат поиска обратно в messages
                messages.append(
                    ToolMessage(
                        content=search_result,
                        tool_call_id=tool_call['id']
                    )
                )

            # Второй вызов LLM — теперь он видит результаты поиска и формулирует ответ
            final_response = llm_with_tools.invoke(messages)
            result = final_response.content.strip()
        else:
            # LLM решил не искать, ответил сам
            result = response.content.strip()

        results.append(f"{step} → {result}")
        print(f"✅ Готово")

    return {**state, "results": results}

# --- AGENT 3: REPORTER ---
def reporter(state: AgentState) -> AgentState:
    print("\n📝 Reporter пишет отчёт...")

    results_text = "\n".join(state["results"])
    response = llm.invoke(
        f"Ты составляешь финальный отчёт. На основе выполненных шагов напиши краткое резюме.\n\n"
        f"Исходная задача: {state['task']}\n\n"
        f"Выполненные шаги:\n{results_text}"
    )

    return {**state, "final_report": response.content.strip()}

# --- GRAPH ---
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("reporter", reporter)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "reporter")
graph.add_edge("reporter", END)

app = graph.compile()

# --- RUN ---
if __name__ == "__main__":
    result = app.invoke({
        "task": "Автоматизировать онбординг новых сотрудников в компании",
        "steps": [],
        "results": [],
        "final_report": ""
    })

    print("\n" + "="*50)
    print("📊 ФИНАЛЬНЫЙ ОТЧЁТ:")
    print("="*50)
    print(result["final_report"])