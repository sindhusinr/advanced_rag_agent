from langchain_core.messages import HumanMessage
from langgraph.types import Command

from advanced_rag_agent.graph.rag_graph import graph

#to write
mermaid = graph.get_graph().draw_mermaid()
with open("graph.mmd", "w") as f:
    f.write(mermaid)

def main():

    thread_id = "user_1"

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        if "__interrupt__" in result:

            interrupt_data = result[
                "__interrupt__"
            ][0].value

            print("\nHuman Input Required:")
            print(interrupt_data)

            human_reply = input(
                "\nYour Response: "
            )

            result = graph.invoke(
                Command(
                    resume=human_reply
                ),
                config={
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )

        print("\nANSWER:")

        print(
            result["messages"][-1].content
        )


if __name__ == "__main__":
    main()