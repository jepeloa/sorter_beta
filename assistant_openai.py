from openai import OpenAI

client = OpenAI()

# file = client.files.create(
#     file=open("files/output_san_cristobal/DS_1.pdf", "rb"),
#     purpose="assistants"
# )

# print("FILE ID: ", file.id)


# assistant = client.beta.assistants.create(
#     name="Data visualizer",
#     model="gpt-3.5-turbo-1106",
#     tools=[{"type": "retrieval"}],
#     file_ids=[file.id]
# )

# thread = client.beta.threads.create()

# print("THREAD ID: ", thread.id)

message = client.beta.threads.messages.create(
    thread_id="thread_objI64Kj60FdS2EQrNVB04EN",
    role="user",
    content="Qué beneficios tienen?",
    file_ids=["file-5rQuwUoQzjjKSX8XvTc8W9eJ"]
)

print("MESSAGE ID: ", message.id)

run = client.beta.threads.runs.create(
    thread_id="thread_objI64Kj60FdS2EQrNVB04EN",
    assistant_id="asst_pSbwUNXpra1hWUQ8cKN8in6U",
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}]
)

print("RUN ID: ", run.id)

while True:
    run = client.beta.threads.runs.retrieve(thread_id="thread_objI64Kj60FdS2EQrNVB04EN", run_id=run.id)
    if (run.completed_at):
        print("ESPERANDO...")
        break

messages = client.beta.threads.messages.list(thread_id="thread_objI64Kj60FdS2EQrNVB04EN")
last_message = messages.data[0]
text = last_message.content[0].text.value
print("RESPUESTA: ", text)