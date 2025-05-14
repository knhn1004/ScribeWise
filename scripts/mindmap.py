from groq import Groq

# the text to create the mindmap from
with open("text.txt", "r") as file:
    text = file.read().strip()

system_prompt = f"""The following is the syntax for creating mermaid mindmap notation:

1. It should start with the word mindmap, and the first node has to be the "root" node followed by two barkets like:

root((main idea))

2. ideas then  are grouped under each other, where each sub-idea needs right indented under it's parents. 

3. Except for the root node, Ideas and sub-ideas shouldn't include any types of brackets, like curly brackets, brackets, or square brackets

4. Also, you can use ::icon(fa fa-icon-name) from https://fontawesome.com/ to illustrate a node with relevant in a seperate line under the node or subnode for illustation

5. Limit the node to 3 words maximum and the sub-nodes and downwards to 2 words maximum. 

As per the mermaid mindmap notation above, please organize the given information into a mindmap

Only output the mindmap
"""

client = Groq()
chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Create a mindmap for the following text:\n {text}",
        },
    ],
    temperature=0.5,
    stream=False,
)

print(chat_completion.choices[0].message.content)
