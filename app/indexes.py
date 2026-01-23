from langchain_community.vectorstores import Chroma

def load_indexes(persist_dir="vectorstore"):
    return {
        "intent": Chroma(persist_directory=f"{persist_dir}/intent"),
        "schema": Chroma(persist_directory=f"{persist_dir}/schema"),
        "examples": Chroma(persist_directory=f"{persist_dir}/examples")
    }
