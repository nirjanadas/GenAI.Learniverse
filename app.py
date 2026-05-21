from transformers import pipeline
import gradio as gr

# Load Hugging Face Question Answering Model
question_answer = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)


def read_file_content(file_obj):
    """
    Reads uploaded text file content.
    """

    try:
        with open(file_obj.name, "r", encoding="utf-8") as file:
            return file.read()

    except Exception as e:
        return f"Error reading file: {e}"


def get_answer(file, question):
    """
    Generates answer from uploaded document context.
    """

    # Validation
    if file is None:
        return "Please upload a text file."

    if not question.strip():
        return "Please enter a question."

    # Read document content
    context = read_file_content(file)

    # Generate answer
    result = question_answer(
        question=question,
        context=context
    )

    return result["answer"]


# Gradio Interface
demo = gr.Interface(
    fn=get_answer,

    inputs=[
        gr.File(
            label="Upload Text File",
            file_types=[".txt"]
        ),

        gr.Textbox(
            label="Ask a Question",
            placeholder="Enter your question here..."
        )
    ],

    outputs=gr.Textbox(
        label="Answer"
    ),

    title="Document Question Answering System",

    description=(
        "Upload a text document and ask contextual questions "
        "using Hugging Face Transformers and NLP."
    )
)

# Launch Application
demo.launch()