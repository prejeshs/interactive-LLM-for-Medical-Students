from pymed import PubMed
from typing import List, Dict, Any
from haystack import component, Document
from haystack.components.generators import HuggingFaceAPIGenerator
from dotenv import load_dotenv
import os
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
import gradio as gr
from haystack.utils import Secret

# Load API Key from .env
load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

# Initialize PubMed API
pubmed = PubMed(tool="Haystack2.0Prototype", email="dummyemail@gmail.com")

def documentize(article):
    return Document(content=article.abstract, meta={'title': article.title, 'keywords': article.keywords})

@component
class PubMedFetcher():
    @component.output_types(articles=List[Document])
    def run(self, queries: list[str]):
        cleaned_queries = queries[0].strip().split('\n')
        articles = []
        try:
            for query in cleaned_queries:
                response = pubmed.query(query, max_results=1)
                documents = [documentize(article) for article in response]
                articles.extend(documents)
        except Exception as e:
            print(e)
            print(f"Couldn't fetch articles for queries: {queries}")
        return {'articles': articles}

@component
class ArticleFormatter():
    @component.output_types(template_variables=Dict[str, Any])
    def run(self, articles: List[Document], question: str):
        formatted_articles = [{"content": doc.content, "title": doc.meta['title'], "keywords": doc.meta.get('keywords', [])} for doc in articles]
        return {'template_variables': {"question": question, "articles": formatted_articles}}

# Initialize HuggingFace Generator
llm = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    token=Secret.from_env_var("HUGGINGFACE_API_KEY"),
    generation_kwargs={"max_new_tokens": 500, "temperature": 0.6, "do_sample": True}
)

# Quiz Generation Prompt
quiz_prompt_template = """
Generate a medical quiz based on the given articles.

For each document, create:
1. A multiple-choice question (MCQ) with four options, focused on **clinical decision-making or disease pathophysiology**. Explain why the correct answer is right.
2. A case-based scenario question.
3. A short-answer question requiring concise medical reasoning.

If there's no relevant content, generate a question based on general medical knowledge.

**Topic:** {{ question }}

**Articles:**
{% for article in articles %}
  {{ article.content }}
  keywords: {{ article.keywords }}
  title: {{ article.title }}
{% endfor %}

**Multiple-choice (MCQ):**  
Q: <clinical or pathophysiology-based question>  
A. <option1>  
B. <option2>  
C. <option3>  
D. <option4>  
**Correct answer:** <correct option>  
**Explanation:** <why the answer is correct, include relevant medical reasoning>  

**Case-based scenario:**  
A 45-year-old male patient presents with <symptoms>. He has a history of <relevant medical history>. Based on the given information, what is the most appropriate next step in management?  
**Answer:** <correct management approach>  

**Short-answer:**  
Q: <Short but conceptually challenging medical question>  
**Answer:** <concise and precise medical response>  

  """


# Initialize Components
fetcher = PubMedFetcher()
formatter = ArticleFormatter()
prompt_builder = PromptBuilder(template=quiz_prompt_template)

# Create Pipeline
pipe = Pipeline()

pipe.add_component("pubmed_fetcher", fetcher)
pipe.add_component("article_formatter", formatter)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)

# Connect Pipeline Components
pipe.connect("pubmed_fetcher.articles", "article_formatter.articles")
pipe.connect("article_formatter.template_variables", "prompt_builder.template_variables")
pipe.connect("prompt_builder.prompt", "llm.prompt")

# Function to Generate Quiz
def generate_quiz(topic):
    output = pipe.run(data={
        "pubmed_fetcher": {"queries": [topic]},
        "article_formatter": {"question": topic},
        "llm": {"generation_kwargs": {"max_new_tokens": 500}}
    })
    
    return output['llm']['replies'][0]

# Gradio Interface
iface = gr.Interface(
    fn=generate_quiz,
    inputs=gr.Textbox(value="Generate a quiz on COVID-19 treatments."),
    outputs="markdown",
    title="Medical Quiz Generator",
    description="Enter a medical topic and get an AI-generated quiz based on research.",
    examples=[
        ["Generate a quiz on COVID-19 treatments."],
        ["Generate a quiz on Autoimmune Disorders."],
        ["Create a quiz about Pneumonia."],
        ["Give me practice questions on Diabetes."],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

# Launch Gradio App
iface.launch(debug=True)
