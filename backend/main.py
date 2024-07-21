from crewai import Agent, Process, Task, Crew
from textwrap import dedent
# from crewai_tools import (
    # PDFSearchTool,
    # WebsiteSearchTool
# )
from data_magic.data_job import PreProcess
from langchain_cohere import ChatCohere
from decouple import config
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
preprocessor = PreProcess()
# agents
class Agents:
    def __init__(self):
        # Initialize language model
        self.llm = ChatCohere()

    def pdf_agent(self):
        return Agent(
            role="Constitutional Scholar",
            backstory=dedent(
                f"""You are a legal expert who study and analyze constitutions. The people need you."""),
            goal=dedent(
                f"""Uncover any information from Kenyan constitution exceptionally well."""),
            verbose=True,
            llm=self.llm,
            allow_delegation=False
            
        )

    def writer_agent(self):
        return Agent(
            role="Writer",
            backstory=dedent(
                f"""You are good at breaking down and explaining information and writing summaries."""),
            goal=dedent(
                f"""Take the information from the pdf agent and explain it to people without education backgrounds summarize it nicely."""),
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )

# tasks

class Tasks:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def __tip_section(self):
        return "If you do your BEST WORK, You'll be helping a lot of people!"

    def pdf_task(self, agent, question):
        relevant_chunks = preprocessor.query_vector_store(question, self.vector_store)
        return Task(
            description=dedent(
                f"""
            get as munch information as fast as you can, retreived from {relevant_chunks}.
            Use this as what I want to be explained: {question}
            
            {self.__tip_section()}
    
            Make sure to be as accurate as possible. 
        """
            ),
            expected_output="Full analysis.",
            agent=agent,
        )

    def writer_task(self, agent):
        return Task(
            description=dedent(
                f"""Take the input from task 1 and write a summary about it.{self.__tip_section()}"""
            ),
            expected_output="Craft a title, and a brief explained summary in markdown.",
            agent=agent,
        )

# the crew

class OurCrew:
    def __init__(self, question, vector_store):
        self.question = question
        self.vector_store = vector_store

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = Agents()
        tasks = Tasks(self.vector_store)

        # Define your custom agents and tasks here
        pdf_agent = agents.pdf_agent()
        writer_agent = agents.writer_agent()

        # Custom tasks include agent name and variables as input
        task1 = tasks.pdf_task(
            pdf_agent,
            self.question
        )

        task2 = tasks.writer_task(
            writer_agent,
        )

        #custom crew
        crew = Crew(
                    agents=[pdf_agent, writer_agent],
                    tasks=[task1, task2],
                    verbose=True,
                    process=Process.sequential,
                            embedder={
                        "provider": "cohere",
                        "config":{
                            "model": "embed-english-light-v3.0"
                            }
                            }
                )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    vector_store = preprocessor.store_embeddings("data_magic/constitution.pdf")
    print("## Welcome to Katifunza AI")
    print("-------------------------------")
    question = input(dedent("""Enter your question: """))

    custom_crew = OurCrew(question, vector_store)
    result = custom_crew.run()
#     print("########################\n")
#     print(result)
