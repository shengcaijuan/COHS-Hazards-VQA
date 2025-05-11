import json
import os
from colorama import Fore
from langchain_core.vectorstores import VectorStoreRetriever
from openai import OpenAI
import base64
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import tkinter as tk
from tkinter import filedialog
import sys

class ImageProcessor:
    @staticmethod
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

class InitialRetrievalQueryExtraction:
    def __init__(self, model, api_key, base_url, image_base64):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.image_base64 = image_base64

    def get_object_query(self) -> list:
        # Preset object anchors
        preset_objects = ["Vertical reserved opening", "Horizontal reserved opening", "Edge", "Scaffolding",
                          "Electrical distribution box", "Electric wire", "Electric welding machine", "Gas cylinder",
                          "Crane", "Opening or Edge guardrails", "Hanging basket",
                          "Hanging basket suspension mechanism",
                          "Mechanical transmission part", "Foundation pit(Trench)", "Fall arrest safety flat net",
                          "Operation platform"
                          ]
        # Output template
        object_anchor_output_sample = """
        ["opening","Edge"]
        """
        object_anchor = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant capable of detecting various elements from construction site images."
                            f"Please determine whether the elements in {preset_objects} are included in the image."
                            f"If so, please output the elements contained in {preset_objects}. "
                            f"Please ensure that the output is simple, only including the elements in {preset_objects} and avoiding redundant output."
                            f"Please ensure that the output is in the format of a Python list like {object_anchor_output_sample}.Please only output a Python list and avoid other output.Please pay attention to using double quotes instead of single quotes."
                 },

                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                ]}
            ],
            max_tokens=300,
            temperature=0
        ).choices[0].message.content

        object_query_list = json.loads(object_anchor) 
        return object_query_list

    def get_operation_query(self, no_operation=None) -> list:
        # Preset operation anchors
        preset_operations = ["Manual Earth excavation", "Mechanical earth excavation", "Earth blasting",
                             "Foundation compaction",
                             "Rebar tying", "Concrete pouring", "Concrete vibration", "Masonry operation",
                             "Plastering work",
                             "Waterproofing operation", "Mechanical cutting operation", "Carrying materials",
                             "Chiseling or chipping work",
                             "Welding operation", "Formwork operation", "Gas cutting operation", "Lifting operation",
                             "Hanging basket operation",
                             "painting operation", "Work on scaffolding", "Operation near the edge",
                             "Climbing operation", "Work at heights"
                             ]

        operation_anchor = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant capable of identifying the ongoing operations in construction images."
                            f"Please determine whether the construction operations in {preset_operations} are included in the image."
                            f"If so, please output the most matching one construction operation contained in {preset_operations}. "
                            f"Please ensure that the output is simple, only including the elements in {preset_operations} and avoiding redundant output."
                            f"Please note that some of the construction images don't contain any operations. If there isn't a construction operation in image, please only output {no_operation}"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                ]}
            ],
            max_tokens=300,
            temperature=0
        ).choices[0].message.content.strip()

        if operation_anchor in preset_operations:
            return [operation_anchor]
        else:
            return []

class FAISSDatabaseManager:
    @staticmethod
    def save_faiss(index, file_path: str):
        index.save_local(file_path)
        print(f"FAISS index saved at {file_path}")

    @staticmethod
    def load_faiss(file_path: str):
        if os.path.exists(file_path):
            print(f"Loading FAISS index from {file_path}")
            # Create an embedding model
            embeddings = OpenAIEmbeddings(model="your_embedding_model")
            return FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(f"FAISS index not found at {file_path}")
        return None

    @staticmethod
    def delete_existing_faiss(folder_path: str):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted files: {file_path}")

    # Define the function to generate a temporary FAISS database
    def create_temp_faiss_db(self, doc_object_list: list, doc_operation_list: list, temp_faiss_path:str) -> FAISS:
        # Store temporary db text block list
        temp_db_text_splits = []
        if doc_object_list:
            for doc_temp in doc_object_list:
                temp_db_text_splits.append(doc_temp)
        if doc_operation_list:
            for doc_temp in doc_operation_list:
                temp_db_text_splits.append(doc_temp)
        # Modify to convert each string into a Document object
        temp_document = [Document(page_content=content) for content in temp_db_text_splits]
        # Create and save a temporary database
        temp_faiss_db = FAISS.from_documents(temp_document, OpenAIEmbeddings(model="your_embedding_model"))
        self.save_faiss(temp_faiss_db, temp_faiss_path)

        return temp_faiss_db

class DefineRetriever:
    def __init__(self):
        self.load_faiss = FAISSDatabaseManager.load_faiss

    def define_initial_retriever(self, object_faiss_path: str, operation_faiss_path: str) -> tuple[VectorStoreRetriever, VectorStoreRetriever]:
        # Load the FAISS database
        safe_object_rules_db = self.load_faiss(object_faiss_path)  # object safety guidelines
        safe_operation_rules_db = self.load_faiss(operation_faiss_path)  # operation safety guidelines
        # Define the retriever for the object
        retriever_object = safe_object_rules_db.as_retriever()
        # Define the retriever for the operation
        retriever_operation = safe_operation_rules_db.as_retriever()
        return retriever_object, retriever_operation

    def define_secondary_retriever(self, temp_faiss_path: str) -> VectorStoreRetriever:
        # Load the FAISS database
        temp_faiss_db = self.load_faiss(temp_faiss_path)
        # Define the retriever for the temp db
        secondary_retriever = temp_faiss_db.as_retriever()
        return secondary_retriever

class DualStageRetrieval:
    @staticmethod
    # Define a function to retrieve text from object FAISS db
    def retrieve_from_object_faiss_db(object_query: list, retriever_object: VectorStoreRetriever) -> list:
        # Define a variable to store retrieval text
        doc_object_list = []
        # Retrieve object text
        if object_query:
            for sub_query in object_query:
                try:
                    docs1 = retriever_object.invoke(sub_query)
                    context_object = str(docs1[0].page_content)
                    doc_object_list.append(context_object)
                except Exception as e:
                    print(f"Error occurred: {e}")
                    continue
        return doc_object_list

    @staticmethod
    # Define a function to retrieve text from operation FAISS db
    def retrieve_from_operation_faiss_db(operation_query: list, retriever_operation: VectorStoreRetriever) -> list:
        # Define a variable to store retrieval text
        doc_operation_list = []
        if operation_query:
            # Retrieve operation text
            docs2 = retriever_operation.invoke(operation_query[0])
            context_operation = str(docs2[0].page_content)
            doc_operation_list.append(context_operation)
            return doc_operation_list

    @staticmethod
    # Define a function to retrieve text from temp faiss
    def retrieve_from_temp_faiss_db(query: str, secondary_retriever) -> str:
        doc_list = secondary_retriever.invoke(query)
        doc = str(doc_list[0].page_content)
        return doc

class SecondaryRetrievalQueryGeneration:
    def __init__(self, model, api_key, base_url):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    # Define the secondary retrieval queries generation agent function
    def agent(self, user_prompt: str, object_query: list, operation_query: list,
              doc_object_list: list, doc_operation_list: list) -> list:
        # Build initial retrieval text dict
        temp_knowledge = {}

        if object_query:
            for query, doc in zip(object_query, doc_object_list):
                temp_knowledge[query] = doc

        if operation_query:
            query = operation_query[0]
            temp_knowledge[query] = doc_operation_list[0]

        # Define the output template
        answer_sample = ["Opening", "Edge", "Welding operation"]
        # Define system prompt
        system_prompt = f"""
        Act as a construction safety expert. Analyze the {user_prompt} and determine which knowledge entries are required to answer it.
        **Instructions**
        Please read the content of the values in the {temp_knowledge} carefully. 
        If the content of a certain value is needed to answer the user's question, then output the key corresponding to this value.
        Please refer to {answer_sample} for the output format. Please only output the python list and don't output any redundant content.
        For questions asking about the unsafe behaviors of workers, only select the texts that contain the content related to "operation".

        **Examples**
        Entries:
        {{'Opening': 'Safety guidelines for Vertical opening:...', 'Edge': 'Safety guidelines for edge:...', 'Welding operation':'Safety guidelines for Welding Operation:...'}}
        Output: ["Opening"]

        Q1: "What safety measures are needed for vertical openings?"
        Output: ["Opening"]

        Q2: "What PPE is required for welding?"
        Output: ["Welding operation"]

        Q3: "What occupational health and safety hazards exist in the image?"
        Entries: 
        Output: ["Opening", "Edge", "Welding operation"]

        Q4: "What unsafe states of objects exist in the construction images?"
        Entries: 
        Output: ["Opening", "Edge"]

        Q5: "What unsafe states of objects exist in the construction images?"
        Output: ["Opening", "Edge"]

        Q6: "What are the unsafe behaviors of the workers in the construction images?"
        Output: ["Welding operation"]
        """

        secondary_retrieval_query = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": f"{system_prompt}"
                 }
            ],
            temperature=0
        ).choices[0].message.content.strip()

        secondary_search_query_list = json.loads(secondary_retrieval_query)
        return secondary_search_query_list

class COHSHazardDetection:
    def __init__(self, image_base64, model, api_key, base_url, user_question):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.image_base64 = image_base64
        self.secondary_retrieval = DualStageRetrieval.retrieve_from_temp_faiss_db
        self.user_question = user_question

    # Define a function to identify object hazards
    def object_hazards_detection(self, object_query_list: list, operation_query_list: list, secondary_retriever):
        # Define output template
        answer_sample1 = """
        1.Sweeping rod isn't installed at the bottom of the scaffold.
        2......
        3......
        4......
        5......
        .......
        """

        object_hazards_list = []
        num_object_hazards = 0

        for i, query in enumerate(object_query_list):
            try:
                # Retrieve the COHS rules of objects for knowledge enhancement of MLLMs.
                context_object = self.secondary_retrieval(query, secondary_retriever)
                # Long text chunking
                if query == "Scaffolding" or query == "Electrical distribution box" or query == "Gas cylinder":
                    context_operation_all_splits = context_object.split('|')
                    if not operation_query_list:
                        num_descriptions = 2
                    else:
                        num_descriptions = 1
                    for context_chunk in context_operation_all_splits:
                        response_object = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system",
                                 "content": f"You are an assistant capable of identifying safety hazards at construction sites."
                                            f"Your task is to detect construction safety hazards from the images only based on the safety guidelines in the {context_chunk} instead of your inherent knowledge."
                                            f"Only output the content related to {context_chunk}. Don't output the content related to human and unrelated to {context_chunk}."
                                            f"Please refer to the format of {answer_sample1} for output strictly."
                                            f"Please note that the number of potential safety hazards in each picture is not fixed, and the content in {answer_sample1} is just a reference for the output format."
                                            f"Please note that some of the safety guidelines in the {context_chunk} may not be applicable. You should select the relevant safety rules based on the image."
                                            f"The maximum number of hazard descriptions that can be output each time is {num_descriptions}."
                                            "Avoid answering the content that you are not sure about and that is based on imagination."
                                 },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text",
                                         "text": f"{self.user_question}"
                                         },
                                        {"type": "image_url",
                                         "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                                    ]
                                }
                            ],
                            max_tokens=400,
                            temperature=0.2
                        )

                        object_hazards = response_object.choices[0].message.content
                        if object_hazards == "None" or object_hazards == "None.":
                            num_object_hazards += 0
                        else:
                            num_object_hazards += 1
                            object_hazards_list.append(object_hazards + "\n")
                # Short text not chunking
                else:
                    if not operation_query_list:
                        num_descriptions = 3
                    else:
                        num_descriptions = 2

                    response_object = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system",
                             "content": f"You are an assistant capable of identifying safety hazards at construction sites."
                                        f"Your task is to detect construction safety hazards from the images only based on the safety guidelines in the {context_object} instead of your inherent knowledge."
                                        f"Only output the content related to {context_object}. Don't output the content related to human and unrelated to {context_object}."
                                        f"Please refer to the format of {answer_sample1} for output strictly."
                                        f"Please note that the number of potential safety hazards in each picture is not fixed, and the content in {answer_sample1} is just a reference for the output format."
                                        f"Please note that some of the safety guidelines in the {context_object} may not be applicable. You should select the relevant safety rules based on the image."
                                        f"The maximum number of hazard descriptions that can be output each time is {num_descriptions}"
                                        "Please don't over-imagine and output the content that is not included in the picture. "
                                        "Avoid answering the content that you are not sure about and that is based on imagination."
                             },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text",
                                     "text": f"{self.user_question}"
                                     },
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                                ]
                            }
                        ],
                        max_tokens=400,
                        temperature=0.2
                    )

                    object_hazards = response_object.choices[0].message.content
                    if object_hazards == "None" or object_hazards == "None.":
                        num_object_hazards += 0
                    else:
                        num_object_hazards += 1
                        object_hazards_list.append(object_hazards + "\n")

            except Exception as e:
                print(f"Error occurred: {e}")
                continue

        return "".join(object_hazards_list), num_object_hazards

    # Define a function to identify operation hazards
    def operation_hazards_detection(self, operation_query: list, secondary_retriever):
        # Define output template
        answer_sample2 = """
        1.Workers don't wear safety harnesses.
        2......
        3......
        4......
        5......
        .......
        """

        operation_hazards = ""
        num_operation_hazards = 0

        # Retrieve the COHS rules of operations for knowledge enhancement of MLLMs.
        context_operation = self.secondary_retrieval(operation_query[0], secondary_retriever)

        # Short text not chunking
        if operation_query == "Carrying materials" or "Chiseling or chipping work" or "Concrete vibration" or "Earth blasting" or "operation near the edge":
            response_operation = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": f"You are an assistant capable of identifying construction safety and occupational health hazards at construction sites. "
                                f"Your task is to detect construction safety and occupational health hazards from the images only based on the guidelines in the {context_operation} instead of your inherent knowledge."
                                f"Please provide accurate and concise answers point by point. Only output the content related to the content in {context_operation}."
                                f"Please refer to the format of {answer_sample2} for output strictly."
                                f"Please note that the number of safety and occupational health hazards in each picture is not fixed, and the content in {answer_sample2} is just a reference for the output format."
                                f"Please note that some of the safety guidelines in the {context_operation} may not be applicable. You should select the relevant safety rules based on the image."
                                f"Avoid answering the content that you are not sure about and that is based on imagination."
                     },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": f"{self.user_question}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                        ]
                    }
                ],
                max_tokens=400,
                temperature=0.2
            )

            operation_hazards = response_operation.choices[0].message.content

            if operation_hazards == "None" or operation_hazards == "None.":
                num_operation_hazards += 0
            else:
                num_operation_hazards += 1

        # Long text chunking
        else:
            context_operation_all_splits = context_operation.split('|')
            for context_operation in context_operation_all_splits:
                response_operation = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": f"You are an assistant capable of identifying construction safety and occupational health hazards at construction sites. "
                                    f"Your task is to detect construction safety and occupational health hazards from the images only based on the guidelines in the {context_operation} instead of your inherent knowledge."
                                    f"Please provide accurate and concise answers point by point. Only output the content related to the content in {context_operation}."
                                    f"Please refer to the format of {answer_sample2} for output strictly."
                                    f"Please note that the number of safety and occupational health hazards in each picture is not fixed, and the content in {answer_sample2} is just a reference for the output format."
                                    f"Please note that some of the safety guidelines in the {context_operation} may not be applicable. You should select the relevant safety rules based on the image."
                                    f"Avoid answering the content that you are not sure about and that is based on imagination."
                         },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text",
                                 "text": f"{self.user_question}."},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}
                            ]
                        }
                    ],
                    max_tokens=400,
                    temperature=0.2
                )

                operation_hazards_based_on_one_chunk = response_operation.choices[0].message.content

                if operation_hazards_based_on_one_chunk == "None" or operation_hazards_based_on_one_chunk == "None.":
                    num_operation_hazards += 0
                else:
                    num_operation_hazards += 1
                    operation_hazards += operation_hazards_based_on_one_chunk + "\n"

        return operation_hazards, num_operation_hazards

    # Define a function to identify general hazards
    def generality_hazards_detection(self):
        # Define output template
        answer_sample3 = """
        1.Worker doesn't wear a safety helmet.
        2......
        3......
        4......
        5......
        .......
        """
        num_general_safety_hazards = 0
        general_safety_guidelines = """1.Workers must wear safety helmets on the construction site.
        2.Worker should wear safety helmet correctly: The chin straps of the safety helmets must be fastened. It is prohibited for worker to wear other hat under the safety helmet.
        3.Workers should wear reflective vests on the construction site.
        4.Smoking is prohibited on the construction site."""
        general_safety_guidelines_detection_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": f"You are an assistant whose task is to detect safety hazards from images based on the guidelines in the {general_safety_guidelines} instead of your inherent knowledge"
                            f"Please provide accurate and concise answers point by point. Each point of description should be on a new line."
                            f"Only output the content related to {general_safety_guidelines} and it is prohibited to output other content unrelated to {general_safety_guidelines}."
                            f"Please refer to {answer_sample3} for output strictly."
                 },
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}},
                    {"type": "text",
                     "text": f"{self.user_question}"},
                ]
                 }
            ],
            temperature=0.2
        )

        safety_hazards_generality = general_safety_guidelines_detection_response.choices[0].message.content
        if safety_hazards_generality == "None" or safety_hazards_generality == "None.":
            num_general_safety_hazards += 0
            return None, num_general_safety_hazards
        else:
            num_general_safety_hazards += 1
            return safety_hazards_generality, num_general_safety_hazards

    @staticmethod
    # Define the renumbering function
    def renumber_sentences(original_string: str):
        lines = []
        for line in original_string.splitlines():
            stripped_line = line.strip()
            if stripped_line:
                if not stripped_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.')):
                    stripped_line = f"1. {stripped_line}"
                lines.append(stripped_line)
        new_lines = []
        count = 1
        for line in lines:
            new_line = f"{count}. {line.split('.', 1)[1].strip()}"
            new_lines.append(new_line)
            count += 1
        new_string = '\n'.join(new_lines)
        return new_string

    # Define image COHS hazards identification Function
    def integrate_cohs_hazards(self, secondary_object_query: list, secondary_operation_query: list, secondary_retriever) -> str:
        cohs_hazards = None
        # No operation scenario
        if not secondary_operation_query:
            hazards_object_no_renumber, num_object_hazards = self.object_hazards_detection(secondary_object_query, secondary_operation_query, secondary_retriever)
            # No unsafe states of object
            if num_object_hazards == 0:
                print(Fore.GREEN + "There is no safety hazards of object in the construction image." + Fore.RESET)
            # Unsafe states of object
            else:
                hazards_object = self.renumber_sentences(hazards_object_no_renumber)  # renumbering
                # safety hazards detection
                cohs_hazards = f"\n{hazards_object}"
        # Operation scenario
        else:
            hazards_object, num_object_hazards = self.object_hazards_detection(secondary_object_query, secondary_operation_query, secondary_retriever)
            hazards_operation, num_operation_hazards = self.operation_hazards_detection(secondary_operation_query, secondary_retriever)
            hazards_generality, num_general_hazards = self.generality_hazards_detection()
            cohs_hazards_list = []
            num_hazards_list = [num_object_hazards, num_operation_hazards, num_general_hazards]
            safety_hazards_list = [hazards_object, hazards_operation, hazards_generality]
            for i, num_hazards in enumerate(num_hazards_list):
                if num_hazards == 0:
                    pass
                else:
                    cohs_hazards_list.append(safety_hazards_list[i])
            cohs_hazards = self.renumber_sentences("\n".join(cohs_hazards_list))  # renumbering
        return cohs_hazards

class COHSHazardsVQAAssistant:
    def __init__(self, model, api_key, base_url, user_question):
        self.image_path = self.upload_image()
        image_base64 = ImageProcessor.encode_image(self.image_path)
        initialretrievalqueryextraction = InitialRetrievalQueryExtraction(model, api_key, base_url, image_base64)
        self.get_object_query = initialretrievalqueryextraction.get_object_query
        self.get_operation_query = initialretrievalqueryextraction.get_operation_query
        define_retriever = DefineRetriever()
        self.define_initial_retriever = define_retriever.define_initial_retriever
        self.define_secondary_retriever = define_retriever.define_secondary_retriever
        retrieval = DualStageRetrieval()
        self.initial_retrieval_object = retrieval.retrieve_from_object_faiss_db
        self.initial_retrieval_operation = retrieval.retrieve_from_operation_faiss_db
        faiss_manager = FAISSDatabaseManager()
        self.create_temp_faiss_db = faiss_manager.create_temp_faiss_db
        secondaryretrievalquerygeneration = SecondaryRetrievalQueryGeneration(model, api_key, base_url)
        self.secondary_query_generation = secondaryretrievalquerygeneration.agent
        answer = COHSHazardDetection(image_base64, model, api_key, base_url,user_question)
        self.answer_open_question = answer.integrate_cohs_hazards

    @staticmethod
    def upload_image() -> str:
        # Hide the tkinter main window
        root = tk.Tk()
        root.withdraw()
        # Set file type filtering (only show image formats)
        file_types = [
            ("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp"),
            ("All Files", "*.*")
        ]
        # Evoke the file selection dialog box
        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=file_types
        )

        if not file_path:
            print("The user cancels the file selection")
            sys.exit(1)

        # Process the selected files (e.g., print the path or read the file)
        print(f"The path of the selected image file：{file_path}")
        with open(file_path, "rb") as f:
            image_data = f.read()
            print("The image file was successfully read, size:", len(image_data), "Byte")
        return file_path

    def qa_assistant(self, user_question):
        """initial retrieval"""
        # Extract initial retrieval queries
        print(f"Visual semantic anchors extraction start running.")
        object_query = self.get_object_query()
        operation_query = self.get_operation_query()
        print(f"Visual semantic anchors have been extracted. Object queries:{object_query}; Operation query:{operation_query}")
        print(f"Initial retrieval starts running.")
        # FAISS db file path
        object_faiss_db_path = "your_object_faiss_db_path"
        operation_faiss_db_path = "your_operation_faiss_db_path"
        # defining initial retriever
        initial_retriever_object, initial_retriever_operation = self.define_initial_retriever(object_faiss_db_path, operation_faiss_db_path)
        # retrieve image-specific text chunks
        context_object = self.initial_retrieval_object(object_query, initial_retriever_object)
        context_operation = self.initial_retrieval_operation(operation_query, initial_retriever_operation)
        print("Initial retrieval has been finished.")
        # Define the storage location for temp faiss db
        temp_faiss_path = "your_temp_faiss_db_path"
        # construting temp db
        self.create_temp_faiss_db(context_object, context_operation, temp_faiss_path)
        print("Temp knowledge base has been constructed.")
        """secondary retrieval"""
        # secodary retrieval queries generation
        print("Secondary retrieval queries generation starts running.")
        secondary_query_list = self.secondary_query_generation(user_question, object_query, operation_query, context_object, context_operation)
        secondary_object_query = []
        secondary_operation_query = []
        for query in secondary_query_list:
            if query in operation_query:
                secondary_operation_query.append(query)
            else:
                secondary_object_query.append(query)
        print(f"Secondary retrieval queries generation has been finished.")
        print(f"Secondary object query:{secondary_object_query}; Secondary operation query：{secondary_operation_query}")
        secondary_retriever = self.define_secondary_retriever(temp_faiss_path)
        # hazards detection
        print("Start answering")
        answer = self.answer_open_question(secondary_object_query, secondary_operation_query, secondary_retriever)
        return answer

# main
if __name__ == "__main__":
    # OpenAI API Key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    # user question
    no_hazard = None
    user_question = f"What occupational health and safety hazards exist in the construction image? if there are, please only describe the hazards in detail; otherwise, please respond with {no_hazard}."
    VQA_assistant = COHSHazardsVQAAssistant(model="model", api_key=api_key, base_url=base_url, user_question=user_question)
    response = VQA_assistant.qa_assistant(user_question)
    print(f"Answer:{response}")
